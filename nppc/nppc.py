import os
import inspect
import datetime

import numpy as np
import json
import tqdm.auto as tqdm
import plotly
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

from . import auxil
from . import networks
from .base_model import BaseModel, BaseTrainer
from .restoration import RestorationModel


## Model
## =====
class NPPCModel(BaseModel):
    def __init__(
            self,
            restoration_model_folder,
            net_type,
            pre_net_type='none',
            n_dirs=5,
            lr=1e-4,
            second_moment_loss_lambda=1e0,
            second_moment_loss_grace=1000,
            weight_decay=0.,
            lr_lambda=None,
            random_seed = 42,
            device='cpu',
        ):
        input_args = dict(
            restoration_model_folder=restoration_model_folder,
            pre_net_type=pre_net_type,
            net_type=net_type,
            n_dirs=n_dirs,
            lr=lr,
            second_moment_loss_grace=second_moment_loss_grace,
            second_moment_loss_lambda=second_moment_loss_lambda,
            weight_decay=weight_decay,
            lr_lambda=lr_lambda,
            random_seed=random_seed,
        )
        super().__init__(input_args=input_args, device=device)

        self.n_dirs = n_dirs
        self.second_moment_loss_grace = second_moment_loss_grace
        self.second_moment_loss_lambda = second_moment_loss_lambda

        ## Load restoration model
        ## ----------------------
        self.restoration_model = RestorationModel.load(os.path.join(restoration_model_folder, 'checkpoint.pt'), device=device)
        self.restoration_model.net.requires_grad_ = False

        self.data_module = self.restoration_model.data_module
        self.x_shape = self.restoration_model.x_shape
        self.x_distorted_shape = self.restoration_model.x_distorted_shape

        ## Set parametric model
        ## --------------------
        upscale_factor = self.x_shape[-1] // self.x_distorted_shape[-1]

        if pre_net_type == 'none':
            if upscale_factor == 1:
                pre_net = None
            else:
                pre_net = nn.Upsample(scale_factor=upscale_factor, mode='nearest')
            pre_out_channels = self.x_distorted_shape[0]

        elif pre_net_type == 'res_cnn':
            ## EDSR
            pre_out_channels = 64
            pre_net = networks.ResCNN(
                channels_in=self.x_distorted_shape[0],
                channels_out=pre_out_channels,
                channels_hidden=64,
                n_blocks=16,
                upscale_factor=self.restoration_model.upscale_factor,
            )

        else:
            raise Exception(f'Unsupported net_type: "{pre_net_type}"')

        if net_type == 'unet':
            ## Vanilla U-Net
            base_net = networks.UNet(
                in_channels=pre_out_channels + self.x_shape[0],
                out_channels=self.x_shape[0] * n_dirs,
                channels_list=(32, 64, 128),
                bottleneck_channels=256,
                n_blocks=1,
                n_blocks_bottleneck=2,
                min_channels_decoder=64,
            )

        elif net_type == 'unet2':
            base_net = networks.UNet(
                channels_in=pre_out_channels + self.x_shape[0],
                channels_out=self.x_shape[0] * n_dirs,
                channels_list=(32, 64, 128, 256, 512),
                n_blocks_list=(2, 2, 2, 2, 2),
                min_channels_decoder=64,
            )
            pad_base_size = 2 ** 4

        elif net_type == 'res_unet':
            ## DDPM for 256x256
            base_net = networks.ResUNet(
                in_channels=pre_out_channels + self.x_shape[0],
                out_channels=self.x_shape[0] * n_dirs,
                channels_list=(64, 64, 128, 128, 256, 256),
                bottleneck_channels=512,
                downsample_list=(False, True, True, True, True, True),
                attn_list=(False, False, False, False, True, False),
                n_blocks=2,
                n_groups=8,
                attn_heads=1,
            )

        else:
            raise Exception(f'Unsupported net_type: "{net_type}"')

        net = PCWrapper(
            net=base_net,
            pre_net=pre_net,
            n_dirs=n_dirs,
            offset=self.data_module.mean,
            scale=self.data_module.std,
            project_func=self.restoration_model.project,
            pre_pad_base_size=None if (pre_net is None) else pre_net.max_scale_factor,
            pad_base_size=base_net.max_scale_factor,
        )

        ## Set network wrapper (optimizer, scheduler, ema & ddp)
        self.net = auxil.NetWrapper(
            net,
            optimizer_type='adam',
            optimizer_params=dict(lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay),
            lr_lambda=lr_lambda,
            device=self.device,
            ddp_active=self.ddp.is_active,
        )

    def process_batch(self, batch):
        with torch.no_grad():
            x_org, x_distorted = self.restoration_model.process_batch(batch)
            x_restored = self.restoration_model.restore(x_distorted)
        return x_org, x_distorted, x_restored

    def get_dirs(self, x_distorted, x_restored, use_best=True, **kwargs):
        w_mat = self.net(x_distorted, x_restored, use_best=use_best, **kwargs)
        return w_mat


def gram_schmidt(x):
    x_shape = x.shape
    x = x.flatten(2)

    x_orth = []
    proj_vec_list = []
    for i in range(x.shape[1]):
        w = x[:, i, :]
        for w2 in proj_vec_list:
            w = w - w2 * torch.sum(w * w2, dim=-1, keepdim=True)
        w_hat = w.detach() / w.detach().norm(dim=-1, keepdim=True)

        x_orth.append(w)
        proj_vec_list.append(w_hat)

    x_orth = torch.stack(x_orth, dim=1).view(*x_shape)
    return x_orth


class PCWrapper(nn.Module):
    def __init__(
            self,
            net,
            n_dirs,
            pre_net=None,
            offset=None,
            scale=None,
            project_func=None,
            pad_base_size=None,
            pre_pad_base_size=None,
        ):
        super().__init__()

        self.net = net
        self.pre_net = pre_net
        self.n_dirs = n_dirs
        self.offset = offset
        self.scale = scale
        self.project_func = project_func
        self.pre_pad_base_size = pre_pad_base_size
        self.pad_base_size = pad_base_size

    @staticmethod
    def _get_padding(x, base_size):
        s = base_size
        _, _, height, width = x.shape
        if (s is not None) and ((height % s != 0) or (width % s != 0)):
            pad_h = height % s
            pad_w = width % s
            padding = torch.tensor((pad_h // 2, pad_h // 2, pad_w // 2, pad_w // 2))
        else:
            padding = None
        return padding

    def forward(self, x_distorted, x_restored):
        if self.offset is not None:
            x_distorted = x_distorted - self.offset
            x_restored = x_restored - self.offset
        if self.scale is not None:
            x_distorted = x_distorted / self.scale
            x_restored = x_restored / self.scale

        ## Pre-process distorted image
        ## ---------------------------
        if self.pre_net is None:
            x = x_distorted
        else:
            padding = self._get_padding(x_distorted, self.pre_pad_base_size)
            if padding is not None:
                x_distorted = F.pad(x_distorted, tuple(padding))
            x = self.pre_net(x_distorted)

            if padding is not None:
                x = F.pad(x, tuple(-padding))  # pylint: disable=invalid-unary-operand-type

        ## Process both images
        ## -------------------
        x = torch.cat((x, x_restored), dim=1)

        padding = self._get_padding(x, self.pad_base_size)
        if padding is not None:
            x = F.pad(x, tuple(padding))

        w_mat = self.net(x)
        if self.scale is not None:
            w_mat = w_mat * self.scale
        if padding is not None:
            w_mat = F.pad(w_mat, tuple(-padding))  # pylint: disable=invalid-unary-operand-type
        w_mat = w_mat.unflatten(1, (self.n_dirs, w_mat.shape[1] // self.n_dirs))
        if self.project_func is not None:
            w_mat = w_mat.flatten(0, 1)
            w_mat = self.project_func(w_mat)
            w_mat = w_mat.unflatten(0, (w_mat.shape[0] // self.n_dirs, self.n_dirs))
        w_mat = gram_schmidt(w_mat)
        # w_mat = w_mat / w_mat.flatten(3).shape[-1] ** 0.5

        return w_mat

## Trainer
## =======
class NPPCTrainer(BaseTrainer):
    def __init__(self, *args,
            max_benchmark_samples=256,
            **kwargs):

        super().__init__(*args, **kwargs)
        self.max_benchmark_samples = max_benchmark_samples

    def base_step(self, batch):
        model = self.model

        dim = np.prod(model.x_shape)

        x_org, x_distorted, x_restored = model.process_batch(batch)

        w_mat = model.get_dirs(x_distorted, x_restored, use_best=False, use_ddp=True)

        w_mat_ = w_mat.flatten(2)
        w_norms = w_mat_.norm(dim=2)
        w_hat_mat = w_mat_ / w_norms[:, :, None]

        err = (x_org - x_restored).flatten(1)

        ## Normalizing by the error's norm
        ## -------------------------------
        err_norm = err.norm(dim=1)
        err = err / err_norm[:, None]
        w_norms = w_norms / err_norm[:, None]

        ## W hat loss
        ## ----------
        err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)
        reconst_err = 1 - err_proj.pow(2).sum(dim=1)

        ## W norms loss
        ## ------------
        second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)

        second_moment_loss_lambda = -1 + 2 * model.net.step / model.second_moment_loss_grace
        second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1) ,1e-6)
        second_moment_loss_lambda *= model.second_moment_loss_lambda
        objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()

        ## Store log
        ## ---------
        log = dict(
            x_org=x_org,
            x_distorted=x_distorted,
            x_restored=x_restored.detach(),
            w_mat=w_mat.detach(),

            err_mse=err_norm.detach().pow(2) / dim,
            err_proj=err_proj.detach(),
            w_norms=w_norms.detach(),
            reconst_err=reconst_err.detach(),
            second_moment_mse=second_moment_mse.detach(),

            objective=objective.detach(),
        )
        return objective, log

    ## Logging
    ## -------
    @staticmethod
    def cat_logs(logs):
        log = {}
        weights = torch.tensor([log_['x_org'].shape[0] for log_ in logs]).float()
        weights /= weights.sum()
        for key in logs[0].keys():
            if key in ('x_org', 'x_distorted', 'x_restored', 'w_mat', 'err_mse', 'err_proj', 'w_norms', 'reconst_err', 'second_moment_mse'):
                log[key] = torch.cat([log_[key] for log_ in logs], dim=0)
            elif key in ('objective',):
                log[key] = (torch.tensor([log_[key] for log_ in logs]) * weights).sum()
            else:
                raise Exception(f'Unknown key {key}')
        return log

    def init_training_data(self, random_seed=0):
        model = self.model

        training_data = {}

        training_data['general'] = {}

        ## Prepare running logs
        ## --------------------
        training_data['logs'] = {f'{key}_{field}': [] for field in ('fixed', 'valid', 'fullv')
                                                      for key in ('step', 'lr', 'objective', 'err_mse',
                                                                  'reconst_err', 'second_moment_mse',
                                                      ) \
                                                      + tuple(f'err_proj_square_{k}' for k in range(model.n_dirs))
                                                      + tuple(f'w_norms_square_{k}' for k in range(model.n_dirs))
        }

        ## Prepare summary
        ## ---------------
        batch = self.fixed_batch
        with auxil.EncapsulatedRandomState(random_seed):
            x_org, x_distorted, x_restored = model.process_batch(batch)
        summary = torchinfo.summary(model.net.net, input_data=x_distorted[:1], depth=10, verbose=0, device=model.device, x_restored=x_restored[:1])
        # summary.formatting.verbose = 2
        summary = str(summary)
        training_data['summary'] = summary

        ## Prepare images
        ## --------------
        training_data['imgs'] = {}
        for field, batch in (('fixed', self.fixed_batch), ('valid', self.valid_batch)):
            batch = self.split_batch(batch)[0]
            with auxil.EncapsulatedRandomState(random_seed):
                x_org, x_distorted, x_restored = model.process_batch(batch)
            err = x_org - x_restored

            training_data['imgs'][f'fix_batch_{field}'] = auxil.imgs_to_grid(torch.stack((
                auxil.sample_to_width(x_org, width=780),
                auxil.sample_to_width(model.restoration_model.naive_restore(x_distorted), width=780),
                auxil.sample_to_width(x_restored, width=780),
                auxil.sample_to_width(auxil.scale_img(err), width=780),
            )))
            zeros = auxil.sample_to_width(x_org, width=780) * 0
            training_data['imgs'][f'batch_{field}'] = auxil.imgs_to_grid(torch.stack(([zeros] * model.n_dirs)))

        ## Prepare figures
        ## ---------------
        training_data['figs'] = {}

        training_data['figs']['lr'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[])],
            layout=go.Layout(yaxis_title='lr', xaxis_title='step', yaxis_type='log', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        training_data['figs']['objective'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Objective', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        training_data['figs']['err_mse'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Error MSE', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        training_data['figs']['reconst_err'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Reconstruction Error', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        training_data['figs']['second_moment_mse'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Second Moment', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0))
        )

        for field in ('fixed', 'valid'):
            training_data['figs'][f'err_proj_{field}'] = go.Figure(
                data=[go.Scatter(mode='lines', x=[], y=[], name=str(i)) for i in range(model.n_dirs)],
                layout=go.Layout(yaxis_title='Error proj.', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
            )
            training_data['figs'][f'w_norms_{field}'] = go.Figure(
                data=[go.Scatter(mode='lines', x=[], y=[], name=str(i)) for i in range(model.n_dirs)],
                layout=go.Layout(yaxis_title='W norms', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
            )

        return training_data

    def log_batch(self, log, field):
        model = self.model

        ## Prepare data
        ## ------------
        step = model.net.step
        lr = model.net.lr

        w_mat = log['w_mat']

        err_mse_full = log['err_mse']
        err_proj_full = log['err_proj']
        w_norms_full = log['w_norms']
        reconst_err_full = log['reconst_err']
        second_moment_mse_full = log['second_moment_mse']

        objective = log['objective']

        err_mse = err_mse_full.mean().item()
        reconst_err = reconst_err_full.mean().item()
        second_moment_mse = second_moment_mse_full.mean().item()
        objective = objective.item()

        batch_img = auxil.imgs_to_grid(auxil.sample_to_width(auxil.scale_img(w_mat), width=780).transpose(0, 1).contiguous())
        model.training_data['imgs'][f'batch_{field}'] = batch_img

        ## Update console message
        ## ----------------------
        self.set_msg(field, f'{field}: step:{step:7d};   reconst err: {reconst_err:9.4g};   second moment mse: {second_moment_mse:9.4g};   objective: {objective:9.4g}')

        ## Store log data
        ## --------------
        logs = model.training_data['logs']
        logs[f'step_{field}'].append(step)
        logs[f'lr_{field}'].append(lr)
        logs[f'objective_{field}'].append(objective)
        logs[f'err_mse_{field}'].append(err_mse)
        logs[f'reconst_err_{field}'].append(reconst_err)
        logs[f'second_moment_mse_{field}'].append(second_moment_mse)
        for k in range(model.n_dirs):
            logs[f'err_proj_square_{k}_{field}'].append(err_proj_full[:, k].pow(2).mean().item())
            logs[f'w_norms_square_{k}_{field}'].append(w_norms_full[:, k].pow(2).mean().item())

        figs = model.training_data['figs']
        if field in ('fixed',):
            figs['lr'].data[0].update(x=logs['step_fixed'], y=logs['lr_fixed'])
        for key in ('objective', 'err_mse', 'reconst_err', 'second_moment_mse'):
            figs[key].data[('fixed', 'valid', 'fullv').index(field)].update(x=logs[f'step_{field}'], y=logs[f'{key}_{field}'])
        if field in ('fixed', 'valid'):
            for k in range(model.n_dirs):
                figs[f'err_proj_{field}'].data[k].update(x=logs[f'step_{field}'], y=logs[f'err_proj_square_{k}_{field}'])
                figs[f'w_norms_{field}'].data[k].update(x=logs[f'step_{field}'], y=logs[f'w_norms_square_{k}_{field}'])

    def log_step(self, train_log, fixed_log, valid_log):
        for field, log in (('fixed', fixed_log), ('valid', valid_log)):
            self.log_batch(log, field)

    def benchmark(self):
        model = self.model

        dataset = model.data_module.valid_set
        indices = np.random.RandomState(42).permutation(len(dataset))  # pylint: disable=no-member
        indices = indices[:self.max_benchmark_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            # pin_memory=True,
        )

        logs = []
        for batch in tqdm.tqdm(dataloader, dynamic_ncols=True, mininterval=1., leave=False):
            with torch.no_grad():
                _, log = self.base_step(batch)
            logs.append(log)
        log = self.cat_logs(logs)
        self.log_batch(log, 'fullv')

        score = log['err_mse'].mean().item()
        return score

    def log_html(self):
        if self.output_folder is not None:
            model = self.model

            html = inspect.cleandoc("""
            <!DOCTYPE html>
            <html lang="en">

            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no"/>
                <!--<meta http-equiv="refresh" content="30" />-->
                <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.1/plotly.min.js" integrity="sha512-R4Vz0/42Zw4rKUc8nc8TuKQGMtOMzCHxtXIFQoCB/fOT3q+L8f5l2s8n3CZsbJveMhoAV9yP8yWdj8vzwQx0hA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
                <style>
                html {font-size: 16px; line-height: 1.25em;}
                body {background-color: #4c5e79;}
                .page {max-width: 1600px; padding-right:15px; padding-left: 15px; margin: auto;}
                .twoColumns {display: grid; grid-template-columns: repeat(auto-fit, minmax(750px, 1fr));}
                </style>
            </head>

            <body>
              <div class="page">
                <h1>Results</h1>

                <div>
                    <h2>Experiment detailes</h2>
                    <div id="header"></div>
                    Updated at: <span id="now"></span> (GMT)<br>
                </div>

                <div>
                    <h2>Training images</h2>
                    <div class="twoColumns">
                        <div><img id="fix_batch_fixed_img"/><br><img id="batch_fixed_img"/></div>
                        <div><img id="fix_batch_valid_img"/><br><img id="batch_valid_img"/></div>
                    </div>
                </div>

                <div>
                    <h2>Training metrics</h2>
                    <div id="objective_fig" style="display:inline-block;"></div>
                    <br>
                    <div id="err_mse_fig" style="display:inline-block;"></div>
                    <div id="reconst_err_fig" style="display:inline-block;"></div>
                    <div id="second_moment_mse_fig" style="display:inline-block;"></div>
                    <br>
                    <div id="err_proj_fixed_fig" style="display:inline-block;"></div>
                    <div id="err_proj_valid_fig" style="display:inline-block;"></div>
                    <br>
                    <div id="w_norms_fixed_fig" style="display:inline-block;"></div>
                    <div id="w_norms_valid_fig" style="display:inline-block;"></div>
                    <br>
                    <div id="lr_fig" style="display:inline-block;"></div>
                </div>

                <div>
                    <h2>Network</h2>
                    <div id="network_summary" style="white-space:pre;"></div>
                </div>
              </div>

              <script>
            """)

            data = dict(
                header='<br>\n'.join(model.training_data['header'] + list(self.status_msgs.values())),
                now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                summary=model.training_data['summary'],
                imgs={key: auxil.img_to_png_str(img) for key, img in model.training_data['imgs'].items()},
                figs={key: plotly.io.to_json(fig) for key, fig in model.training_data['figs'].items()},
            )

            html += f'data = {json.dumps(data, indent=2)};\n'

            html += inspect.cleandoc("""
                var fig;
                window.onload = (event) => {
                    document.getElementById("header").innerHTML = data["header"];
                    document.getElementById("now").text = data["now"];
                    document.getElementById("fix_batch_fixed_img").src = "data:image/png;base64, " + data["imgs"]["fix_batch_fixed"];
                    document.getElementById("batch_fixed_img").src = "data:image/png;base64, " + data["imgs"]["batch_fixed"];
                    document.getElementById("fix_batch_valid_img").src = "data:image/png;base64, " + data["imgs"]["fix_batch_valid"];
                    document.getElementById("batch_valid_img").src = "data:image/png;base64, " + data["imgs"]["batch_valid"];

                    fig = JSON.parse(data["figs"]["objective"]);
                    Plotly.newPlot(document.getElementById("objective_fig"), fig.data, fig.layout, {responsive: true});
                    fig = JSON.parse(data["figs"]["err_mse"]);
                    Plotly.newPlot(document.getElementById("err_mse_fig"), fig.data, fig.layout, {responsive: true});
                    fig = JSON.parse(data["figs"]["reconst_err"]);
                    Plotly.newPlot(document.getElementById("reconst_err_fig"), fig.data, fig.layout, {responsive: true});
                    fig = JSON.parse(data["figs"]["err_proj_fixed"]);
                    Plotly.newPlot(document.getElementById("second_moment_mse_fig"), fig.data, fig.layout, {responsive: true});
                    fig = JSON.parse(data["figs"]["err_proj_fixed"]);
                    Plotly.newPlot(document.getElementById("err_proj_fixed_fig"), fig.data, fig.layout, {responsive: true});
                    fig = JSON.parse(data["figs"]["err_proj_valid"]);
                    Plotly.newPlot(document.getElementById("err_proj_valid_fig"), fig.data, fig.layout, {responsive: true});
                    fig = JSON.parse(data["figs"]["w_norms_fixed"]);
                    Plotly.newPlot(document.getElementById("w_norms_fixed_fig"), fig.data, fig.layout, {responsive: true});
                    fig = JSON.parse(data["figs"]["w_norms_valid"]);
                    Plotly.newPlot(document.getElementById("w_norm_valid_fig"), fig.data, fig.layout, {responsive: true});
                    fig = JSON.parse(data["figs"]["lr"]);
                    Plotly.newPlot(document.getElementById("lr_fig"), fig.data, fig.layout, {responsive: true});

                    document.getElementById("network_summary").innerText = data["summary"];
                }
              </script>
            </body>
            </html>
            """)

            with open(os.path.join(self.output_folder, 'index.html'), 'w', encoding="utf-8") as fid:
                fid.write(html)
