#!python
# pylint: disable=too-many-lines,unused-argument,unnecessary-lambda,unnecessary-lambda-assignment
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
from torch.nn.parallel import DistributedDataParallel as DDP
import torchinfo

from . import auxil
from . import datasets
from . import networks
from .base_model import BaseModel, BaseTrainer


## Model
## =====
class RestorationModel(BaseModel):
    def __init__(
            self,
            dataset,
            data_folder,
            distortion_type,
            net_type,
            img_size=None,
            store_dataset=False,
            loss_type='mse',
            lr=1e-4,
            weight_decay=0.,
            lr_lambda=None,
            random_seed = 42,
            device='cpu',
        ):
        input_args = dict(
            data_folder=data_folder,
            dataset=dataset,
            distortion_type=distortion_type,
            net_type=net_type,
            img_size=img_size,
            loss_type=loss_type,
            lr=lr,
            weight_decay=weight_decay,
            lr_lambda=lr_lambda,
            random_seed=random_seed,
        )
        super().__init__(input_args=input_args, device=device)

        self.loss_type = loss_type

        ## Initializing random state
        ## -------------------------
        auxil.set_random_seed(random_seed)

        ## Prepare datasets
        ## ----------------
        if dataset == 'mnist':
            self.data_module = datasets.MNISTDataModule(data_folder=data_folder, remove_labels=True, n_valid=0, device=self.device)

        elif dataset == 'celeba_hq_256':
            self.data_module = datasets.CelebAHQ256DataModule(img_size=img_size, data_folder=data_folder, store_dataset=store_dataset)

        elif dataset == 'celeba_srflow':
            self.data_module = datasets.CelebASRFlowDataModule(data_folders=data_folder, scale=8, store_dataset=store_dataset)

        else:
            raise Exception(f'Unsupported dataset: "{dataset}"')

        self.x_shape = self.data_module.shape

        ## Prepare distortion model
        ## ------------------------
        if distortion_type == 'denoising_1':
            ## Adding noise with sigma=1 + clipping
            self.distortion_model = Denoising(noise_std=1., clip_noise=True)

        elif distortion_type == 'inpainting_1':
            ## Inpainting all but the last 8 rows
            mask = gen_mask(self.x_shape, 0, self.x_shape[-2] - 9, 0, self.x_shape[-1]).to(self.device)
            self.distortion_model = Inpainting(mask=mask, fill=self.data_module.mean).to(self.device)

        elif distortion_type == 'inpainting_2':
            ## Inpainting eyes in CelebA 256x256 images
            mask = gen_mask(self.x_shape, 80, 149, 40, 214).to(self.device)
            self.distortion_model = Inpainting(mask=mask, fill=self.data_module.mean).to(self.device)

        elif distortion_type == 'colorization_1':
            ## Removing color be averaging the color channels
            self.distortion_model = Colorization().to(self.device)

        elif distortion_type == 'super_resolution_1':
            ## Upscaling by a factor of 4x
            self.distortion_model = SuperResolution(factor=4)

        elif distortion_type == 'distorted_dataset':
            self.distortion_model = None

        else:
            raise Exception(f'Unsupported distortion_type: "{distortion_type}"')

        if self.distortion_model is not None:
            x = self.data_module.train_set[0].to(self.device)
            x_distorted = self.distortion_model(x[None])
        else:
            x_distorted = self.data_module.train_set[0][1].to(self.device)
        self.x_distorted_shape = x_distorted.shape[1:]
        self.naive_restore = self.distortion_model.naive_restore
        self.project = self.distortion_model.project

        ## Set parametric model
        ## --------------------
        upscale_factor = self.x_shape[-1] // self.x_distorted_shape[-1]

        if net_type == 'unet':
            ## Vanilla U-Net
            base_net = networks.UNet(
                in_channels=self.x_distorted_shape[0],
                out_channels=self.x_shape[0],
                channels_list=(32, 64, 128),
                bottleneck_channels=256,
                downsample_list=(False, True, True),
                n_blocks=1,
                n_blocks_bottleneck=2,
                min_channels_decoder=64,
                upscale_factor=upscale_factor,
            )

        elif net_type == 'res_unet':
            ## DDPM for 256x256
            base_net = networks.ResUNet(
                in_channels=self.x_distorted_shape[0],
                out_channels=self.x_shape[0],
                channels_list=(64, 64, 128, 128, 256, 256),
                bottleneck_channels=512,
                downsample_list=(False, True, True, True, True, True),
                attn_list=(False, False, False, False, True, False),
                n_blocks=2,
                n_groups=8,
                attn_heads=1,
                min_channels_decoder=1,
                upscale_factor=upscale_factor,
            )

        elif net_type == 'res_cnn':
            ## EDSR
            base_net = networks.ResCNN(
                channels_in=self.x_distorted_shape[0],
                channels_out=self.x_shape[0],
                channels_hidden=64,
                n_blocks=16,
                upscale_factor=upscale_factor,
            )

        else:
            raise Exception(f'Unsupported net_type: "{net_type}"')

        net = RestorationWrapper(
            net=base_net,
            offset=self.data_module.mean,
            scale=self.data_module.std,
            naive_restore_func=self.naive_restore,
            project_func=self.project,
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
        if isinstance(batch, (tuple, list)):
            x_org = batch[0].to(self.device)
            x_distorted = batch[1].to(self.device)
        else:
            x_org = batch.to(self.device)
            x_distorted = self.distort(x_org)
        return x_org, x_distorted

    def distort(self, x):
        x_distorted = self.distortion_model.distort(x)
        return x_distorted

    def restore(self, x_distorted, use_best=True, **kwargs):
        x_restored = self.net(x_distorted, use_best=use_best, **kwargs)
        return x_restored


class RestorationWrapper(nn.Module):
    def __init__(self, net, offset=None, scale=None, naive_restore_func=None, project_func=None, pad_base_size=None):
        super().__init__()

        self.net = net
        self.offset = offset
        self.scale = scale
        self.naive_restore_func = naive_restore_func
        self.project_func = project_func
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

    def forward(self, x_distorted):
        x_distorted_org = x_distorted

        if self.offset is not None:
            x_distorted = x_distorted - self.offset
        if self.scale is not None:
            x_distorted = x_distorted / self.scale

        padding = self._get_padding(x_distorted, self.pad_base_size)
        if padding is not None:
            x_distorted = F.pad(x_distorted, tuple(padding))

        x_restored = self.net(x_distorted)

        if padding is not None:
            x_restored = F.pad(x_restored, tuple(-padding))  # pylint: disable=invalid-unary-operand-type

        if self.scale is not None:
            x_restored = x_restored * self.scale

        x_restored = self.naive_restore_func(x_distorted_org) + self.project_func(x_restored)

        return x_restored


## Distortion models
## =================
class Denoising(nn.Module):
    def __init__(
            self,
            noise_std,
            clip_noise=False,
        ):

        super().__init__()
        self.noise_std = noise_std
        self.clip_noise= clip_noise

    def distort(self, x, random_seed=None):
        with auxil.EncapsulatedRandomState(random_seed):
            x_distorted = x + torch.randn_like(x) * self.noise_std
        if self.clip_noise:
            x_distorted = x_distorted.clamp(0, 1)
        return x_distorted

    def forward(self, x, random_seed=None):
        return self.distort(x, random_seed=random_seed)

    def naive_restore(self, x):
        return x

    def project(self, x):
        return x

def gen_mask(shape, top, bottom, left, right):
    mask = torch.zeros(shape)
    mask[:, top:(bottom + 1), left:(right + 1)] = 1.
    return mask


class Inpainting(nn.Module):
    def __init__(
            self,
            mask,
            fill=0.
        ):

        super().__init__()
        self.fill = fill
        self.register_buffer('mask', mask)

    def distort(self, x, random_seed=None):
        x = x * (1 - self.mask) + self.fill * self.mask
        return x

    def forward(self, x, random_seed=None):
        return self.distort(x, random_seed=random_seed)
    
    def naive_restore(self, x):
        return x

    def project(self, x):
        x = x * self.mask
        return x


class Colorization(nn.Module):
    def distort(self, x, random_seed=None):
        x = x.mean(dim=1)
        return x

    def forward(self, x, random_seed=None):
        return self.distort(x, random_seed=random_seed)

    def naive_restore(self, x):
        return x.repeat_interleave(3, dim=1)

    def project(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        return x



class SuperResolution(nn.Module):
    def __init__(
            self,
            factor,
            noise_std=0.,
        ):

        super().__init__()
        self.factor = factor
        self.noise_std = noise_std

    def distort(self, x, random_seed=None):
        x = F.avg_pool2d(x, self.factor)
        if self.noise_std > 0:
            with auxil.EncapsulatedRandomState(random_seed):
                x = x + torch.randn_like(x) * self.noise_std
        return x

    def forward(self, x, random_seed=None):
        return self.distort(x, random_seed=random_seed)

    def naive_restore(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode='nearest')

    def project(self, x):
        if self.noise_std == 0:
            x = F.avg_pool2d(x, self.factor)
            x_proj = F.interpolate(x_proj, scale_factor=self.factor, mode='nearest')
            x = x - x_proj
        return x


## Trainer
## =======
class RestorationTrainer(BaseTrainer):
    def __init__(self, *args,
            max_benchmark_samples=256,
            **kwargs):

        super().__init__(*args, **kwargs)
        self.max_benchmark_samples = max_benchmark_samples

    def base_step(self, batch):
        model = self.model

        x_org, x_distorted = model.process_batch(batch)
        x_restored = model.restore(x_distorted, use_best=False, use_ddp=True)
        err = x_org - x_restored

        ## Mean vector loss
        ## ----------------
        err_mse = err.pow(2).flatten(1).mean(dim=-1)
        err_mae = err.abs().flatten(1).mean(dim=-1)

        if model.loss_type == 'mse':
            objective = err_mse.mean()
        elif model.loss_type == 'mae':
            objective = err_mae.mean()
        else:
            raise Exception(f'Unsupported loss type "{model.loss_type}"')

        ## Store log
        ## ---------
        log = dict(
            x_org=x_org,
            x_distorted=x_distorted,
            x_restored=x_restored.detach(),

            err_mse=err_mse.detach(),
            err_mae=err_mae.detach(),

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
            if key in ('x_org', 'x_distorted', 'x_restored', 'err_mse', 'err_mae'):
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
                                                      for key in ('step', 'lr', 'objective', 'err_rmse', 'err_mae')}

        ## Prepare summary
        ## ---------------
        batch = self.fixed_batch
        with auxil.EncapsulatedRandomState(random_seed):
            x_org, x_distorted = model.process_batch(batch)
        summary = torchinfo.summary(model.net.net, input_data=x_distorted[:1], depth=10, verbose=0, device=model.device)
        # summary.formatting.verbose = 2
        summary = str(summary)
        training_data['summary'] = summary

        ## Prepare images
        ## --------------
        training_data['imgs'] = {}
        for field, batch in (('fixed', self.fixed_batch), ('valid', self.valid_batch)):
            with auxil.EncapsulatedRandomState(random_seed):
                x_org, x_distorted = model.process_batch(batch)

            training_data['imgs'][f'fix_batch_{field}'] = auxil.imgs_to_grid(torch.stack((
                auxil.sample_to_width(x_org, width=780),
                auxil.sample_to_width(model.naive_restore(x_distorted), width=780),
            )))
            zeros = auxil.sample_to_width(x_org, width=780) * 0
            training_data['imgs'][f'batch_{field}'] = auxil.imgs_to_grid(torch.stack(([zeros] * 2)))

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

        training_data['figs']['err_rmse'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Error RMSE', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        training_data['figs']['err_mae'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Error MAE', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        return training_data

    def log_batch(self, log, field):
        model = self.model

        ## Prepare data
        ## ------------
        step = model.net.step
        lr = model.net.lr

        x_org = log['x_org']
        # x_distorted = log['x_distorted']
        x_restored = log['x_restored']

        err_mse_full = log['err_mse']
        err_mae_full = log['err_mae']

        objective = log['objective']


        err_mse = err_mse_full.mean().item()
        err_mae = err_mae_full.mean().item()
        err_rmse = err_mse ** 0.5
        objective = objective.item()

        err = x_org - x_restored

        batch_img = auxil.imgs_to_grid(torch.stack((
            auxil.sample_to_width(x_restored, width=780),
            auxil.sample_to_width(auxil.scale_img(err), width=780),
        )))
        model.training_data['imgs'][f'batch_{field}'] = batch_img

        ## Update console message
        ## ----------------------
        self.set_msg(field, f'{field}: step:{step:7d};   err_rmse: {err_rmse:9.4g};   err_mae: {err_mae:9.4g};   objective: {objective:9.4g}')

        ## Store log data
        ## --------------
        logs = model.training_data['logs']
        logs[f'step_{field}'].append(step)
        logs[f'lr_{field}'].append(lr)
        logs[f'objective_{field}'].append(objective)
        logs[f'err_rmse_{field}'].append(err_rmse)
        logs[f'err_mae_{field}'].append(err_mae)

        figs = model.training_data['figs']
        if field in ('fixed',):
            figs['lr'].data[0].update(x=logs['step_fixed'], y=logs['lr_fixed'])
        for key in ('objective', 'err_rmse', 'err_mae'):
            figs[key].data[('fixed', 'valid', 'fullv').index(field)].update(x=logs[f'step_{field}'], y=logs[f'{key}_{field}'])

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
                    <div id="err_rmse_fig" style="display:inline-block;"></div>
                    <div id="err_mae_fig" style="display:inline-block;"></div>
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
                    fig = JSON.parse(data["figs"]["err_rmse"]);
                    Plotly.newPlot(document.getElementById("err_rmse_fig"), fig.data, fig.layout, {responsive: true});
                    fig = JSON.parse(data["figs"]["err_mae"]);
                    Plotly.newPlot(document.getElementById("err_mae_fig"), fig.data, fig.layout, {responsive: true});
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
