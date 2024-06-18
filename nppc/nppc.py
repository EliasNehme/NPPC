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
from .restoration import RestorationModel


## Model
## =====
class NPPCModel(object):
    def __init__(
            self,
            restoration_model_folder,
            net_type,
            pre_net_type='none',
            n_dirs=5,
            lr=1e-4,
            second_moment_loss_lambda=1e-1,
            second_moment_loss_grace=200,
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

        self.device = device
        self.input_args = input_args
        self.ddp = auxil.DDPManager()
        self.extra_data = {}

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
                in_channels=self.x_distorted_shape[0],
                out_channels=pre_out_channels,
                hidden_channels=64,
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

    def split_batch(self, batch, n_chunks):
        return self.restoration_model.split_batch(batch, n_chunks)

    def process_batch(self, batch):
        with torch.no_grad():
            x_org, x_distorted = self.restoration_model.process_batch(batch)
            x_restored = self.restoration_model.restore(x_distorted)
        return x_org, x_distorted, x_restored

    def get_dirs(self, x_distorted, x_restored, use_best=True, **kwargs):
        w_mat = self.net(x_distorted, x_restored, use_best=use_best, **kwargs)
        return w_mat

    ## Save and load
    ## -------------
    def state_dict(self):
        state_dict = {}
        state_dict['input_args'] = self.input_args
        state_dict['net'] = self.net.state_dict()
        state_dict['extra_data'] = self.extra_data
        return state_dict

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict['net'])
        self.extra_data = state_dict['extra_data']

    @classmethod
    def load(cls, checkpoint_filename, device='cpu', **kwargs):
        state_dict = torch.load(checkpoint_filename, map_location=device)
        input_args = state_dict.pop('input_args')
        input_args.update(kwargs)

        model = cls(device=device, **input_args)
        model.load_state_dict(state_dict)
        return model


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
class NPPCTrainer(object):
    def __init__(
            self,
            model,
            batch_size,
            max_chunk_size=None,
            output_folder=None,
            gradient_clip_val=None,
            overfit=False,
            num_workers=0,
            max_benchmark_samples=256,
        ):

        self.model = model
        self.status_msgs = None

        self.output_folder = output_folder
        self.gradient_clip_val = gradient_clip_val
        self.overfit = overfit
        self.num_workers = num_workers
        self.max_benchmark_samples = max_benchmark_samples

        ## Set batch size
        ## --------------
        self.batch_size = batch_size // model.ddp.size

        ## Use batches accumulation if necessary
        if (max_chunk_size is not None) and (self.batch_size > max_chunk_size):
            self.n_chunks = self.batch_size // max_chunk_size
            self.batch_size = self.batch_size // self.n_chunks * self.n_chunks
        else:
            self.n_chunks = 1

        ## Store a fixed batch from the validation set (for visualizations and benchmarking)
        ## ---------------------------------------------------------------------------------
        dataloader = torch.utils.data.DataLoader(
            model.data_module.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(0),
        )
        self.fixed_batch = next(iter(dataloader))

        dataloader = torch.utils.data.DataLoader(
            model.data_module.valid_set,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(0),
        )
        self.valid_batch = next(iter(dataloader))
    
    def train(
            self,
            n_steps=None,
            n_epochs=None,
            log_every=None,
            benchmark_every=None,
            html_min_interval=60,
            save_min_interval=None,
            random_seed=43,
        ):

        model = self.model

        ## Initializing random state
        ## -------------------------
        auxil.set_random_seed(random_seed)

        ## PyTorch settings
        ## ---------------
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True  # !!Warning!!: This makes things run A LOT slower
        # torch.autograd.set_detect_anomaly(True)  # !!Warning!!: This also makes things run slower

        ## Test step
        ## ---------
        batch = self.split_batch(self.fixed_batch)[0]
        self.base_step(batch)

        ## Initializing data loaders
        ## -------------------------
        sampler = torch.utils.data.distributed.DistributedSampler(
            model.data_module.train_set,
            num_replicas=model.ddp.size,
            rank=model.ddp.rank,
            shuffle=True,
            drop_last=True,
        )
        dataloader = torch.utils.data.DataLoader(
            model.data_module.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            # pin_memory=True,
        )

        ## Set logging periods
        ## -------------------
        if model.ddp.is_main:
            ## Initialize train logging
            ## ------------------------
            model.training_data = self.init_training_data()

            ## Set up output folder
            ## --------------------
            if self.output_folder is not None:
                if not os.path.isdir(self.output_folder):
                    os.makedirs(self.output_folder)

            ## Set up console logging
            ## ----------------------
            header = []
            header.append(f'Output folder: {self.output_folder}')
            header.append(f'Batch size: {self.batch_size * model.ddp.size} = {self.batch_size // self.n_chunks} x {self.n_chunks} x {model.ddp.size} (chunk size x accumulation x GPUs)')
            header.append(f'Number of steps in epoch: {len(dataloader)}; Logging every {log_every}. Benchmarking every {benchmark_every}')
            model.training_data['header'] = header
            print('\n'.join(header))
            self.status_msgs = auxil.StatusMassages(('fixed', 'valid', 'fullv', 'state', 'step', 'next', 'lr'))

        ## Initialize timers
        ## -----------------
        timers = dict(
            html = auxil.Timer(html_min_interval, reset=False),
            save = auxil.Timer(save_min_interval, reset=True),
        )

        ## Training loop
        ## -------------
        loop_loader = auxil.LoopLoader(dataloader, n_steps=n_steps, n_epochs=n_epochs)
        for i_step, batch in enumerate(tqdm.tqdm(loop_loader, ncols=0, disable=not model.ddp.is_main, mininterval=1.)):
            if model.ddp.is_main:
                self.status_msgs.set('state', 'Training...')
                self.status_msgs.set('step', f'Step: {self.model.net.step}')
                self.status_msgs.set('next', 'Next: ' + ', '.join([f'{key}: {val}' for key, val in timers.items()]))
                self.status_msgs.set('lr', f'Learning rate: {self.model.net.lr}')

            ## Train step
            ## ----------
            if self.overfit:
                batch = self.fixed_batch

            logs = []
            model.net.optimizer.zero_grad()
            for i_chunk, chunk in enumerate(self.split_batch(batch)):
                last_part = i_chunk == (self.n_chunks - 1)
                with model.net.set_ddp_sync(last_part):
                    objective, log = self.base_step(chunk)
                    objective = objective / self.n_chunks
                    objective.backward()
                logs.append(log)
            train_log = self.cat_logs(logs)
            model.net.clip_grad_norm(self.gradient_clip_val)
            model.net.optimizer.step()
            model.net.increment()

            ## Logging, benchmarking & save
            ## ----------------------------
            if model.ddp.is_main:
                if i_step == len(loop_loader) - 1:
                    benchmark_flag = True
                elif benchmark_every is not None:
                    benchmark_flag = (i_step + 1) % benchmark_every == 0
                else:
                    benchmark_flag = (i_step + 1) % len(dataloader) == 0
                
                if benchmark_flag or (model.net.step == 1):
                    log_flag = True
                elif log_every is not None:
                    log_flag = model.net.step % log_every == 0
                else:
                    log_flag = False

                if log_flag:
                    model.net.eval()

                    self.status_msgs.set('state', 'Running fixed batch...')
                    logs = []
                    for chunk in self.split_batch(self.fixed_batch):
                        with auxil.EncapsulatedRandomState(random_seed):
                            with torch.no_grad():
                                _, log = self.base_step(chunk)
                        logs.append(log)
                    fixed_log = self.cat_logs(logs)

                    logs = []
                    for chunk in self.split_batch(self.valid_batch):
                        with auxil.EncapsulatedRandomState(random_seed + 1):
                            with torch.no_grad():
                                _, log = self.base_step(chunk)
                        logs.append(log)
                    valid_log = self.cat_logs(logs)

                    self.log_step(train_log, fixed_log, valid_log)

                    if benchmark_flag:
                        self.status_msgs.set('state', 'Running benchmark...')
                        with auxil.EncapsulatedRandomState(random_seed + 2):
                            score = self.benchmark()  # pylint: disable=assignment-from-none
                        model.net.update_best(score)
                        if timers['save']:
                            if self.output_folder is not None:
                                self.status_msgs.set('state', 'Saving...')
                                torch.save(model.state_dict(), os.path.join(self.output_folder, 'checkpoint.pt'))
                            timers['save'].reset()

                    if timers['html']:
                        self.status_msgs.set('state', 'Writing HTML ...')
                        self.log_html()
                        timers['html'].reset()
                    model.net.train()

        if self.output_folder is not None:
            self.status_msgs.set('state', 'Saving...')
            torch.save(model.state_dict(), os.path.join(self.output_folder, 'checkpoint.pt'))

    def split_batch(self, batch):
        return self.model.split_batch(batch, self.n_chunks)

    def base_step(self, batch):
        model = self.model

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

            err_mse=err_norm.detach().pow(2),
            err_proj=err_proj.detach(),
            w_norms=w_norms.detach(),
            reconst_err=reconst_err.detach(),
            second_moment_mse=second_moment_mse.detach(),

            objective=objective.detach(),
        )
        return objective, log

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

        score = log['reconst_err'].mean().item()
        return score

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

        if 'training_data' in model.extra_data:
            training_data = model.extra_data['training_data']
        else:
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
                    layout=go.Layout(yaxis_title=f'Error proj. {field}', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
                )
                training_data['figs'][f'w_norms_{field}'] = go.Figure(
                    data=[go.Scatter(mode='lines', x=[], y=[], name=str(i)) for i in range(model.n_dirs)],
                    layout=go.Layout(yaxis_title=f'W norms {field}', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
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
        self.status_msgs.set(field, f'{field}: step:{step:7d};   reconst err: {reconst_err:9.4g};   second moment mse: {second_moment_mse:9.4g};   objective: {objective:9.4g}')

        ## Store log data
        ## --------------
        logs = model.training_data['logs']
        logs[f'step_{field}'].append(step)
        logs[f'lr_{field}'].append(lr)
        logs[f'objective_{field}'].append(objective)
        dim = np.prod(model.x_shape)
        logs[f'err_mse_{field}'].append(err_mse / dim)
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

    def log_html(self):
        if self.output_folder is not None:
            model = self.model

            data = dict(
                html = dict(
                    header='<br>\n'.join(model.training_data['header'] + list(self.status_msgs.msgs.values())),
                ),
                text = dict(
                    now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    network_summary=model.training_data['summary'],
                ),
                imgs={key: auxil.img_to_png_str(img) for key, img in model.training_data['imgs'].items()},
                figs={key: fig.to_plotly_json() for key, fig in model.training_data['figs'].items()},
            )

            html = inspect.cleandoc(r"""
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

                  <div id=loading_block>
                    Loading ...
                  </div>

                  <div id=main_block style="display: none;">
                    <div>
                      <h2>Experiment detailes</h2>
                      <div id="data_header_html"></div>
                      Updated at: <span id="data_now_text"></span> (GMT)<br>
                    </div>

                    <div>
                      <h2>Training images</h2>
                      <div class="twoColumns">
                        <div><img id="data_fix_batch_fixed_img"/><br><img id="data_batch_fixed_img"/></div>
                        <div><img id="data_fix_batch_valid_img"/><br><img id="data_batch_valid_img"/></div>
                      </div>
                    </div>

                    <div>
                      <h2>Training metrics</h2>
                      <div id="data_objective_fig" style="display:inline-block;"></div>
                      <div id="data_err_mse_fig" style="display:inline-block;"></div>
                      <div id="data_reconst_err_fig" style="display:inline-block;"></div>
                      <div id="data_second_moment_mse_fig" style="display:inline-block;"></div>
                      <br>
                      <div id="data_err_proj_fixed_fig" style="display:inline-block;"></div>
                      <div id="data_err_proj_valid_fig" style="display:inline-block;"></div>
                      <br>
                      <div id="data_w_norms_fixed_fig" style="display:inline-block;"></div>
                      <div id="data_w_norms_valid_fig" style="display:inline-block;"></div>
                      <br>
                      <div id="data_lr_fig" style="display:inline-block;"></div>
                    </div>

                    <div>
                      <h2>Network</h2>
                      <div id="data_network_summary_text" style="white-space:pre;"></div>
                    </div>
                  </div>
                </div>

                <script>
                  window.onload = (event) => {
            """)

            html += '\n\n' + 'const data = ' + json.dumps(data, indent=2) + ';\n\n'

            html += inspect.cleandoc(r"""
                    var el;
                    for (const key in data["html"]) {
                      el = document.getElementById("data_" + key + "_html");
                      if (el != null) {
                        el.innerHTML = data["html"][key];
                      }
                    }
                    for (const key in data["text"]) {
                      el = document.getElementById("data_" + key + "_text");
                      if (el != null) {
                        el.innerText = data["text"][key];
                      }
                    }
                    for (const key in data["imgs"]) {
                      el = document.getElementById("data_" + key + "_img");
                      if (el != null) {
                        el.src = "data:image/png;base64, " + data["imgs"][key];
                      }
                    }
                    for (const key in data["figs"]) {
                      el = document.getElementById("data_" + key + "_fig");
                      if (el != null) {
                        Plotly.newPlot(el, data["figs"][key]["data"], data["figs"][key]["layout"], {responsive: true});
                      }
                    }
                    document.getElementById("loading_block").style.display = "none";
                    document.getElementById("main_block").style.display = "block";
                  }
                </script>
                </body>
                </html>
            """)

            with open(os.path.join(self.output_folder, 'index.html'), 'w', encoding="utf-8") as fid:
                fid.write(html)
