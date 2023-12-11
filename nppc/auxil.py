import copy
import random
import datetime
import time
from io import StringIO, BytesIO
import base64
from contextlib import contextmanager

from PIL import Image
import numpy as np
import torch
import torch.distributed as distrib
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import line_profiler


## General auxiliary functions
## ===========================
@contextmanager
def run_and_profile(funcs_list, enable=True, output_filename='/tmp/profile.txt'):
    if enable and (len(funcs_list) > 0):
        profiler = line_profiler.LineProfiler()
        for func in funcs_list:
            profiler.add_function(func)
        profiler.enable_by_count()
        try:
            yield
        finally:
            with StringIO() as str_stream:
                profiler.print_stats(str_stream)
                string = str_stream.getvalue()
            print(f'Writing profile data to "{output_filename}"')
            with open(output_filename, 'w', encoding='utf-8') as fid:
                fid.write(string)
    else:
        yield

def set_random_seed(random_seed=0):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


class EncapsulatedRandomState:
    def __init__(self, random_seed=0):
        self._random_seed = random_seed
        self._random_state = None
        self._np_random_state = None
        self._torch_random_state = None
        self._torch_cuda_random_state = None

    def __enter__(self):
        if self._random_seed is not None:
            self._random_state = random.getstate()
            self._np_random_state = np.random.get_state()
            self._torch_random_state = torch.random.get_rng_state()
            self._torch_cuda_random_state = {i: torch.cuda.get_rng_state(i)  for i in range(torch.cuda.device_count())}

            random.seed(self._random_seed)
            np.random.seed(self._random_seed)
            torch.random.manual_seed(self._random_seed)
            torch.cuda.manual_seed_all(self._random_seed)

    def __exit__(self, type_, value, traceback):
        if self._random_seed is not None:
            random.setstate(self._random_state)
            np.random.set_state(self._np_random_state)
            torch.random.set_rng_state(self._torch_random_state)
            for i, random_state in self._torch_cuda_random_state.items():
                torch.cuda.set_rng_state(random_state, i)


class Timer(object):
    def __init__(self, interval, reset=True):
        self._interval = interval
        if self._interval is None:
            self._end_time = None
        else:
            self._end_time = time.time()

        if reset:
            self.reset()

    def __bool__(self):
        if self._end_time is None:
            return False
        else:
            return time.time() > self._end_time

    def __str__(self):
        if self._end_time is None:
            return '----'
        else:
            timedelta = int(self._end_time - time.time())
            if timedelta >= 0:
                return str(datetime.timedelta(seconds=timedelta))
            else:
                return '-' + str(datetime.timedelta(seconds=-timedelta))

    def reset(self, interval=None):
        if interval is not None:
            self._interval = interval

        if self._interval is None:
            self._end_time = None
        else:
            self._end_time = time.time() + self._interval


## Visualizations
## ==============
def sample_to_width(x, width=1580, padding_size=2):
    n_samples = min((width - padding_size) // (x.shape[-1] + padding_size), x.shape[0])
    indices = np.linspace(0, x.shape[0] - 1, n_samples).astype(int)
    return x[indices]

def imgs_to_grid(imgs, nrows=None, **make_grid_args):
    imgs = imgs.detach().cpu()
    if imgs.ndim == 5:
        nrow = imgs.shape[1]
        imgs = imgs.reshape(imgs.shape[0] * imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4])
    elif nrows is None:
        nrow = int(np.ceil(imgs.shape[0] ** 0.5))

    make_grid_args2 = dict(scale_range=(0, 1), pad_value=1.)
    make_grid_args2.update(make_grid_args)
    img = torchvision.utils.make_grid(imgs, nrow=nrow, **make_grid_args2).clamp(0, 1)
    return img

def scale_img(x):
    return x / torch.abs(x).flatten(-3).max(-1)[0][..., None, None, None] / 1.5 + 0.5

def tensor_img_to_numpy(x):
    return x.detach().permute(-2, -1, -3).cpu().numpy()

def to_data_str(x):
    return base64.b64encode(x).decode('ascii')

def img_to_png_str(img, round=True):
    img = tensor_img_to_numpy(img)
    if not np.issubdtype(img.dtype, np.integer):
        img = img * (2 ** 8 - 1)
        if round:
            img = img.round()
    img = img.clip(0, 2 ** 8 - 1).astype('uint8')
    if img.shape[-1] == 1:
        img = img.repeat(3, -1)

    with BytesIO() as buffered:
        Image.fromarray(img, 'RGB').save(buffered, format='PNG')
        png_str = to_data_str(buffered.getvalue())
    return png_str


## DDP Manager
## ===========
class DDPManager:
    def __init__(self, store_port=56895):
        self._store_port = store_port

        self.is_active = distrib.is_available() and distrib.is_initialized()
        if self.is_active:
            self.rank = distrib.get_rank()
            self.is_main = self.rank == 0
            self.size = distrib.get_world_size()
        else:
            self.rank = 0
            self.is_main = True
            self.size = 1
        self.store = None

    def broadcast(self, x):
        if self.is_active:
            if self.store is None:
                self.store = distrib.TCPStore("127.0.0.1", self._store_port, self.size, self.is_main)

            if self.is_main:
                self.store.set('broadcasted_var', x)
                distrib.barrier()
            else:
                self.store.wait(['broadcasted_var'], datetime.timedelta(minutes=5))
                x = self.store.get('broadcasted_var')
                distrib.barrier()
        return x

    def gather(self, x):
        if self.is_active:
            res = [x.clone() for _ in range(self.size)]
            distrib.gather(x, res if self.is_main else None)
        else:
            res = [x]
        return res

    def convert_model(self, model, device_ids):
        if self.is_active:
            model = DDP(model, device_ids=device_ids)
        return model


## Net wrapper
## ===========
class NetWrapper(object):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    def __init__(
            self,
            net,
            optimizer_type='adam',
            optimizer_params=None,
            lr_lambda=None,
            ema_alpha=None,
            ema_update_every=1,
            use_ema_for_best=None,
            device='cpu',
            ddp_active=False,
        ):

        self._device = device
        self._step = 0

        self._requires_grad = True
        self.net = net
        self.net.to(self.device)
        self.net.eval()

        ## Set field to store best network
        ## -------------------------------
        if use_ema_for_best is None:
            use_ema_for_best = ema_alpha is not None
        self._use_ema_for_best = use_ema_for_best
        self._net_best = None
        self._score_best = None
        self._step_best = None

        ## Setup Exponantial moving average (EMA)
        ## --------------------------------------
        self._ema_alpha = ema_alpha
        self._ema_update_every = ema_update_every
        if self._ema_alpha is not None:
            self._net_ema = self._make_copy(self.net)
        else:
            self._net_ema = None

        ## Initialize DDP
        ## --------------
        if ddp_active:
            self._net_ddp = DDP(self.net, device_ids=[self.device])
        else:
            self._net_ddp = self.net

        ## Set optimizer
        ## -------------
        if optimizer_type.lower() == 'adam':
            if ('weight_decay' in optimizer_params) and (optimizer_params['weight_decay'] > 0.):
                self.optimizer = torch.optim.AdamW(self._net_ddp.parameters(), **optimizer_params)
            else:
                self.optimizer = torch.optim.Adam(self._net_ddp.parameters(), **optimizer_params)
        else:
            raise Exception(f'Unsuported optimizer type: "{optimizer_type}"')

        ## Set schedualer
        ## --------------
        if lr_lambda is None:
            lr_lambda = lambda step: 1
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    @property
    def device(self):
        return self._device

    @property
    def step(self):
        return self._step

    def get_net(self, use_ddp=False, use_ema=False, use_best=False):
        if use_ddp:
            net = self._net_ddp
        elif use_ema:
            if self._net_ema is not None:
                net = self._net_ema
            else:
                net = self.net
        elif use_best:
            if self._net_best is not None:
                net = self._net_best
            else:
                if self._use_ema_for_best:
                    net = self._net_ema
                else:
                    net = self.net
        else:
            net = self.net
        return net

    def __call__(self, *args, use_ddp=False, use_ema=False, use_best=False, **kwargs):
        return self.get_net(use_ddp=use_ddp, use_ema=use_ema, use_best=use_best)(*args, **kwargs)

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train(self):
        self._net_ddp.train()

    def eval(self):
        self._net_ddp.eval()

    @property
    def requires_grad_(self):
        return self._requires_grad

    @requires_grad_.setter
    def requires_grad_(self, requires_grad):
        self.net.requires_grad_(requires_grad)
        self._requires_grad = requires_grad

    def increment(self):
        self._step += 1
        self.scheduler.step()

        ## EMA step
        if (self._net_ema is not None) and (self.step % self._ema_update_every == 0):
            alpha = max(self._ema_alpha, 1 / (self.step // self._ema_update_every))
            for p, p_ema in zip(self.net.parameters(), self._net_ema.parameters()):
                p_ema.data.mul_(1 - alpha).add_(p.data, alpha=alpha)

    def clip_grad_norm(self, max_norm):
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._net_ddp.parameters(), max_norm=max_norm)

    @contextmanager
    def set_ddp_sync(self, active=True):
        if isinstance(self._net_ddp, torch.nn.parallel.distributed.DistributedDataParallel):
            old_val = self._net_ddp.require_backward_grad_sync
            self.net.require_backward_grad_sync = active  # pylint: disable=attribute-defined-outside-init
            try:
                yield
            finally:
                self.net.require_backward_grad_sync = old_val  # pylint: disable=attribute-defined-outside-init
        else:
            try:
                yield
            finally:
                pass

    def update_best(self, score):
        if score is not None:
            if (self._score_best is None) or (score <= self._score_best):
                if self._use_ema_for_best and (self._net_ema is not None):
                    self._net_best = self._make_copy(self._net_ema)
                else:
                    self._net_best = self._make_copy(self.net)
                self._score_best = score
                self._step_best = self._step

    @staticmethod
    def _make_copy(network):
        network = copy.deepcopy(network)
        network.requires_grad_(False)
        network.eval()
        return network

    ## Save and load
    ## -------------
    def state_dict(self):
        state_dict = dict(
            step = self._step,
            net = self.net.state_dict(),
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            net_ema = self._net_ema.state_dict() if (self._net_ema is not None) else None,
            net_best = self._net_best.state_dict() if (self._net_best is not None) else None,
            score_best = self._score_best,
            step_best = self._step_best,
        )
        return state_dict

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        self.net.load_state_dict(state_dict['net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        if (self._net_ema is not None) and (state_dict['net_ema'] is not None):
            self._net_ema.load_state_dict(state_dict['net_ema'])
        if state_dict['net_best'] is None:
            self._net_best = None
        else:
            self._net_best = self._make_copy(self.net)
            self._net_best.load_state_dict(state_dict['net_best'])
        self._score_best = state_dict['score_best']
        self._step_best = state_dict['step_best']


