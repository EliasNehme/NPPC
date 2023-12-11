import os
from collections import OrderedDict

import tqdm.auto as tqdm
import torch

from . import auxil
from . import datasets


## Base Model Class
## ================
class BaseModel(object):
    def __init__(self, input_args, device):
        self.device = device
        self.input_args = input_args
        self.ddp = auxil.DDPManager()

        self.networks = {}
        self.training_data = {}

        self.data_module = None

        self.net = None

    def split_batch(self, batch, n_chunks):
        return datasets.split_batch(batch, n_chunks)

    def process_batch(self, batch):
        return batch

    ## Save and load
    ## -------------
    def state_dict(self):
        state_dict = {}
        state_dict['input_args'] = self.input_args
        state_dict['net'] = self.net.state_dict()
        state_dict['training_data'] = self.net.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict['net'])
        self.training_data = state_dict['training_data']

    @classmethod
    def load(cls, checkpoint_filename, device='cpu', **kwargs):
        state_dict = torch.load(checkpoint_filename, map_location=device)
        input_args = state_dict.pop('input_args')
        input_args.update(kwargs)

        model = cls(device=device, **input_args)
        model.load_state_dict(state_dict)
        return model


## Trainer
## =======
class BaseTrainer(object):
    def __init__(
            self,
            model,
            batch_size,
            max_chunk_size=None,
            output_folder=None,
            gradient_clip_val=None,
            overfit=False,
            num_workers=0,
        ):

        self.model = model

        self.output_folder = output_folder
        self.gradient_clip_val = gradient_clip_val
        self.overfit = overfit
        self.num_workers = num_workers

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

        ## Setup console massages
        ## ----------------------

        status_fields = ('train', 'fixed', 'valid', 'fullv', 'state', 'step', 'next', 'lr')
        self.status_msgs = OrderedDict([(key, f'--({key})--') for key in status_fields])
        self._status_msgs_h = None
    
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
        if benchmark_every is None:
            benchmark_every = len(dataloader)
        if log_every is None:
            log_every = benchmark_every

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
            self.init_msgs()

        ## Initialize timers
        ## -----------------
        timers = dict(
            html = auxil.Timer(html_min_interval, reset=False),
            save = auxil.Timer(save_min_interval, reset=True),
        )

        ## Training loop
        ## -------------
        if n_epochs is not None:
            if n_steps is None:
                n_steps = n_epochs * len(dataloader)
            else:
                raise Exception('"n_steps" and "n_epochs" cannot be used simultaneously.')
        if n_steps is None:
            raise Exception('Either "n_steps" or "n_epochs" must be provided.')

        train_data_iter = iter(datasets.loop_loader(dataloader))
        for i_step in tqdm.trange(n_steps, ncols=0, disable=not model.ddp.is_main, mininterval=1.):
            last = i_step == n_steps - 1
            if model.ddp.is_main:
                self.set_msg('state', 'Training...')
                self.update_status(timers)

            ## Train step
            ## ----------
            batch = next(train_data_iter)
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
                benchmark_flag = ((i_step + 1) % benchmark_every == 0) or last
                if log_every is not None:
                    log_flag = (model.net.step % log_every == 0) or (model.net.step == 1) or benchmark_flag
                else:
                    log_flag = (model.net.step == 1) or benchmark_flag

                if log_flag:
                    model.net.eval()

                    self.set_msg('state', 'Running fixed batch...')
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
                        self.set_msg('state', 'Running benchmark...')
                        with auxil.EncapsulatedRandomState(random_seed + 2):
                            score = self.benchmark()  # pylint: disable=assignment-from-none
                        model.net.update_best(score)
                        if timers['save']:
                            if self.output_folder is not None:
                                self.set_msg('state', 'Saving...')
                                torch.save(model.state_dict(), os.path.join(self.output_folder, 'checkpoint.pt'))
                            timers['save'].reset()

                    if timers['html']:
                        self.set_msg('state', 'Writing HTML ...')
                        self.log_html()
                        timers['html'].reset()
                    model.net.train()

        if self.output_folder is not None:
            self.set_msg('state', 'Saving...')
            torch.save(model.state_dict(), os.path.join(self.output_folder, 'checkpoint.pt'))

    def update_status(self, timers):
        self.set_msg('step', f'Step: {self.model.net.step}')
        self.set_msg('next', 'Next: ' + ', '.join([f'{key}: {val}' for key, val in timers.items()]))
        self.set_msg('lr', f'Learning rate: {self.model.net.lr}')

    def split_batch(self, batch):
        return self.model.split_batch(batch, self.n_chunks)

    def base_step(self, batch):
        objective = None
        log = None
        return objective, log

    ## Logging
    ## -------
    @staticmethod
    def cat_logs(logs):
        log = {}
        return log

    def init_training_data(self, random_seed=0):
        self.model.training_data = {}

    def benchmark(self):
        score = None
        return score

    def log_step(self, train_log, fixed_log, valid_log, batch_log):
        pass

    def log_html(self):
        pass

    ## Console logging
    ## ---------------
    def init_msgs(self):
        self._status_msgs_h = {key: tqdm.tqdm([], desc=msg, bar_format='{desc}', mininterval=1.) for key, msg in self.status_msgs.items()}

    def set_msg(self, key, msg):
        self.status_msgs[key] = msg
        if self._status_msgs_h is not None:
            self._status_msgs_h[key].set_description_str(msg)