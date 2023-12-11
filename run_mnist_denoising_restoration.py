#!python
import os
import argparse
import socket

import nppc

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_group = parser.add_argument_group('model')
    parser_group.add_argument(f'--device', default='cuda:0', type=str)
    args = parser.parse_args()

    device = args.device

    print('Running ...')
    print(f'Hostname: {socket.gethostname()}-{device}')
    print(f'Process ID: {os.getgid()}')

    ## Define model
    ## ------------
    model = nppc.RestorationModel(
        dataset='mnist',
        data_folder=os.path.join(os.environ['HOME'], 'datasets/'),
        distortion_type='denoising_1',
        net_type='unet',
        lr=1e-3,
        device=device,
    )

    ## Train
    ## -----
    trainer = nppc.RestorationTrainer(
        model=model,
        batch_size=256,
        output_folder='./results/mnist_denoising/restoration/',
        max_benchmark_samples=256,
    )
    trainer.train(
        n_steps=150000,
        log_every=None,
    )

if __name__ == '__main__':
    main()
