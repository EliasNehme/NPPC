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
    model = nppc.NPPCModel(
        restoration_model_folder='./results/mnist_inpainting/restoration/',
        net_type='unet',
        n_dirs=5,
        lr=1e-4,
        device=device,
    )

    ## Train
    ## -----
    trainer = nppc.NPPCTrainer(
        model=model,
        batch_size=256,
        output_folder='./results/mnist_inpainting/nppc/',
    )
    trainer.train(
        n_steps=10000,
        log_every=None,
    )

if __name__ == '__main__':
    main()
