import os
import sys
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from trainer import GTETrainer
from utils.file_utils import make_dir



def load_config(config_path):
    config = Config(config_path)
    return config


def main(args):    
    config = load_config(args.config)
    config.checkpoint = args.checkpoint
    config.continuous = args.continuous

    if config.device in ['cpu', 'cuda']:
        single_train(args, config)
    else:
        # multi gpu
        raise NotImplementedError

    
def single_train(args, config):
    if config.device == 'cuda':
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    config.device = device

    trainer = GTETrainer(config)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        raise NotImplementedError


def multi_train():
    pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--continuous', '-con', action="store_true", help="Continuous training from checkpoint")
    parser.add_argument('--checkpoint', '-cp', type=str, help="Path of checkpoint")
    args = parser.parse_args()

    main(args)