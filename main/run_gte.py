import os
import sys
from sconf import Config
from argparse import ArgumentParser

import torch.distributed
import torch.multiprocessing.spawn
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from trainer import GTETrainer, HfTrainer
from utils import setup_env, hf_setup
from utils.train_utils import get_dataset


def load_config(config_path):
    config = Config(config_path)
    return config


def main(args):    
    config = load_config(args.config)
    config.checkpoint = args.checkpoint
    config.continuous = args.continuous

    setup_env(config)

    if len(config.device) <= 1 or config.device in ['cpu', 'cuda']:
        single_train(args, config)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.device))
        ngpus_per_node = len(config.device)
        torch.multiprocessing.spawn(multi_train, nprocs=ngpus_per_node, args=(ngpus_per_node, config, args))


def single_train(args, config):
    if config.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device("cuda:0") if config.device == "cuda" else torch.device(f"cuda:{config.device[0]}")

    if config.use_hf_trainer:
        modes = ['train', 'valid'] if config.do_eval else ['train']
        dict_dataset, tokenizer = get_dataset(config, modes)

        model, training_args, data_collator = hf_setup(config, tokenizer)

        trainer = HfTrainer(
            model = model,
            train_dataset=dict_dataset['train'],
            eval_dataset=dict_dataset['valid']if 'valid' in dict_dataset else None,
            args = training_args,
            data_collator=data_collator,
        )

    else:
        trainer = GTETrainer(
            config,
            device,
            )

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        raise NotImplementedError


def multi_train(rank, ngpus_per_node, config, args):
    
    torch.distributed.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)

    torch.cuda.set_device(rank)
    torch.distributed.barrier()

    trainer = GTETrainer(
        config,
        rank,
    )

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        raise NotImplementedError


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--continuous', '-con', action="store_true", help="Continuous training from checkpoint")
    parser.add_argument('--checkpoint', '-cp', type=str, help="Path of checkpoint")
    args = parser.parse_args()

    main(args)


# if __name__ == "__main__":
#     main()
    # python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=<0 or 1> --master_addr="master_node_ip" --master_port=12345 script.py