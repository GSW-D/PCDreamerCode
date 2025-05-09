import argparse
import logging
import os
import numpy as np
import sys
import torch
from pprint import pprint
from config_pcn_svd import cfg
from core.train_pcn_svd import train_net
from core.test_pcn_svd import test_net
from core.inference_pcn_svd import infer_net
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
    parser.add_argument('--infer_root', help='Inference root dir', default='./demo_pcn/pts')
    args = parser.parse_args()

    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()
    print('cuda available ', torch.cuda.is_available())

    # Print config
    print('Use config:')
    pprint(cfg)

    if not args.test and not args.inference:
        train_net(cfg)
    else:
        if cfg.CONST.WEIGHTS is None:
            raise Exception('Please specify the path to checkpoint in the configuration file!')

        if args.test:
            test_net(cfg)
        else:
            infer_net(cfg, root=args.infer_root)


if __name__ == '__main__':
    # Check python version
    seed = 1
    set_seed(seed)
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    main()
