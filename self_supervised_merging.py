from __future__ import print_function

import sys
sys.path.insert(1, '../')
from models.vgg_multitask import vgg16_multitask

import os
import random
import argparse
import torch.nn.parallel
import torch.nn as nn

from utils import Logger, get_time_str, create_dir, backup_code
from trainer import testing_loop, training_loop, training_loop_subtask, training_loop_SSM

from task_utils import get_task_sets

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

model_names = ['vgg16', ]

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets

parser.add_argument('--jobid', type=str, default='ABC_100')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg16)')
parser.add_argument('--task_set', default='vehicles10-large_animals10-reset80', choices=['vehicles10-manmade_objects15-reset75', 'vehicles10-manmade_objects10-reset80', 'vehicles10-large_animals10-reset80'])
parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--mse_weight', default=0.01, type=float)

# Task2 options
parser.add_argument('--include_t1_data', type=str2bool, nargs='?', const=True, default=False, help='')
parser.add_argument('--use_imagenet', type=str2bool, nargs='?', const=True, default=False, help='')
parser.add_argument('--use_rand', type=str2bool, nargs='?', const=True, default=False, help='')
parser.add_argument('--data_avail', default=100, type=int)
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--schedule', type=int, nargs='+', default=[15, 30],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--lr', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--backbone_lr', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--logs', default='logs', type=str, metavar='PATH',
                    help='path to save the training logs (default: logs)')
# Architecture
# Miscs
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

def main():
    print(args)
    exp_name = args.jobid + '_' + args.arch + '_' + args.arch
    checkpoint_dir = os.path.join(args.checkpoint, exp_name)
    create_dir([checkpoint_dir, args.logs])

    task_idx, cp_name = get_task_sets(args.task_set)

    args.task1 = {'class_id': task_idx[0], 'out_id':0, 'tid':0}
    args.task2 = {'class_id': task_idx[1], 'out_id': 1, 'tid':1}
    args.task3 = {'class_id': task_idx[2], 'out_id': 2, 'tid':2}

    model_A = vgg16_multitask([args.task1['class_id'].__len__()])
    sd = torch.load(os.path.join('checkpoint', cp_name[0] + '_vgg16_t1', cp_name[0] + '_vgg16_t1_best.pth'))
    model_A.load_state_dict(sd)
    model_A.cuda()

    model_AB = vgg16_multitask([args.task1['class_id'].__len__(), args.task2['class_id'].__len__()])
    sd = torch.load(os.path.join('checkpoint', cp_name[1] + '_vgg16_t2', cp_name[1] + '_vgg16_t2.pth'))
    model_AB.load_state_dict(sd)
    model_AB.cuda()

    model_AC = vgg16_multitask([args.task1['class_id'].__len__(), args.task3['class_id'].__len__()])
    sd = torch.load(os.path.join('checkpoint', cp_name[2] + '_vgg16_t2', cp_name[2] + '_vgg16_t2.pth'))
    model_AC.load_state_dict(sd)
    model_AC.cuda()

    model_ABC = vgg16_multitask([args.task1['class_id'].__len__(), args.task2['class_id'].__len__(), args.task3['class_id'].__len__()])
    model_ABC.features.load_state_dict(model_A.features.state_dict())
    model_ABC.classifiers[0].load_state_dict(model_A.classifiers[0].state_dict())
    model_ABC.classifiers[1].load_state_dict(model_AB.classifiers[1].state_dict())
    model_ABC.classifiers[2].load_state_dict(model_AC.classifiers[1].state_dict())
    model_ABC.cuda()

    logger = Logger(dir_path=args.logs, fname=exp_name,
                    keys=['time', 'acc1', 'acc5', 'ce_loss'])
    logger.one_time({'seed':args.manualSeed, 'comments': 'Sample percent: 100'})
    logger.set_names(['lr', 'train_stats', 'test_stats_A', 'test_stats_B', 'test_stats_C'])

    # print('Model AB of task1')
    # testing_loop(model=model_AB, args=args, task=args.task1, keys=logger.keys)
    # print('Model AB of task2')
    # testing_loop(model=model_AB, args=args, task=args.task2, keys=logger.keys)
    # print('Model AC of task1')
    # testing_loop(model=model_AC, args=args, task=args.task1, keys=logger.keys)
    # print('Model AC of task3')
    # testing_loop(model=model_AC, args=args, task={'class_id': task_idx[2], 'out_id': 1, 'tid':1}, keys=logger.keys)

    testing_loop(model=model_ABC, args=args, task=args.task1, keys=logger.keys)
    testing_loop(model=model_ABC, args=args, task=args.task2, keys=logger.keys)
    testing_loop(model=model_ABC, args=args, task=args.task3, keys=logger.keys)

    training_loop_SSM(models=[model_ABC, model_AB, model_AC], logger=logger, args=args, save_best=True)
    print('\n', args)

if __name__ == '__main__':
    main()