import os
import torch
import argparse
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument('-data', type=str, default='fashion')
parser.add_argument('-availability', type=str, default='always')  #always, periodic
parser.add_argument('-pretrained-model', type=str, default='')
parser.add_argument('-out', type=str, default='results.csv')

parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-lr-global', type=float, default=1.0)
parser.add_argument('-minibatch', type=int, default=16)

parser.add_argument('-lr-warmup', type=float, default=0.1)
parser.add_argument('-iters-warmup', type=int, default=0)

parser.add_argument('-iters-total', type=int, default=300000)
parser.add_argument('-seeds', type=str, default='1')  # e.g., 1,2,3

parser.add_argument('-iters-per-round', type=int, default=5)
parser.add_argument('-iters-per-eval', type=int, default=250)

parser.add_argument('-similarity', type=float, default=0.05)
parser.add_argument('-disconnect', type=int, default=400)
parser.add_argument('-total-workers', type=int, default=250)
parser.add_argument('-sampled-workers', type=int, default=10)

parser.add_argument('-gpu', type=int, default=1)  # 1 - use GPU if available; 0 - do not use GPU
parser.add_argument('-cuda-device', type=int, default=0)

parser.add_argument('-permute', type=int, default=1)

parser.add_argument('-save-checkpoint', type=int, default=1)
parser.add_argument('-iters-checkpoint', type=int, default=150000)

parser.add_argument('-wait-all', type=int, default=0)   # specifies whether to wait for all, after warm up
parser.add_argument('-full-batch', type=int, default=0)  # specifies whether to use full batch, after warm up

parser.add_argument('-p-value', type=int, default=-1)  # Use (active_rounds + inactive_rounds) if < 0

args = parser.parse_args()

print(', '.join(f'{k}={v}' for k, v in vars(args).items()))


use_gpu = bool(args.gpu)
use_gpu = use_gpu and torch.cuda.is_available()
device = torch.device('cuda:' + str(args.cuda_device)) if use_gpu else torch.device('cpu')

use_permute = bool(args.permute)
wait_for_all = bool(args.wait_all)
use_full_batch = bool(args.full_batch)

use_global_update = bool(args.lr_global > 1.0)
assert (args.lr_global >= 1.0)

save_checkpoint = bool(args.save_checkpoint)
iters_checkpoint = args.iters_checkpoint

if args.data == 'fashion':
    dataset = 'FashionMNIST'
    model_name = 'ModelCNNMnist'
elif args.data == 'cifar' or args.data == 'cifar10':
    dataset = 'CIFAR10'
    model_name = 'ModelCNNCifar10'
else:
    raise Exception('Unknown data name')

max_iter = args.iters_total

simulations_str = args.seeds.split(',')
simulations = [int(i) for i in simulations_str]

dataset_file_path = os.path.join(os.path.dirname(__file__), 'data_files')

if args.availability == 'periodic':
    periodic_availability = True
else:
    periodic_availability = False

if wait_for_all:
    assert periodic_availability  # wait_for_all can only be used with periodic_availability

mixing_ratio = args.similarity

active_rounds = int(args.disconnect / float(4))
inactive_rounds = args.disconnect

if args.p_value < 0:
    p_value = active_rounds + inactive_rounds
    print('Actual p_value:', p_value)
else:
    p_value = args.p_value

n_nodes = args.total_workers
n_nodes_in_each_round = args.sampled_workers
step_size_local = args.lr  # learning rate of clients
step_size_global = args.lr_global

step_size_warmup = args.lr_warmup
iters_warmup = args.iters_warmup

batch_size_train = args.minibatch
batch_size_eval = 256

iters_per_round = args.iters_per_round  # number of iterations in local training
min_iters_per_eval = args.iters_per_eval

results_file = args.out
save_model_file = results_file + '.model'
if args.pretrained_model != '':
    load_model_file = args.pretrained_model
else:
    load_model_file = None

if dataset == 'CIFAR10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
else:
    transform_train = None