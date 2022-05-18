'''
this code is for 1 node(computer) and multi gpus
use distributeddataparallel for multi gpu
'''

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from models import MLP, CNN, Resnet
# from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler

import argparse
import os


# set parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-3
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
epochs = 50


def get_dataset():
    # get mnist dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    data_train = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
    # data_test = datasets.MNIST(root='data/', train=False, transform=transforms, download=True)
    return data_train


def set_parser():
    parser = argparse.ArgumentParser(description='mnist multi gpu classification models')
    parser.add_argument('--lr', default=0.1, help='')
    parser.add_argument('--resume', default=None, help='')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='')
    parser.add_argument('--num_workers', type=int, default=4, help='')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    args = parser.parse_args()

    args.gpu_devices = torch.cuda.device_count()
    return args


def main():
    dataset = get_dataset()
    ngpus_per_node = torch.cuda.device_count()
    args = set_parser()
    args.num_workers = ngpus_per_node*4
    args.world_size = ngpus_per_node * args.world_size
    args.batch_size = args.batch_size//ngpus_per_node
    gpu_devices = ','.join([str(id) for id in range(args.gpu_devices)])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, dataset))


def main_worker(gpu, ngpus_per_node, args, dataset):
    args.gpu = gpu
    print(f"Use GPU: {args.gpu} for training")
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    # set gpu to network
    network = CNN()
    torch.cuda.set_device(args.gpu)
    network.cuda(args.gpu)
    network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[args.gpu])


    optimizer = optim.Adam(network.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss().to(device)
    # writer_real = SummaryWriter(f"logs/")

    network.train()
    for epoch in range(epochs):

        # use DistributedSampler to split dataset to each gpus
        train_sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank)
        train_sampler.epoch = epoch
        loader = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=(train_sampler is None), num_workers=args.num_workers,
                                  sampler=train_sampler)

        total_cost = 0
        for _, (X, Y) in enumerate(loader):
            X = X.cuda(device)
            Y = Y.cuda(device)

            optimizer.zero_grad()
            pred = network(X)
            cost = loss(pred, Y)
            cost.backward()
            optimizer.step()

            total_cost += cost

        average_cost = total_cost/len(dataset)
        print(f'epoch: {epoch}, average cost: {average_cost}')


if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        main()
    else:
        print(f"This Computer has {num_gpus} gpu")

