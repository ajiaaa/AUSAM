import argparse
import copy

import torch

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar10, Cifar100
# from data.tiny_imagenet import TinyImageNet
from utility.log import Log
from utility.initialize import initialize
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.scheduler import CosineScheduler, ProportionScheduler
import sys;

sys.path.append("..")
from sam.ausam import AUSAM
from model.resnet import ResNet18 as resnet18
from model.PyramidNet import PyramidNet as PYRM
import time
import random
import matplotlib.pyplot as plt
from utility.save_file import write_to_file, copy_files_to_folders, sivefile_config
from utility.time_record import TIME_RECORD
import numpy as np
from tqdm import tqdm
# torch.backends.cudnn.enable=True
# torch.backends.cuda.benchmark=True

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive",        default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size",      default=256, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth",           default=28, type=int, help="Number of layers.")
    parser.add_argument("--width_factor",    default=10, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--dropout",         default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs",          default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate",   default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum",        default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads",         default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho",             default=0.1, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--rho_max",         default=0.1, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--rho_min",         default=0.1, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay",    default=0.001, type=float, help="L2 weight decay.")

    parser.add_argument("--fmin",            default=0.1, type=float, help="")
    parser.add_argument("--fmax",            default=0.5, type=float, help="")
    parser.add_argument("--start_epoch",     default=0, type=int, help="")
    parser.add_argument("--end_epoch",       default=170, type=int, help="")
    parser.add_argument("--trans_epoch",     default=30, type=int, help="")

    parser.add_argument("--model",           default='wideresnet', type=str, help="resnet18, wideresnet, pyramidnet")
    parser.add_argument("--dataset",         default='cifar100', type=str, help="cifar10, cifar100, tinyimagenet")
    parser.add_argument("--storage_size",    default=50000, type=int, help="")
    parser.add_argument("--alpha",           default=0.5, type=float, help="")
    parser.add_argument("--beta",            default=0.5, type=float, help="")


    args = parser.parse_args()

    save_file_list, save_file_dir = sivefile_config("results/", args.dataset, args.model, "AUSAM")
    timer = TIME_RECORD()

    index_num = random.randint(1, 2000)
    # index_num = 42
    print('Seed:', index_num)
    write_to_file("other/whole_train_time.txt", 'Seed:'+str(index_num))
    initialize(args, seed=index_num)

    print('Cuda:', torch.cuda.is_available())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.device_count()

    if args.dataset == 'cifar100':
        class_num = 100
        args.storage_size = 50000
        dataset = Cifar100(args.batch_size, args.threads)
    elif args.dataset == 'cifar10':
        class_num = 10
        args.storage_size = 50000
        dataset = Cifar10(args.batch_size, args.threads)
    # elif args.dataset == 'tinyimagenet':
    #     class_num = 200
    #     args.storage_size = 100000
    #     dataset = TinyImageNet(args.batch_size, args.threads)

    log = Log(log_each=10)

    if args.model == 'resnet18':
        model = resnet18(num_classes=class_num).to(device)
    elif args.model == 'wideresnet':
        model = WideResNet(28, 10, args.dropout, in_channels=3, labels=class_num).to(device)
    elif args.model == 'pyramidnet':
        model = PYRM('cifar'+str(class_num), 110, 270, class_num, False).to(device)

    base_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                     weight_decay=args.weight_decay)
    scheduler = CosineScheduler(T_max=args.epochs * len(dataset.train), max_value=args.learning_rate, min_value=0.0,
                                optimizer=base_optimizer)
    rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.learning_rate, min_lr=0.0,
                                        max_value=args.rho_max, min_value=args.rho_min)

    optimizer = AUSAM(model.parameters(), base_optimizer, rho=args.rho, rho_scheduler=rho_scheduler, adaptive=args.adaptive,
                     storage_size=args.storage_size, alpha=args.alpha, beta=args.beta)


    whole_time = 0

    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()

        if epoch <= args.start_epoch + args.trans_epoch:
            fmax_ = args.fmax - ((args.start_epoch + args.trans_epoch - epoch) / args.trans_epoch) * (args.fmax - args.fmin) + 0.000000001
        # elif epoch >= args.end_epoch:
        #     args.fmax_ = args.fmax - ((epoch - args.end_epoch) / args.trans_epoch) * (args.fmax - args.fmin) + 0.01

        for batch in tqdm(dataset.train):


            inputs, targets, index = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            # print(targets)

            tf = optimizer.sampledata_index(args, epoch, index, fmax_)

            enable_running_stats(model)
            loss_bef = smooth_crossentropy(model(inputs[tf]), targets[tf], smoothing=args.label_smoothing)
            loss_bef.mean().backward()

            optimizer.first_step(zero_grad=True)
            disable_running_stats(model)
            loss_aft = smooth_crossentropy(model(inputs[tf]), targets[tf], smoothing=args.label_smoothing)
            loss_aft.mean().backward()

            optimizer.second_step_without_norm(zero_grad=True)

            roc = torch.abs(loss_aft - loss_bef)
            optimizer.impt_roc(epoch, index, tf, roc)


            with torch.no_grad():
                scheduler.step()
                #optimizer.update_rho_t()

        end_time = time.time()
        es_time = end_time - start_time
        whole_time += es_time
        write_to_file("other/whole_train_time.txt", str(whole_time))

        '''
        for ii in range(len(importent_record[:, 1])):
            select_time_list.append(importent_record[:, 1][ii].cpu().item())
            write_to_file("other/select_time_record.txt", str(select_time_list[ii]))
        plt.clf()
        fig = sns.kdeplot(select_time_list, fill=True, color="r", label='')
        plt.savefig("other/img/select_time_record.png")
        '''

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets, _ = batch[0].to(device), batch[1].to(device), batch[2]
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
            log.flush()

        write_to_file("other/accuracy.txt", str(log.acc))

        '''
        if args.epochs - epoch <= 15:
            torch.save(model, 'save_model/model_sam_ResNet_cifar100/sam_ResNet_cifar100_epoch'+ str(epoch) +'.pth')
        '''

    copy_files_to_folders(save_file_list, save_file_dir)

if __name__ == "__main__":
    for i in range(1):
        train()
