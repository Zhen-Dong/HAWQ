import argparse
import os
import random
import shutil
import time
import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from bit_config import *
from utils import *
from pytorchcv.model_provider import get_model as ptcv_get_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture')
parser.add_argument('--teacher-arch',
                    type=str,
                    default='resnet101',
                    help='teacher network used to do distillation')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--act-range-momentum',
                    type=float,
                    default=-1,
                    help='momentum of the activation range moving average, '
                         '-1 stands for using minimum of min and maximum of max')
parser.add_argument('--quant-mode',
                    type=str,
                    default='symmetric',
                    choices=['asymmetric', 'symmetric'],
                    help='quantization mode')
parser.add_argument('--save-path',
                    type=str,
                    default='checkpoints/imagenet/test/',
                    help='path to save the quantized model')
parser.add_argument('--data-percentage',
                    type=float,
                    default=1,
                    help='data percentage of training data')
parser.add_argument('--fix-BN',
                    action='store_true',
                    help='whether to fix BN statistics and fold BN during training')
parser.add_argument('--fix-BN-threshold',
                    type=int,
                    default=None,
                    help='when to start training with fixed and folded BN,'
                         'after the threshold iteration, the original fix-BN will be overwritten to be True')
parser.add_argument('--checkpoint-iter',
                    type=int,
                    default=-1,
                    help='the iteration that we save all the featuremap for analysis')
parser.add_argument('--evaluate-times',
                    type=int,
                    default=-1,
                    help='The number of evaluations during one epoch')
parser.add_argument('--quant-scheme',
                    type=str,
                    default='uniform4',
                    help='quantization bit configuration')
parser.add_argument('--resume-quantize',
                    action='store_true',
                    help='if True map the checkpoint to a quantized model,'
                         'otherwise map the checkpoint to an ordinary model and then quantize')
parser.add_argument('--act-percentile',
                    type=float,
                    default=0,
                    help='the percentage used for activation percentile'
                         '(0 means no percentile, 99.9 means cut off 0.1%)')
parser.add_argument('--weight-percentile',
                    type=float,
                    default=0,
                    help='the percentage used for weight percentile'
                         '(0 means no percentile, 99.9 means cut off 0.1%)')
parser.add_argument('--channel-wise',
                    action='store_false',
                    help='whether to use channel-wise quantizaiton or not')
parser.add_argument('--bias-bit',
                    type=int,
                    default=32,
                    help='quantizaiton bit-width for bias')
parser.add_argument('--distill-method',
                    type=str,
                    default='None',
                    help='you can choose None or KD_naive')
parser.add_argument('--distill-alpha',
                    type=float,
                    default=0.95,
                    help='how large is the ratio of normal loss and teacher loss')
parser.add_argument('--temperature',
                    type=float,
                    default=6,
                    help='how large is the temperature factor for distillation')
parser.add_argument('--fixed-point-quantization',
                    action='store_true',
                    help='whether to skip deployment-oriented operations and '
                         'use fixed-point rather than integer-only quantization')

best_acc1 = 0
quantize_arch_dict = {'resnet50': q_resnet50, 'resnet50b': q_resnet50,
                      'resnet18': q_resnet18, 'resnet101': q_resnet101,
                      'inceptionv3': q_inceptionv3,
                      'mobilenetv2_w1': q_mobilenetv2_w1}
args = parser.parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

hook_counter = args.checkpoint_iter
hook_keys = []
hook_keys_counter = 0

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filename=args.save_path + 'log.log')
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

logging.info(args)


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logging.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained and not args.resume:
        logging.info("=> using pre-trained PyTorchCV model '{}'".format(args.arch))
        model = ptcv_get_model(args.arch, pretrained=True)
        if args.distill_method != 'None':
            logging.info("=> using pre-trained PyTorchCV teacher '{}'".format(args.teacher_arch))
            teacher = ptcv_get_model(args.teacher_arch, pretrained=True)
    else:
        logging.info("=> creating PyTorchCV model '{}'".format(args.arch))
        model = ptcv_get_model(args.arch, pretrained=False)
        if args.distill_method != 'None':
            logging.info("=> creating PyTorchCV teacher '{}'".format(args.teacher_arch))
            teacher = ptcv_get_model(args.teacher_arch, pretrained=False)

    if args.resume and not args.resume_quantize:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume)['state_dict']
            model_key_list = list(model.state_dict().keys())
            for key in model_key_list:
                if 'num_batches_tracked' in key: model_key_list.remove(key)
            i = 0
            modified_dict = {}
            for key, value in checkpoint.items():
                if 'scaling_factor' in key: continue
                if 'num_batches_tracked' in key: continue
                if 'weight_integer' in key: continue
                if 'min' in key or 'max' in key: continue
                modified_key = model_key_list[i]
                modified_dict[modified_key] = value
                i += 1
            logging.info(model.load_state_dict(modified_dict, strict=False))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    quantize_arch = quantize_arch_dict[args.arch]

    model = quantize_arch(model)

    bit_config = bit_config_dict["bit_config_" + args.arch + "_" + args.quant_scheme]
    name_counter = 0

    for name, m in model.named_modules():
        if name in bit_config.keys():
            name_counter += 1
            setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', args.bias_bit)
            setattr(m, 'quantize_bias', (args.bias_bit != 0))
            setattr(m, 'per_channel', args.channel_wise)
            setattr(m, 'act_percentile', args.act_percentile)
            setattr(m, 'act_range_momentum', args.act_range_momentum)
            setattr(m, 'weight_percentile', args.weight_percentile)
            setattr(m, 'fix_flag', False)
            setattr(m, 'fix_BN', args.fix_BN)
            setattr(m, 'fix_BN_threshold', args.fix_BN_threshold)
            setattr(m, 'training_BN_mode', args.fix_BN)
            setattr(m, 'checkpoint_iter_threshold', args.checkpoint_iter)
            setattr(m, 'save_path', args.save_path)
            setattr(m, 'fixed_point_quantization', args.fixed_point_quantization)

            if type(bit_config[name]) is tuple:
                bitwidth = bit_config[name][0]
                if bit_config[name][1] == 'hook':
                    m.register_forward_hook(hook_fn_forward)
                    global hook_keys
                    hook_keys.append(name)
            else:
                bitwidth = bit_config[name]

            if hasattr(m, 'activation_bit'):
                setattr(m, 'activation_bit', bitwidth)
                if bitwidth == 4:
                    setattr(m, 'quant_mode', 'asymmetric')
            else:
                setattr(m, 'weight_bit', bitwidth)

    logging.info("match all modules defined in bit_config: {}".format(len(bit_config.keys()) == name_counter))
    logging.info(model)

    if args.resume and args.resume_quantize:
        if os.path.isfile(args.resume):
            logging.info("=> loading quantized checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)['state_dict']
            modified_dict = {}
            for key, value in checkpoint.items():
                if 'num_batches_tracked' in key: continue
                if 'weight_integer' in key: continue
                if 'bias_integer' in key: continue

                modified_key = key.replace("module.", "")
                modified_dict[modified_key] = value
            model.load_state_dict(modified_dict, strict=False)
        else:
            logging.info("=> no quantized checkpoint found at '{}'".format(args.resume))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            if args.distill_method != 'None':
                teacher.cuda(args.gpu)
                teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            if args.distill_method != 'None':
                teacher.cuda()
                teacher = torch.nn.parallel.DistributedDataParallel(teacher)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if args.distill_method != 'None':
            teacher = teacher.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            # teacher is not alexnet or vgg
            if args.distill_method != 'None':
                teacher = torch.nn.DataParallel(teacher).cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            if args.distill_method != 'None':
                teacher = torch.nn.DataParallel(teacher).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume optimizer and meta information from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=> loaded optimizer and meta information from checkpoint '{}' (epoch {})".
                         format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_resolution = 224
    if args.arch == "inceptionv3":
        train_resolution = 299

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(train_resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataset_length = int(len(train_dataset) * args.data_percentage)
    if args.data_percentage == 1:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    else:
        partial_train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                                 [dataset_length, len(train_dataset) - dataset_length])
        train_loader = torch.utils.data.DataLoader(
            partial_train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_resolution = (256, 224)
    if args.arch == 'inceptionv3':
        test_resolution = (342, 299)

    # evaluate on validation set
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(test_resolution[0]),
            transforms.CenterCrop(test_resolution[1]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if args.distill_method != 'None':
            train_kd(train_loader, model, teacher, criterion, optimizer, epoch, val_loader,
                     args, ngpus_per_node, dataset_length)
        else:
            train(train_loader, model, criterion, optimizer, epoch, args)

        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        logging.info(f'Best acc at epoch {epoch}: {best_acc1}')
        if is_best:
            # record the best epoch
            best_epoch = epoch

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save_path)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if args.fix_BN == True:
        model.eval()
    else:
        model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def train_kd(train_loader, model, teacher, criterion, optimizer, epoch, val_loader, args, ngpus_per_node,
             dataset_length):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if args.fix_BN == True:
        model.eval()
    else:
        model.train()
    teacher.eval()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        if args.distill_method != 'None':
            with torch.no_grad():
                teacher_output = teacher(images)

        if args.distill_method == 'None':
            loss = criterion(output, target)
        elif args.distill_method == 'KD_naive':
            loss = loss_kd(output, target, teacher_output, args)
        else:
            raise NotImplementedError

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if i % args.print_freq == 0 and args.rank == 0:
            print('Epoch {epoch_} [{iters}]  Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(epoch_=epoch, iters=i,
                                                                                               top1=top1, top5=top5))

        if i % ((dataset_length // (
                args.batch_size * args.evaluate_times)) + 2) == 0 and i > 0 and args.evaluate_times > 0:
            acc1 = validate(val_loader, model, criterion, args)

            # switch to train mode
            if args.fix_BN == True:
                model.eval()
            else:
                model.train()

            # remember best acc@1 and save checkpoint
            global best_acc1
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, args.save_path)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    freeze_model(model)
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    torch.save({'convbn_scaling_factor': {k: v for k, v in model.state_dict().items() if 'convbn_scaling_factor' in k},
                'fc_scaling_factor': {k: v for k, v in model.state_dict().items() if 'fc_scaling_factor' in k},
                'weight_integer': {k: v for k, v in model.state_dict().items() if 'weight_integer' in k},
                'bias_integer': {k: v for k, v in model.state_dict().items() if 'bias_integer' in k},
                'act_scaling_factor': {k: v for k, v in model.state_dict().items() if 'act_scaling_factor' in k},
                }, args.save_path + 'quantized_checkpoint.pth.tar')

    unfreeze_model(model)

    return top1.avg


def save_checkpoint(state, is_best, filename=None):
    torch.save(state, filename + 'checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename + 'checkpoint.pth.tar', filename + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    print('lr = ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def loss_kd(output, target, teacher_output, args):
    """
    Compute the knowledge-distillation (KD) loss given outputs and labels.
    "Hyperparameters": temperature and alpha
    The KL Divergence for PyTorch comparing the softmaxs of teacher and student.
    The KL Divergence expects the input tensor to be log probabilities.
    """
    alpha = args.distill_alpha
    T = args.temperature
    KD_loss = F.kl_div(F.log_softmax(output / T, dim=1), F.softmax(teacher_output / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(output, target) * (1. - alpha)

    return KD_loss


if __name__ == '__main__':
    main()
