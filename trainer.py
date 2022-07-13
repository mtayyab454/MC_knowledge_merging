import copy
import os
import torch
import time
import torch.nn as nn
import torch.optim as optim

from utils import AverageAccumulator, VectorAccumulator, accuracy, Progressbar, adjust_learning_rate, get_num_parameters
from datasets import get_cifar_multitask
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

mse = nn.MSELoss()
# nn.SmoothL1Loss()
mse.cuda()

def train(trainloader, model, optimizer, criterion, out_id, keys):
    print('Training...')
    model.train()

    accumulator = VectorAccumulator(keys)
    end = time.time()

    for batch_idx, (inputs, _, targets, _) in enumerate(Progressbar(trainloader)):
        # measure data loading time
        # print(batch_idx)
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs[out_id], targets)

        # prec1 = sum(model_pred.squeeze(1) == targets)
        prec1, prec5 = accuracy(outputs[out_id].data, targets.data, topk=(1, 5))
        # gt_acc.update(prec1.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        accumulator.update( [(time.time() - end), prec1.item(), prec5.item(), loss.item()])
        end = time.time()

    return accumulator.avg

def test(testloader, model, criterion, out_id, keys, talk=False):
    print('Testing...')
    # switch to evaluate mode
    model.eval()

    accumulator = VectorAccumulator(keys)
    end = time.time()

    for batch_idx, (inputs, _, targets, _) in enumerate(Progressbar(testloader, talk=talk)):

        inputs, targets  = inputs.cuda(), targets.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs[out_id], targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs[out_id].data, targets.data, topk=(1, 5))

        accumulator.update( [(time.time() - end), prec1.item(), prec5.item(), loss.item()])

        end = time.time()

    return accumulator.avg

def testing_loop(model, args, task, keys):
    criterion = nn.CrossEntropyLoss()
    _, testloader = get_cifar_multitask(args.dataset, task_dict=task, split='test', batch_size=args.test_batch, num_workers=args.workers)
    test_stats = test(testloader, model, criterion, task['out_id'], keys)
    print('\nTest loss: %.4f \nVal accuracy: %.2f%%' % (test_stats[3], test_stats[1]))

def training_loop(model, logger, args, save_best=False):
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    ###################### Initialization ###################
    lr = args.lr_t1
    # Load data
    _, trainloader = get_cifar_multitask(args.dataset, task_dict=args.task1, split='train', batch_size=args.train_batch, num_workers=args.workers)
    _, testloader = get_cifar_multitask(args.dataset, task_dict=args.task1, split='test', batch_size=args.test_batch, num_workers=args.workers)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    num_param = get_num_parameters(model)

    print('    Total params: %.2fM' % (num_param / 1000000.0))
    logger.one_time({'num_param': num_param})

    ###################### Main Loop ########################
    best_acc = 0
    for epoch in range(args.epochs_t1):
        lr = adjust_learning_rate(optimizer, lr, epoch, args.schedule_t1, args.gamma)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs_t1, lr))

        train_stats = train(trainloader, model, optimizer, criterion, args.task1['out_id'], logger.keys)
        test_stats = test(testloader, model, criterion, args.task1['out_id'], logger.keys)

        torch.save(model.state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '.pth'))

        if best_acc < test_stats[1]:
            best_acc = test_stats[1]
            if save_best:
                torch.save(model.state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '_best.pth'))

        print('\nKeys: ', logger.keys)
        print('Training: ', train_stats)
        print('Testing: ', test_stats)
        print('Best Acc: ', best_acc)

        logger.append([lr, train_stats, test_stats])

    return model

def train_subtask(trainloader, model, model_copy, optimizer, criterion, args, keys):
    print('Training...')
    model.train()

    accumulator = VectorAccumulator(keys)
    end = time.time()

    for batch_idx, (inputs, _, task2_targets, _) in enumerate(Progressbar(trainloader)):
        # measure data loading time
        # print(batch_idx)
        inputs = inputs.cuda()
        task2_targets = task2_targets.cuda()

        with torch.no_grad():
            temp_outs = model_copy(inputs)
            task1_targets = temp_outs[args.task1['out_id']].data
        # compute output
        outputs = model(inputs)

        task1_pred = outputs[args.task1['out_id']]
        task2_pred = outputs[args.task2['out_id']]

        mse_err = mse(task1_pred, task1_targets)
        # print(mse_err)
        ce_err = criterion(task2_pred, task2_targets)
        loss =  ce_err + args.t1_weight*mse_err

        # prec1 = sum(model_pred.squeeze(1) == targets)
        prec1, prec5 = accuracy(task2_pred, task2_targets.data, topk=(1, 5))
        # gt_acc.update(prec1.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        accumulator.update( [(time.time() - end), prec1.item(), prec5.item(), loss.item()])
        end = time.time()

    return accumulator.avg

def training_loop_subtask(model, logger, args, save_best=False):
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    model_copy = copy.deepcopy(model)

    ###################### Initialization ###################
    lr = args.lr_t2
    # Load data
    _, trainloader_t2 = get_cifar_multitask(args.dataset, task_dict=args.task2['class_id'], split='train', batch_size=args.train_batch, num_workers=args.workers)
    _, testloader_t2 = get_cifar_multitask(args.dataset, task_dict=args.task2['class_id'], split='test', batch_size=args.test_batch, num_workers=args.workers)
    _, testloader_t1 = get_cifar_multitask(args.dataset, task_dict=args.task1['class_id'], split='test', batch_size=args.test_batch, num_workers=args.workers)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    num_param = get_num_parameters(model)

    print('    Total params: %.2fM' % (num_param / 1000000.0))
    logger.one_time({'num_param': num_param})

    ###################### Main Loop ########################
    best_acc = 0
    for epoch in range(args.epochs_t2):
        lr = adjust_learning_rate(optimizer, lr, epoch, args.schedule_t2, args.gamma)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs_t2, lr))

        train_stats = train_subtask(trainloader_t2, model, model_copy, optimizer, criterion, args, logger.keys)

        test_stats_t2 = test(testloader_t2, model, criterion, 1, logger.keys)
        test_stats_t1 = test(testloader_t1, model, criterion, 0, logger.keys)

        torch.save(model.state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '.pth'))

        if best_acc < test_stats_t2[1]:
            best_acc = test_stats_t2[1]
            if save_best:
                torch.save(model.state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '_best.pth'))

        print('\nKeys: ', logger.keys)
        print('Training: ', train_stats)
        print('Testing New Task: ', test_stats_t2)
        print('Testing Old Task: ', test_stats_t1)
        print('Best Acc: ', best_acc)

        logger.append([lr, train_stats, test_stats_t2, test_stats_t1])

    return model

def train_multitask(trainloader, models, optimizer, criterion, args, keys):
    print('Training...')
    models[0].train()
    models[1].eval()
    models[2].eval()

    accumulator = VectorAccumulator(keys)
    end = time.time()

    for batch_idx, (inputs, _, targets, task_id) in enumerate(Progressbar(trainloader)):
        # measure data loading time
        # print(batch_idx)
        inputs = inputs.cuda()
        targets = targets.cuda()

        with torch.no_grad():
            targets1 = models[1](inputs)
            targets2 = models[2](inputs)

        # compute output
        outputs = models[0](inputs)

        # loss = 0
        # for ti in task_id.unique():
        #     # idx = torch.where(task_id==ti)
        #     idx = np.where((task_id.numpy() == ti.numpy()))
        #     loss += criterion(outputs[ti-1][idx], targets[idx])

        loss = 0.0
        for i, ti in enumerate(task_id):
            if ti == 0: #blue
                loss += criterion(outputs[0][i:i+1], targets[i:i+1]) + \
                        args.mse_weight*mse(outputs[1][i], targets1[1][i]) + \
                        args.mse_weight*mse(outputs[2][i], targets2[1][i])
            elif ti == 1: #green
                loss += args.mse_weight*mse(outputs[0][i], targets1[0][i]) + \
                        criterion(outputs[1][i:i+1], targets[i:i+1]) + \
                        args.mse_weight*mse(outputs[2][i], targets2[1][i])
            elif ti == 2: #red
                loss += args.mse_weight*mse(outputs[0][i], targets2[0][i]) + \
                        args.mse_weight*mse(outputs[1][i], targets1[1][i]) + \
                        criterion(outputs[2][i:i+1], targets[i:i+1])

        # prec1 = sum(model_pred.squeeze(1) == targets)
        # prec1, prec5 = accuracy(outputs[task_id].data, targets.data, topk=(1, 5))
        # gt_acc.update(prec1.item())

        loss = loss/len(targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        accumulator.update( [(time.time() - end), 0, 0, loss.item()])
        end = time.time()

    return accumulator.avg

def training_loop_multitask(models, logger, args, save_best=False):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion.cuda()

    ###################### Initialization ###################
    lr = args.lr
    # Load data

    if args.include_t1_data:
        train_task_list = [args.task1, args.task2, args.task3]
        print('\nIncluding Task 1 data for training')
    else:
        train_task_list = [args.task2, args.task3]
        print('\nExcluding Task 1 data for training')

    _, trainloader = get_cifar_multitask(args.dataset, task_dict=train_task_list, split='train', batch_size=args.train_batch, num_workers=args.workers, sample_percent=args.data_avail)
    _, testloader_A = get_cifar_multitask(args.dataset, task_dict=args.task1, split='test', batch_size=args.test_batch, num_workers=args.workers)
    _, testloader_B = get_cifar_multitask(args.dataset, task_dict=args.task2, split='test', batch_size=args.test_batch, num_workers=args.workers)
    _, testloader_C = get_cifar_multitask(args.dataset, task_dict=args.task3, split='test', batch_size=args.test_batch, num_workers=args.workers)

    # optimizer = optim.SGD(models[0].parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.SGD([{'params': models[0].features.parameters(), 'lr': args.backbone_lr},
                           {'params': models[0].classifiers.parameters()}], lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    num_param = get_num_parameters(models[0])

    print('    Total params: %.2fM' % (num_param / 1000000.0))
    logger.one_time({'num_param': num_param})

    ###################### Main Loop ########################
    best_acc = 0
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, lr, epoch, args.schedule, args.gamma)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))

        train_stats = train_multitask(trainloader, models, optimizer, criterion, args, logger.keys)

        test_stats_A = test(testloader_A, models[0], criterion, 0, logger.keys, False)
        test_stats_B = test(testloader_B, models[0], criterion, 1, logger.keys, False)
        test_stats_C = test(testloader_C, models[0], criterion, 2, logger.keys, False)

        # torch.save(models[0].state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '.pth'), _use_new_zipfile_serialization=False)
        torch.save(models[0].state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '.pth'))

        if best_acc < test_stats_A[1]:
            best_acc = test_stats_A[1]
            if save_best:
                # torch.save(models[0].state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '_best.pth'), _use_new_zipfile_serialization=False)
                torch.save(models[0].state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '_best.pth'))

        print('\nKeys: ', logger.keys)
        print('Training: ', train_stats)
        print('Testing on Task A: ', test_stats_A)
        print('Testing on Task B: ', test_stats_B)
        print('Testing on Task C: ', test_stats_C)
        print('Best Acc: ', best_acc)

        logger.append([lr, train_stats, test_stats_A, test_stats_B, test_stats_C])

    return models[0]

def train_SSM(trainloader, models, optimizer, criterion, args, keys):
    print('Training...')
    models[0].train()
    models[1].eval()
    models[2].eval()

    accumulator = VectorAccumulator(keys)
    end = time.time()

    for batch_idx, inputs in enumerate(Progressbar(trainloader)):
        # measure data loading time
        # print(batch_idx)
        inputs = inputs[0].cuda()

        with torch.no_grad():
            targets1 = models[1](inputs)
            targets2 = models[2](inputs)

        # compute output
        outputs = models[0](inputs)

        loss = 0.0
        for i in range(inputs.shape[0]):
            rn = torch.rand(1)
            if rn.item() > 0.5 and args.use_rand: #blue
                loss += args.mse_weight*mse(outputs[0][i], targets1[0][i]) + \
                        args.mse_weight*mse(outputs[1][i], targets1[1][i]) + \
                        args.mse_weight*mse(outputs[2][i], targets2[1][i])
            else:
                loss += args.mse_weight*mse(outputs[0][i], targets2[0][i]) + \
                        args.mse_weight*mse(outputs[1][i], targets1[1][i]) + \
                        args.mse_weight*mse(outputs[2][i], targets2[1][i])

        # prec1 = sum(model_pred.squeeze(1) == targets)
        # prec1, prec5 = accuracy(outputs[task_id].data, targets.data, topk=(1, 5))
        # gt_acc.update(prec1.item())

        loss = loss/inputs.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        accumulator.update( [(time.time() - end), 0, 0, loss.item()])
        end = time.time()

    return accumulator.avg

def training_loop_SSM(models, logger, args, save_best=False):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion.cuda()

    ###################### Initialization ###################
    lr = args.lr
    # Load data

    if args.include_t1_data:
        train_task_list = [args.task1, args.task2, args.task3]
        print('\nIncluding Task 1 data for training')
    else:
        train_task_list = [args.task2, args.task3]
        print('\nExcluding Task 1 data for training')

########################################################################################################################
    if args.use_imagenet:
        print('\nUsing Imagenet for training')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        trainloader = torch.utils.data.DataLoader(
            datasets.ImageFolder('../../../data/ImageNet/val', transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers)
    else:
        _, trainloader = get_cifar_multitask(args.dataset, task_dict=train_task_list, split='train', batch_size=args.train_batch, num_workers=args.workers, sample_percent=args.data_avail)
########################################################################################################################
    _, testloader_A = get_cifar_multitask(args.dataset, task_dict=args.task1, split='test', batch_size=args.test_batch, num_workers=args.workers)
    _, testloader_B = get_cifar_multitask(args.dataset, task_dict=args.task2, split='test', batch_size=args.test_batch, num_workers=args.workers)
    _, testloader_C = get_cifar_multitask(args.dataset, task_dict=args.task3, split='test', batch_size=args.test_batch, num_workers=args.workers)

    # optimizer = optim.SGD(models[0].parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.SGD([{'params': models[0].features.parameters(), 'lr': args.backbone_lr},
                           {'params': models[0].classifiers.parameters()}], lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    num_param = get_num_parameters(models[0])

    print('    Total params: %.2fM' % (num_param / 1000000.0))
    logger.one_time({'num_param': num_param})

    ###################### Main Loop ########################
    best_acc = 0
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, lr, epoch, args.schedule, args.gamma)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))

        train_stats = train_SSM(trainloader, models, optimizer, criterion, args, logger.keys)

        test_stats_A = test(testloader_A, models[0], criterion, 0, logger.keys, False)
        test_stats_B = test(testloader_B, models[0], criterion, 1, logger.keys, False)
        test_stats_C = test(testloader_C, models[0], criterion, 2, logger.keys, False)

        # torch.save(models[0].state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '.pth'), _use_new_zipfile_serialization=False)
        torch.save(models[0].state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '.pth'))

        if best_acc < test_stats_A[1]:
            best_acc = test_stats_A[1]
            if save_best:
                # torch.save(models[0].state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '_best.pth'), _use_new_zipfile_serialization=False)
                torch.save(models[0].state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '_best.pth'))

        print('\nKeys: ', logger.keys)
        print('Training: ', train_stats)
        print('Testing on Task A: ', test_stats_A)
        print('Testing on Task B: ', test_stats_B)
        print('Testing on Task C: ', test_stats_C)
        print('Best Acc: ', best_acc)

        logger.append([lr, train_stats, test_stats_A, test_stats_B, test_stats_C])

    return models[0]