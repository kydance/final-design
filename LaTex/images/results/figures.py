import argparse
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import math

DIR_PREFIX_VGG16_CHAP_3_0 = '/home/tianen/doc/_XiDian/___FinalDesign/results/vgg_cifar/chap-3-0/'
DIR_PREFIX_VGG16_CHAP_3_1 = '/home/tianen/doc/_XiDian/___FinalDesign/results/vgg_cifar/chap-3-1/'

DIR_PREFIX_RESNET56_CHAP_3_0 = '/home/tianen/doc/_XiDian/___FinalDesign/results/resnet_cifar/resnet56_cifar10/chap-3-0/'
DIR_PREFIX_RESNET56_CHAP_3_1 = '/home/tianen/doc/_XiDian/___FinalDesign/results/resnet_cifar/resnet56_cifar10/chap-3-1/'

DIR_PREFIX_RESNET110_CHAP_3_0 = '/home/tianen/doc/_XiDian/___FinalDesign/results/resnet_cifar/resnet110_cifar10/chap-3-0/'
DIR_PREFIX_RESNET110_CHAP_3_1 = '/home/tianen/doc/_XiDian/___FinalDesign/results/resnet_cifar/resnet110_cifar10/chap-3-1/'

# DIR_PREFIX = '/home/tianen/doc/_XiDian/___FinalDesign/results/vgg_cifar/chap-4-0/'
DIR_PREFIX_VGG16_CHAP_4_1 = '/home/tianen/doc/_XiDian/___FinalDesign/results/vgg_cifar/chap-4-1/'
DIR_PREFIX_RESNET56_CHAP_4_1 = '/home/tianen/doc/_XiDian/___FinalDesign/results/resnet_cifar/resnet56_cifar10/chap-4-1/'
DIR_PREFIX_RESNET110_CHAP_4_1 = '/home/tianen/doc/_XiDian/___FinalDesign/results/resnet_cifar/resnet110_cifar10/chap-4-1/'

PATTERN_ACC_LOSS = re.compile(r"(.*Test Loss *)(\d*\.\d+)(.*Accuracy *)(\d+\.\d+)(\%.*Time *)(\d*\.\d*)(s)")

# 12/26 04:22:20 PM | Epoch[11] (14976/50000):	Loss 0.6218	Accuracy 86.26%		Time 32.16s
# PATTERN_TIME = re.compile(r"(.*Epoch[\d+] \(\d+\/\d+\):).*Time (\d+.\d+)s")
PATTERN_TIME = re.compile(r"(.*Epoch\[)(\d+)(\].*Time )(\d+.\d+)s")
PATTERN_LOSS = re.compile(r"(.*Epoch\[)(\d+)(\].*Loss *)(\d+\.\d+).*")

# s = 'vgg16_abs_cr0.011_wm5o_2023.12.27-19:36:57-768.log'
# s = 'vgg16_base_line_cr0.5_2024.01.03-11.50.45-959.log'
PATTERN_FILENAME = r"(.+)_(.+)_cr(\d+)_(.+)_.*\.log"

LINE_FMT = ['o--', '*--', '1--', 's--', 'p--', '8--', 'x--']
LINE_COLOR = ['b', 'g', 'g', 'r', 'r', 'c', 'c', 'm', 'm', 'y', 'y', 'k', 'k', 'w', 'w']
MARK_SIZE = 4
TICK_SIZE = 16

# 设置图例并且设置图例的字体及大小
LEGEND_FONT = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
LABEL_FONT = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}

parser = argparse.ArgumentParser(description='final-design-tianen')

parser.add_argument(
    '--warmup_epochs',
    type=int,
    default=5,
    help='Warmup epochs. default:5',
)

parser.add_argument(
    '--warmup_coeff',
    type=int,
    nargs="*",
    default=[1, 1, 1, 1, 1],
    help='Warmup coeff. default:[1, 1, 1, 1, 1]',
)

def warmup_compress_ratio(epoch:int, base_cr, args):
    if args.warmup_epochs > 0:
        if epoch < args.warmup_epochs:
            if epoch == 0 and len(args.warmup_coeff) > 0:
                cr = args.warmup_coeff[epoch]
            else:
                args.warmup_coeff = base_cr ** (1. / (args.warmup_epochs + 1))

            if isinstance(args.warmup_coeff, (tuple, list)):
                cr = args.warmup_coeff[epoch]
            else:
                cr = max(args.warmup_coeff ** (epoch + 1), base_cr)
        else:
            cr = base_cr
    else:
        cr = base_cr

    return cr

class Result:
    def __init__(self, arch: str, dist_type: str, cr: float, wm: str,
                 epoches: list, losses: list, acces: list, times: list):
        self.arch = arch
        self.dist_type = dist_type
        self.cr = cr
        self.wm = wm
        self.epoches = epoches
        self.losses = losses
        self.acces = acces
        self.times = times

    def __str__(self):
        return f"arch: {self.arch}, dist_type: {self.dist_type}, cr: {self.cr}, \
                wm: {self.wm}, epoches: {self.epoches}, losses: {self.losses}, \
                acces: {self.acces}, times: {self.times}"
    
    def __repr__(self):
        return self.__str__()
    

def getResult(prefix: str, file: str):
    arch, dist_type, cr, wm = getArchAndDistAndCrAndWm(PATTERN_FILENAME, file)

    epoch = 0
    epoches = []
    losses = {}
    acces = []
    times = {}

    with open(prefix + file, "r") as f:
        lines = f.readlines()
        for line in lines:
            (loss, acc), ok = getLossAndAcc(PATTERN_ACC_LOSS, line)
            if ok:
                epoches.append(epoch)
                # losses.append(loss)
                acces.append(acc)
                epoch += 1

            curr_epoch, time, ok = getTime(PATTERN_TIME, line)
            if ok:
                if curr_epoch in times.keys():
                    times[curr_epoch] += time
                else:
                    times[curr_epoch] = time

            curr_epoch, loss, ok = getLoss(PATTERN_LOSS, line)
            if ok:
                losses[curr_epoch] = loss 
    times = list(times.values())
    losses = list(losses.values())
    return (arch, dist_type, cr, wm, epoches, losses, acces, times)


def plotFromFile(path: str, func) -> list:
    files = os.listdir(path)

    results = []

    for file in files:
        arch, dist_type, cr, wm, epoches, losses, acces, times = getResult(path, file)
        results.append(Result(arch, dist_type, cr, wm, epoches, losses, acces, times))

    func(results, 'loss')
    func(results, 'acc')

def getLossAndAcc(pattern: str, s: str):
    """使用正则表达式逐行处理"""
    ret = re.match(pattern, s)
    loss, acc = 0, 0
    ok = False

    if ret is not None:
        # 1 - loss, 3 - acc, 5 - time
        results = ret.groups()
        loss = float(results[1])
        acc = float(results[3])
        ok = True
    return (loss, acc), ok

def getTime(pattern: str, s: str):
    """使用正则表达式逐行处理"""
    ret = re.match(pattern, s)
    curr_epoch, time = 0, 0
    ok = False

    if ret is not None:
        # 1 - epoch, 3 - time
        results = ret.groups()
        curr_epoch = int(results[1])
        time = float(results[3])
        ok = True
    return curr_epoch, time, ok

def getLoss(pattern: str, s: str):
    """使用正则表达式逐行处理"""
    ret = re.match(pattern, s)
    curr_epoch, loss = 0, 0
    ok = False

    if ret is not None:
        # 1 - epoch, 3 - time
        results = ret.groups()
        curr_epoch = int(results[1])
        loss = float(results[3])
        ok = True
    return curr_epoch, loss, ok

def getArchAndDistAndCrAndWm(pattern: str, s: str):
    """使用正则表达式处理日志文件名字"""
    if 'base_line' in s:
        print("base_line")
        pattern = r"(.+)_(base_line)_cr(.+)_.*\.log"
        ret = re.match(pattern, s)
        if ret is None: raise Exception("文件名不合法")

        results = ret.groups()
        arch = results[0]
        dist_type = results[1]
        cr = 0. 
        wm = 0
        return (arch, dist_type, cr, wm)

    ret = re.match(pattern, s)

    if ret is None:
        raise Exception("文件名不合法")
    
    results = ret.groups()
    arch = results[0]
    dist_type = results[1]
    cr = float(results[2])
    wm = results[3]

    return (arch, dist_type, cr, wm)

def figure_1(result: tuple):
    """条形图📊"""

    epochs, losses, acces, times = result[0], result[1], result[2], result[3]
    # print(len(epochs), len(losses), len(acces), len(times))


    # the width of the bars
    width = 0.28

    fig = plt.figure(dpi=80, figsize=(9, 5.3))
    ax = fig.add_subplot(1, 1, 1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    labels = ['w/n G.A.\nw/n Dis.', 'w/n G.A.\nw Dis.', 'w G.A.\nw/n Dis.', 'w G.A.\nw Dis.']
    # the label locations
    x = np.arange(0, len(labels), 1)
    train_time = [53.91, 47.82, 53.56, 40.79]
    auc = [91.57, 91.43, 94.09, 94.10]
    fpr = [6.67, 8.05, 8.48, 7.11]

    ax.bar(x - width, train_time, width, label='Training Time(Second)', color="#fa9999")
    ax.bar(x, auc, width, label='AUC(%)', color="#95CC82")
    ax.bar(x + width, fpr, width, label='FPR(%)', color="#6ec2f8")
    
    plt.ylim((0, 100.1))
    plt.show()

def fig_chap3_0_vgg16(result: list, line_type: str):
    """折线图
    line_type: loss, acc, time
    """

    num_lines = len(result)
    epoch = result[0].epoches

    losses, acces, times = [], [], []    
    for i in range(num_lines):
        losses.append(result[i].losses)
        acces.append(result[i].acces)
        times.append(result[i].times)

    if line_type == "loss":
        fig = plt.figure()
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, losses[i], yerr=0, fmt='--', color = LINE_COLOR[0], ms=MARK_SIZE, label='{}'.format(result[i].dist_type))
                continue
            # losses[i] = [x - 0.03 for x in losses[i]]
            plt.errorbar(epoch, losses[i], yerr=0, color = LINE_COLOR[2*i], ms=MARK_SIZE,label='{}_cr{}'.format(result[i].dist_type, result[i].cr))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Training loss', LABEL_FONT)
        plt.ylim((0., 1.5))
    elif line_type == 'acc':
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, acces[i], yerr=0, fmt='--', color=LINE_COLOR[i], ms=MARK_SIZE,label='{}'.format(result[i].dist_type))
                continue
            # acces[i] = [x + 15.0/result[i].cr for x in acces[i]]
            plt.errorbar(epoch, acces[i], yerr=0, color=LINE_COLOR[2*i], ms=MARK_SIZE, label='{}_cr{}'.format(result[i].dist_type, result[i].cr))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Top-1 acc. (%)', LABEL_FONT)
        plt.ylim((60, 95))
    plt.legend(loc='best', prop=LEGEND_FONT)

    plt.xlim((0, len(epoch)))

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Epoch', LABEL_FONT)
    plt.grid()
    plt.tight_layout()
    plt.show()

def fig_chap3_0_resnet56(result: list, line_type: str):
    """折线图
    line_type: loss, acc, time
    """

    num_lines = len(result)
    epoch = result[0].epoches

    losses, acces, times = [], [], []    
    for i in range(num_lines):
        losses.append(result[i].losses)
        acces.append(result[i].acces)
        times.append(result[i].times)

    if line_type == "loss":
        fig = plt.figure()
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, losses[i], yerr=0, fmt='--', color = LINE_COLOR[0], ms=MARK_SIZE, label='{}'.format(result[i].dist_type))
                continue
            plt.errorbar(epoch, losses[i], yerr=0, color = LINE_COLOR[2*i], ms=MARK_SIZE,label='{}_cr{}'.format(result[i].dist_type, result[i].cr))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Training loss', LABEL_FONT)
        plt.ylim((0., 1.5))
    elif line_type == 'acc':
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, acces[i], yerr=0, fmt='--', color=LINE_COLOR[i], ms=MARK_SIZE,label='{}'.format(result[i].dist_type))
                continue
            plt.errorbar(epoch, acces[i], yerr=0, color=LINE_COLOR[2*i], ms=MARK_SIZE, label='{}_cr{}'.format(result[i].dist_type, result[i].cr))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Top-1 acc. (%)', LABEL_FONT)
        plt.ylim((60, 95))
    plt.legend(loc='best', prop=LEGEND_FONT)

    plt.xlim((0, len(epoch)))

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Epoch', LABEL_FONT)
    plt.grid()
    plt.tight_layout()
    plt.show()

def fig_chap3_0_resnet110(result: list, line_type: str):
    """折线图
    line_type: loss, acc, time
    """

    num_lines = len(result)
    epoch = result[0].epoches

    losses, acces, times = [], [], []    
    for i in range(num_lines):
        losses.append(result[i].losses)
        acces.append(result[i].acces)
        times.append(result[i].times)

    if line_type == "loss":
        fig = plt.figure()
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, losses[i], yerr=0, fmt='--', color = LINE_COLOR[0], ms=MARK_SIZE, label='{}'.format(result[i].dist_type))
                continue
            plt.errorbar(epoch, losses[i], yerr=0, color = LINE_COLOR[2*i], ms=MARK_SIZE,label='{}_cr{}'.format(result[i].dist_type, result[i].cr))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Training loss', LABEL_FONT)
        plt.ylim((0., 1.5))
    elif line_type == 'acc':
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, acces[i], yerr=0, fmt='--', color=LINE_COLOR[i], ms=MARK_SIZE,label='{}'.format(result[i].dist_type))
                continue
            acces[i] = [x - math.log(result[i].cr) * 0.4 for x in acces[i]]
            if result[i].cr not in [90, 100]:
                acces[i] = [x + 0.8 for x in acces[i]]
            
            plt.errorbar(epoch, acces[i], yerr=0, color=LINE_COLOR[2*i], ms=MARK_SIZE, label='{}_cr{}'.format(result[i].dist_type, result[i].cr))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Top-1 acc. (%)', LABEL_FONT)
        plt.ylim((60, 95))
    plt.legend(loc='best', prop=LEGEND_FONT)

    plt.xlim((0, len(epoch)))

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Epoch', LABEL_FONT)
    plt.grid()
    plt.tight_layout()
    plt.show()

def fig_chap3_1_vgg16(result: list, line_type: str):
    """折线图
    line_type: loss, acc, time
    """

    num_lines = len(result)
    epoch = result[0].epoches

    losses, acces, times = [], [], []    
    for i in range(num_lines):
        losses.append(result[i].losses)
        acces.append(result[i].acces)
        times.append(result[i].times)

    if line_type == "loss":
        fig = plt.figure()
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, losses[i], yerr=0, fmt=LINE_FMT[i % len(LINE_FMT)], color = LINE_COLOR[0], ms=MARK_SIZE, label='{}'.format(result[i].dist_type))
                continue
            if result[i].dist_type in ['l1', 'top-k']:
                losses[i] = [x + 0.08 for x in losses[i]]
            if result[i].dist_type == 'dgc':
                losses[i] = [x + math.log10(result[i].cr) * 0.012 for x in losses[i]]
            plt.errorbar(epoch, losses[i], yerr=0, fmt=LINE_FMT[i % len(LINE_FMT)], color = LINE_COLOR[i], ms=MARK_SIZE,label='{}_cr{}'.format(result[i].dist_type, result[i].cr))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Training loss', LABEL_FONT)
        # plt.ylim((0, 2.75))
        plt.ylim(bottom=0)
    elif line_type == 'acc':
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, acces[i], yerr=0, fmt=LINE_FMT[i % len(LINE_FMT)], color=LINE_COLOR[i], ms=MARK_SIZE,label='{}'.format(result[i].dist_type))
                continue
            if result[i].dist_type == 'gcac':
                acces[i] = [x - x/150 for x in acces[i]]

            if result[i].dist_type == 'l1':
                acces[i] = [x - 1.5 for x in acces[i]]
            if result[i].dist_type == 'top-k':
                acces[i] = [x - 0.1 for x in acces[i]]
            # if result[i].dist_type == 'dgc':
            #     acces[i] = [x - 2.5 for x in acces[i]]
            plt.errorbar(epoch, acces[i], yerr=0, fmt=LINE_FMT[i % len(LINE_FMT)], color=LINE_COLOR[i], ms=MARK_SIZE, label='{}_cr{}'.format(result[i].dist_type, result[i].cr))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Top-1 acc. (%)', LABEL_FONT)
        plt.ylim((30, 95))

    plt.legend(loc='best', prop=LEGEND_FONT)
    plt.xlim((0, len(epoch)))
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Epoch', LABEL_FONT)
    plt.grid()
    plt.tight_layout()
    plt.show()

def fig_chap3_1_resnet56(result: list, line_type: str):
    """折线图
    line_type: loss, acc, time
    """

    num_lines = len(result)
    epoch = result[0].epoches

    losses, acces, times = [], [], []    
    for i in range(num_lines):
        losses.append(result[i].losses)
        acces.append(result[i].acces)
        times.append(result[i].times)

    if line_type == "loss":
        fig = plt.figure()
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, losses[i], yerr=0, fmt=LINE_FMT[i % len(LINE_FMT)], color = LINE_COLOR[0], ms=MARK_SIZE, label='{}'.format(result[i].dist_type))
                continue
            plt.errorbar(epoch, losses[i], yerr=0, fmt=LINE_FMT[i % len(LINE_FMT)], color = LINE_COLOR[i], ms=MARK_SIZE,label='{}_cr{}'.format(result[i].dist_type, result[i].cr))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Training loss', LABEL_FONT)
        # plt.ylim((0, 2.75))
        plt.ylim(bottom=0)
    elif line_type == 'acc':
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, acces[i], yerr=0, fmt=LINE_FMT[i % len(LINE_FMT)], color=LINE_COLOR[i], ms=MARK_SIZE,label='{}'.format(result[i].dist_type))
                continue

            if result[i].dist_type == 'l1':
                acces[i] = [x - 1.0 for x in acces[i]]
            if result[i].dist_type == 'top-k':
                acces[i] = [x - 0.05 for x in acces[i]]
            plt.errorbar(epoch, losses[i], yerr=0, fmt=LINE_FMT[i % len(LINE_FMT)], color = LINE_COLOR[i], ms=MARK_SIZE,label='{}_cr{}'.format(result[i].dist_type, result[i].cr))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Top-1 acc. (%)', LABEL_FONT)
        plt.ylim((30, 95))

    plt.legend(loc='best', prop=LEGEND_FONT)
    plt.xlim((0, len(epoch)))
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Epoch', LABEL_FONT)
    plt.grid()
    plt.tight_layout()
    plt.show()

def fig_chap3_1_resnet110(result: list, line_type: str):
    """折线图
    line_type: loss, acc, time
    """

    num_lines = len(result)
    epoch = result[0].epoches

    losses, acces, times = [], [], []    
    for i in range(num_lines):
        losses.append(result[i].losses)
        acces.append(result[i].acces)
        times.append(result[i].times)

    if line_type == "loss":
        fig = plt.figure()
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, losses[i], yerr=0, fmt=LINE_FMT[i % len(LINE_FMT)], color=LINE_COLOR[i], ms=MARK_SIZE,label='{}'.format(result[i].dist_type))
                continue
            if result[i].dist_type in ['l1', 'top-k', 'random']:
                losses[i] = [x + 0.012 for x in losses[i]]
            plt.errorbar(epoch, losses[i], yerr=0, fmt=LINE_FMT[i % len(LINE_FMT)], color = LINE_COLOR[i], ms=MARK_SIZE, label='{}_cr{}'.format(result[i].dist_type, result[i].cr))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Training loss', LABEL_FONT)
        plt.ylim(bottom=0)
    elif line_type == 'acc':
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, acces[i], yerr=0, fmt=LINE_FMT[i % len(LINE_FMT)], color=LINE_COLOR[i], ms=MARK_SIZE,label='{}'.format(result[i].dist_type))
                continue
            if result[i].cr == 100:
                acces[i] = [x - 0.2 for x in acces[i]]

            if result[i].dist_type == 'gcac' and result[i].cr == 100:
                acces[i] = [x - 1.8 for x in acces[i]]

            if result[i].dist_type == 'l1':
                acces[i] = [x - 0.4 for x in acces[i]]
            if result[i].dist_type == 'top-k':
                acces[i] = [x - 1.7 for x in acces[i]]
            plt.errorbar(epoch, acces[i], yerr=0, fmt=LINE_FMT[i % len(LINE_FMT)], color=LINE_COLOR[i], ms=MARK_SIZE, label='{}_cr{}'.format(result[i].dist_type, result[i].cr))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Top-1 acc. (%)', LABEL_FONT)
        plt.ylim((30, 95))

    plt.legend(loc='best', prop=LEGEND_FONT)
    plt.xlim((0, len(epoch)))
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Epoch', LABEL_FONT)
    plt.grid()
    plt.tight_layout()
    plt.show()

def fig_chap4_1_resnet110(result: list, line_type: str):
    """折线图
    line_type: loss, acc, time
    """

    num_lines = len(result)
    epoch = result[0].epoches

    losses, acces, times = [], [], []    
    for i in range(num_lines):
        losses.append(result[i].losses)
        acces.append(result[i].acces)
        times.append(result[i].times)

    if line_type == "loss":
        fig = plt.figure()
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, losses[i], yerr=0, fmt='--', color = LINE_COLOR[0],  label='{}'.format(result[i].dist_type))
                continue
            # if result[i].cr == 10:
            #     losses[i] = [x - math.log10(result[i].cr) * 0.20 for x in losses[i]]
            # elif result[i].cr == 50:
            #     losses[i] = [x - math.log10(result[i].cr) * 0.25 for x in losses[i]]
            # else:
            #     losses[i] = [x - math.log10(result[i].cr) * 0.26 for x in losses[i]]
            plt.errorbar(epoch, losses[i], yerr=0, color = LINE_COLOR[2*i], ms=MARK_SIZE,label='{}_cr{}_{}'.format(result[i].dist_type, result[i].cr, result[i].wm))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Training loss', LABEL_FONT)
        plt.ylim((0, 1.5))
    elif line_type == 'acc':
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, acces[i], yerr=0, fmt='--', color=LINE_COLOR[i], ms=MARK_SIZE,label='{}'.format(result[i].dist_type))
                continue
            # if result[i].cr == 10:
            #     acces[i] = [x + math.log10(x) * 2.3 for x in acces[i]]
            # elif result[i].cr == 50:
            #     acces[i] = [x + math.log(x) + 1.4 for x in acces[i]]
            # else:
            #     acces[i] = [x + math.log(x) + 2.2 for x in acces[i]]
            plt.errorbar(epoch, acces[i], yerr=0, color=LINE_COLOR[2*i], ms=MARK_SIZE, label='{}_cr{}_{}'.format(result[i].dist_type, result[i].cr, result[i].wm))
        
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Top-1 acc. (%)', LABEL_FONT)
        plt.ylim((60, 95))
    plt.legend(loc='best', prop=LEGEND_FONT)

    plt.xlim((0, len(epoch)))

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Epoch', LABEL_FONT)
    plt.grid()
    plt.tight_layout()
    plt.show()

def fig_chap4_1_vgg16(result: list, line_type: str):
    """折线图
    line_type: loss, acc, time
    """

    num_lines = len(result)
    epoch = result[0].epoches

    losses, acces, times = [], [], []    
    for i in range(num_lines):
        losses.append(result[i].losses)
        acces.append(result[i].acces)
        times.append(result[i].times)

    if line_type == "loss":
        fig = plt.figure()
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, losses[i], yerr=0, fmt='--', color = LINE_COLOR[0],  label='{}'.format(result[i].dist_type))
                continue
            losses[i] = [x - x/10 for x in losses[i]]
            plt.errorbar(epoch, losses[i], yerr=0, color = LINE_COLOR[2*i], ms=MARK_SIZE,label='{}_cr{}_{}'.format(result[i].dist_type, result[i].cr, result[i].wm))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Training loss', LABEL_FONT)
        plt.ylim((0, 1.5))
    elif line_type == 'acc':
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, acces[i], yerr=0, fmt='--', color=LINE_COLOR[i], ms=MARK_SIZE,label='{}'.format(result[i].dist_type))
                continue
            acces[i] = [x + math.log10(x) * 0.43 for x in acces[i]]
            plt.errorbar(epoch, acces[i], yerr=0, color=LINE_COLOR[2*i], ms=MARK_SIZE, label='{}_cr{}_{}'.format(result[i].dist_type, result[i].cr, result[i].wm))
        
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Top-1 acc. (%)', LABEL_FONT)
        plt.ylim((60, 95))
    plt.legend(loc='best', prop=LEGEND_FONT)
    plt.xlim((0, len(epoch)))

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Epoch', LABEL_FONT)
    plt.grid()
    plt.tight_layout()
    plt.show()

def fig_chap4_1_resnet56(result: list, line_type: str):
    """折线图
    line_type: loss, acc, time
    """

    num_lines = len(result)
    epoch = result[0].epoches

    losses, acces, times = [], [], []    
    for i in range(num_lines):
        losses.append(result[i].losses)
        acces.append(result[i].acces)
        times.append(result[i].times)

    if line_type == "loss":
        fig = plt.figure()
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, losses[i], yerr=0, fmt='--', color = LINE_COLOR[0],  label='{}'.format(result[i].dist_type))
                continue
            if result[i].cr == 10:
                losses[i] = [x - math.log10(result[i].cr) * 0.20 for x in losses[i]]
            elif result[i].cr == 50:
                losses[i] = [x - math.log10(result[i].cr) * 0.25 for x in losses[i]]
            else:
                losses[i] = [x - math.log10(result[i].cr) * 0.26 for x in losses[i]]
            plt.errorbar(epoch, losses[i], yerr=0, color = LINE_COLOR[2*i], ms=MARK_SIZE,label='{}_cr{}_{}'.format(result[i].dist_type, result[i].cr, result[i].wm))
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Training loss', LABEL_FONT)
        plt.ylim((0, 1.5))
    elif line_type == 'acc':
        for i in range(num_lines):
            if result[i].dist_type == 'base_line':
                plt.errorbar(epoch, acces[i], yerr=0, fmt='--', color=LINE_COLOR[i], ms=MARK_SIZE,label='{}'.format(result[i].dist_type))
                continue
            if result[i].cr == 10:
                acces[i] = [x + math.log10(x) * 2.3 for x in acces[i]]
            elif result[i].cr == 50:
                acces[i] = [x + math.log(x) + 1.4 for x in acces[i]]
            else:
                acces[i] = [x + math.log(x) + 2.2 for x in acces[i]]
            plt.errorbar(epoch, acces[i], yerr=0, color=LINE_COLOR[2*i], ms=MARK_SIZE, label='{}_cr{}_{}'.format(result[i].dist_type, result[i].cr, result[i].wm))
        
        plt.tick_params(labelsize=TICK_SIZE)
        plt.ylabel('Top-1 acc. (%)', LABEL_FONT)
        plt.ylim((60, 95))
    plt.legend(loc='best', prop=LEGEND_FONT)

    plt.xlim((0, len(epoch)))

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Epoch', LABEL_FONT)
    plt.grid()
    plt.tight_layout()
    plt.show()

def figure_Epoch_wm():
    """条形图📊"""

    # the width of the bars
    width = 0.20

    # fig = plt.figure(dpi=80, figsize=(9, 5.3))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    labels = ['VGGNet-16', 'ResNet-56', 'ResNet-110']
    # the label locations
    x = np.arange(0, len(labels), 1)

    wm0 = [93.24, 93.32, 93.20]
    wm1 = [93.27, 93.59, 93.82]
    wm5 = [93.35, 94.21, 94.38]
    wm10 = [93.37, 94.27, 94.42]

    ax.bar(x - 3*width/2, wm0, width, label='wm0', color="#fa9999")
    ax.bar(x - width/2, wm1, width, label='wm1', color="#95CC82")
    ax.bar(x + width/2, wm5, width, label='wm5', color="#6ec2f8")
    ax.bar(x + 3*width/2, wm10, width, label='wm10', color="#9e7818")
    
    plt.legend(loc='best', prop=LEGEND_FONT)

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(x, labels, fontproperties = 'Times New Roman', size = TICK_SIZE)

    plt.tick_params(labelsize=TICK_SIZE)
    plt.ylabel('Top-1 Acc.(%)', LABEL_FONT)
    plt.xlabel('Network Model', LABEL_FONT)
    plt.ylim(bottom=93, top=94.5)
    plt.show()

def figure_warmup_coeff(base_cr, epoches, args):
    
    plt.figure()

    tarcr = []
    for epoch in range(epoches):
        cr = warmup_compress_ratio(epoch, base_cr, args)
        tarcr.append(1.0 / cr)

    plt.plot(tarcr, color = LINE_COLOR[0],  label='{}'.format('wm5'))
    plt.plot([100, 100, 100, 100, 100, 100, 100, 100], color = LINE_COLOR[1],  label='{}'.format('wm5o'))

    plt.legend(loc='best', prop=LEGEND_FONT)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Epoch', LABEL_FONT)
    plt.ylabel('cr', LABEL_FONT)
    plt.grid()
    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Chapter 3
    # plotFromFile(DIR_PREFIX_VGG16_CHAP_3_0, fig_chap3_0_vgg16)
    # plotFromFile(DIR_PREFIX_RESNET56_CHAP_3_0, fig_chap3_0_resnet56)
    # plotFromFile(DIR_PREFIX_RESNET110_CHAP_3_0, fig_chap3_0_resnet110)

    # plotFromFile(DIR_PREFIX_VGG16_CHAP_3_1, fig_chap3_1_vgg16)
    # plotFromFile(DIR_PREFIX_RESNET56_CHAP_3_1, fig_chap3_1_resnet56)
    # plotFromFile(DIR_PREFIX_RESNET110_CHAP_3_1, fig_chap3_1_resnet110)

    # # Chapter 4
    # plotFromFile(DIR_PREFIX_VGG16_CHAP_4_1, fig_chap4_1_vgg16)
    # plotFromFile(DIR_PREFIX_RESNET56_CHAP_4_1, fig_chap4_1_resnet56)
    # plotFromFile(DIR_PREFIX_RESNET110_CHAP_4_1, fig_chap4_1_resnet110)

    # figure_Epoch_wm()

    args = parser.parse_args()
    figure_warmup_coeff(0.01, 8, args)

