import os, sys, torch, torchvision, time

from importlib import import_module

from data import cifar10
from utils import args, print_info
from utils.common import AverageMeter, Checkpoint, accuracy
from utils.compress import sparsify

def warmup_compress_ratio(epoch:int, base_cr):
    if args.warmup_epochs > 0:
        if epoch < args.warmup_epochs:
            if isinstance(args.warmup_coeff, (tuple, list)):
                cr = args.warmup_coeff[epoch]
            else:
                cr = max(args.warmup_coeff ** (epoch + 1), base_cr)
        else:
            cr = base_cr
    else:
        cr = base_cr
    if cr != args.cr:
        print_info('Warmup epoch: {}, compress_ratio: {}'.format(epoch, cr))
        args.cr = cr

def main():
    # version information
    print_info("\r\nPython  version : {}\n".format(sys.version.replace('\n', ' '))
                + "PyTorch version : {}\n".format(torch.__version__)
                + "cuDNN version : {}\n".format(torch.backends.cudnn.version())
                + "TorchVision version : {}\n".format(torchvision.__version__))

    device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

    # Data
    print(">>> Preparing data...")
    args.train_batch_size *= args.num_batches_per_step
    data_loader = cifar10.Data(args)

    # Load model
    model = None
    print(">>> Loading model...")
    if args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
    elif args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet().to(device)
    elif args.arch == 'mobilenetv2_cifar':
        model = import_module(f'model.{args.arch}').mobilenet_v2().to(device)
    else:
        raise('arch not exist!') # type: ignore

    print_info(model)

    # criterion
    print(">>> Criterion...")
    criterion = torch.nn.CrossEntropyLoss()

    # Eval
    if args.eval:
        if not args.pretrain_model or not os.path.exists(args.pretrain_model):
            raise('pretrained model must be exist when eval mode!!!') # type: ignore
        
        print(">>> Loading pretrained param...")
        ckpt = torch.load(args.pretrain_model, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        test(model, data_loader.test_loader, criterion, device=device)
        return

    # Load pretrained param
    if args.pretrain_model and os.path.exists(args.pretrain_model): 
        print(">>> Loading pretrained param...")
        ckpt = torch.load(args.pretrain_model, map_location=device)
        model.load_state_dict(ckpt['state_dict'])

    if len(args.gpus) != 1:
        print(">>> Data parallel...")
        model = torch.nn.DataParallel(model, device_ids=args.gpus)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                                weight_decay=args.weight_decay)
    # scheduler
    if args.lr_type == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_decay_step, gamma=0.1)
    elif args.lr_type == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)
    else:
        raise('lr_type not exist!') # type: ignore
    
    train(data_loader, model, optimizer, scheduler, criterion, device)

def train(data_loader, model, optimizer, scheduler, criterion, device):
    if args.warmup_epochs > 0:
        base_cr = args.cr = args.cr if args.cr <= 1.0 else 1.0 / args.cr
        if args.warmup_coeff is None:
            args.warmup_coeff = base_cr ** (1. / (args.warmup_epochs + 1))
        else:
            if isinstance(args.warmup_coeff, (tuple, list)):
                assert len(args.warmup_coeff) >= args.warmup_epochs
                for wc in args.warmup_coeff:
                    assert 0 < wc <= 1
            else:
                assert 0 < args.warmup_coeff <= 1
    else:
        args.warmup_coeff = 1

    # 
    checkpoint = Checkpoint(args)
    start_epoch = 0
    best_acc = 0.0

    # 初始化 v：用于梯度累积
    v = {} 
    for idx, (name, item) in enumerate(model.named_parameters()):
        v[name] = torch.zeros_like(item)

    for epoch in range(start_epoch, args.num_epochs):
        warmup_compress_ratio(epoch, base_cr)

        train_epoch(model, optimizer, criterion, data_loader.train_loader, args, epoch, v, topk=(1, 5) if args.dataset == 'imagenet' else (1, ), device=device)
        scheduler.step()
        test_acc = test(model, data_loader.test_loader, criterion, topk=(1, 5) if args.dataset == 'imagenet' else (1, ), device=device)

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'arch': args.cfg,
            # 'cfg': cfg
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    print_info('Best accuracy: {:.3f}'.format(float(best_acc)))

def train_epoch(model, optimizer, criterion, train_loader, args, epoch, v, topk, device):
    model.train()
    losses = AverageMeter('Time', ':6.3f')
    accurary = AverageMeter('Time', ':6.3f')
    top5_accuracy = AverageMeter('Time', ':6.3f')

    batch_size = int(args.train_batch_size / args.num_batches_per_step)
    step_size = args.num_batches_per_step * batch_size
    _r_num_batches_per_step = 1.0 / args.num_batches_per_step

    print_freq = len(train_loader.dataset) // batch_size // 10
    start_time = time.time()

    for batch, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        model = model.to(device)
        output = model(inputs[0: batch_size])
        loss = criterion(output, targets[0: batch_size])
        loss.mul_(_r_num_batches_per_step)
        loss.backward()
        for b in range(0, step_size, batch_size):
            _inputs = inputs[b:b+batch_size]
            _targets = targets[b:b+batch_size]
            if _inputs.size(0) <= 0:
                break
            _outputs = model(_inputs)
            _loss = criterion(_outputs, _targets)
            _loss.mul_(_r_num_batches_per_step)
            _loss.backward()
            loss += _loss.item()
        
        if (args.warmup_epochs <= 0) or \
            (args.warmup_epochs > 0 and epoch >= args.warmup_epochs):
            sparsify(model, args.cr, v, args.dist_type)
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

        output = model(inputs)
        prec1 = accuracy(output, targets, topk=topk)
        accurary.update(prec1[0], inputs.size(0))
        if len(topk) == 2:
            top5_accuracy.update(prec1[1], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            if len(topk) == 1:
                print_info('Epoch[{}] ({}/{}):\tLoss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(train_loader.dataset),
                        float(losses.avg), float(accurary.avg), cost_time))
            else:
                print_info(
                    'Epoch[{}] ({}/{}):\tLoss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(train_loader.dataset),
                        float(losses.avg), float(accurary.avg), float(top5_accuracy.avg), cost_time))
            start_time = current_time

def test(model, test_loader, criterion, topk=(1,), device='cpu'):
    model.eval()

    losses = AverageMeter('Time', ':6.3f')
    accurary = AverageMeter('Time', ':6.3f')
    top5_accuracy = AverageMeter('Time', ':6.3f')

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = accuracy(outputs, targets, topk=topk)
            accurary.update(predicted[0], inputs.size(0))
            if len(topk) == 2:
                top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        if len(topk) == 1:
            print_info(
                'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
            )
        else:
            print_info(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(accurary.avg), float(top5_accuracy.avg), (current_time - start_time))
            )
    if len(topk) == 1:
        return accurary.avg
    else:
        return top5_accuracy.avg


if __name__ == '__main__':
    main()
