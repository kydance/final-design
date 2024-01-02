import os
import sys, torch, torchvision, random
import numpy as np

from utils import args, print_info
from data import htd
# from model.aae_htd import AE, GAN
from model.AAE import AE, GAN
from utils.common import AverageMeter, Checkpoint, accuracy
from utils.compress import sparsify
from hdf5storage import savemat

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

def main() :
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # version information
    print_info("\r\nPython  version : {}\n".format(sys.version.replace('\n', ' '))
                + "PyTorch version : {}\n".format(torch.__version__)
                + "cuDNN version : {}\n".format(torch.backends.cudnn.version())
                + "TorchVision version : {}\n".format(torchvision.__version__))

    device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

    # Data
    print(">>> Preparing data...")
    args.train_batch_size *= args.num_batches_per_step
    data_loader = htd.Data(args.train_batch_size, shuffle=False,
                           data_path=args.data_path, data_name=args.dataset)
    args.bands = data_loader.dataset.bands
    args.hidden_dim = 50

    # Load model
    print(">>> Loading model...")
    model_ae = AE(dim_data=args.bands, dim_z=args.hidden_dim).to(device)
    model_gan = GAN(dim_data=args.bands, dim_z=args.hidden_dim).to(device)

    # optimizer
    optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(model_gan.parameters(), lr=args.lr)
    optimizer_g = torch.optim.Adam(model_gan.parameters(), lr=args.lr)

    train(data_loader, model_ae, model_gan, optimizer_ae, optimizer_d, optimizer_g, device)

def train(data_loader, model_ae, model_gan, optimizer_ae, optimizer_d, optimizer_g, device):
    base_cr = args.cr = args.cr if args.cr <= 1.0 else 1.0 / args.cr
    if args.warmup_epochs > 0:
        if len(args.warmup_coeff) <= 0:
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
    # checkpoint = Checkpoint(args)
    start_epoch = 0
    best_acc = 0.0

    vs = init_V(model_ae, model_gan)

    for epoch in range(start_epoch, args.num_epochs):
        warmup_compress_ratio(epoch, base_cr)

        train_epoch(model_ae, model_gan, optimizer_ae, optimizer_d, optimizer_g, 
                    data_loader, args, epoch, vs, device=device)
        test(model_ae, data_loader)

    print_info('Best accuracy: {:.3f}'.format(float(best_acc)))


def train_epoch(model_ae, model_gan, optimizer_ae, optimizer_d, optimizer_g, 
                data_loader: htd.Data, args, epoch, vs:list, device):
    model_ae.train()

    batch_size = int(args.train_batch_size / args.num_batches_per_step)
    step_size = args.num_batches_per_step * batch_size
    _r_num_batches_per_step = 1.0 / args.num_batches_per_step

    loss_per_epoch_d = 0
    loss_per_epoch_g = 0

    d = torch.tensor(data_loader.dataset.d, dtype=torch.float32).to(device)

    # Train ae
    loss_per_epoch_r = torch.tensor(0.0, dtype=torch.float32).to(device)
    for _, batch_data in enumerate(data_loader.train_loader):
        x = batch_data.to(device)
        x = x.float()
        y, _, R_loss = model_ae(x, d)
        R_loss.backward(retain_graph=True)
        loss_per_epoch_r += R_loss
        # loss_per_epoch_r = loss_per_epoch_r + R_loss.detach().cpu().numpy()
    if (args.warmup_epochs <= 0) or (args.warmup_epochs > 0 and epoch >= args.warmup_epochs):
        sparsify(model_ae, args.cr, vs[0], args.dist_type)
    optimizer_ae.step()
    optimizer_ae.zero_grad()

    # Train d
    for _, batch_data in enumerate(data_loader.train_loader):
        x = batch_data.cuda()
        x = x.float()
        z = model_ae(x, d, with_decoder=False)
        D_loss = model_gan(z, with_G=False)
        D_loss.backward(retain_graph=True)
        loss_per_epoch_d = loss_per_epoch_d + D_loss.detach().cpu().numpy()
    if (args.warmup_epochs <= 0) or (args.warmup_epochs > 0 and epoch >= args.warmup_epochs):
        sparsify(model_gan, args.cr, vs[1], args.dist_type)
    optimizer_d.step()
    optimizer_d.zero_grad()

    # Train g
    for _, batch_data in enumerate(data_loader.train_loader):
        x = batch_data.cuda()
        x = x.float()
        z = model_ae(x, d, with_decoder=False)
        _, G_loss = model_gan(z)
        G_loss.backward()
        loss_per_epoch_g = loss_per_epoch_g + G_loss.detach().cpu().numpy()
    if (args.warmup_epochs <= 0) or (args.warmup_epochs > 0 and epoch >= args.warmup_epochs):
        sparsify(model_gan, args.cr, vs[2], args.dist_type)
    optimizer_g.step()
    optimizer_g.zero_grad()

def test(model_ae, data_loader:htd.Data):
    model_ae.eval()
    with torch.no_grad():
        for index, batch_data in enumerate(data_loader.test_loader):
            x = batch_data.cuda()
            x = x.float()
            d = torch.tensor(data_loader.dataset.d, dtype=torch.float32).cuda()
            d = d.float()
            y, _, _ = model_ae(x, d)
            y = y.cpu().numpy()
            if index == 0:
                y_pred = y
            else:
                y_pred = np.concatenate((y_pred, y), axis=0)
        y_pred = np.reshape(y_pred, [data_loader.dataset.size[0], 
                                     data_loader.dataset.size[1], data_loader.dataset.bands])
        _path = os.path.join(args.job_dir, args.dataset + ".mat")
        if os.path.exists(_path):
            open(_path, 'w').close()

        savemat(_path, {'y': y_pred}, format='7.3')

def init_V(model_ae, model_gan) -> list:
    """初始化 vs - 用于梯度累积"""
    v_ae = {} 
    v_d = {} 
    v_g = {} 
    for idx, (name, item) in enumerate(model_ae.named_parameters()):
        v_ae[name] = torch.zeros_like(item)
    
    for idx, (name, item) in enumerate(model_gan.named_parameters()):
        v_d[name] = torch.zeros_like(item)
        v_g[name] = torch.zeros_like(item)
    return list([v_ae, v_d, v_g])

if __name__ == '__main__':
    main()
