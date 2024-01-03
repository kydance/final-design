import os, sys, torch, torchvision, random, time
import numpy as np

from utils import args, print_info
from data import htd
from model.AAE import AE, GAN
from utils.common import Checkpoint
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
    module_ae = AE(dim_data=args.bands, dim_z=args.hidden_dim).to(device)
    module_gan = GAN(dim_data=args.bands, dim_z=args.hidden_dim).to(device)

    print_info(module_ae)
    print_info(module_gan)

    # optimizer
    optimizer_ae = torch.optim.Adam(module_ae.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(module_gan.parameters(), lr=args.lr)
    optimizer_g = torch.optim.Adam(module_gan.parameters(), lr=args.lr)

    train(data_loader, module_ae, module_gan, optimizer_ae, optimizer_d, optimizer_g, device)

def train(data_loader: htd.Data, module_ae, module_gan, optimizer_ae, optimizer_d, optimizer_g, device):
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

    # XXX 不保存模型，只记录了训练模型时的超参数
    _ = Checkpoint(args)

    start_epoch = 0
    vs = init_vs(module_ae, module_gan)

    for epoch in range(start_epoch, args.num_epochs):
        warmup_compress_ratio(epoch, base_cr)
        train_epoch(module_ae, module_gan, optimizer_ae, optimizer_d, optimizer_g, 
                    data_loader, args, epoch, vs, device=device)

    # XXX 假设最后一次训练效果最好，待优化
    test(module_ae, data_loader, device)

def train_epoch(module_ae, module_gan, optimizer_ae, optimizer_d, optimizer_g, 
                data_loader: htd.Data, args, epoch, vs:list, device):
        start_time = time.time()

        # It must be on CPU, Or CUDA-Memeory leak 
        loss_per_epoch_r = 0
        loss_per_epoch_d = 0
        loss_per_epoch_g = 0

        d = torch.tensor(data_loader.dataset.d, dtype=torch.float32).to(device)

        for index, batch_data in enumerate(data_loader.train_loader):
            x = batch_data.to(device)
            y, _, R_loss = module_ae(x, d)
            R_loss.backward(retain_graph=True)
            loss_per_epoch_r = loss_per_epoch_r + R_loss.detach().cpu().numpy()
        sparsify(module_ae, args.cr, vs[0], args.dist_type)
        optimizer_ae.step()
        optimizer_ae.zero_grad()

        for index, batch_data in enumerate(data_loader.train_loader):
            x = batch_data.to(device)
            z = module_ae(x, d, with_decoder=False)
            D_loss = module_gan(z, with_G=False)
            D_loss.backward(retain_graph=True)
            loss_per_epoch_d = loss_per_epoch_d + D_loss.detach().cpu().numpy()
        sparsify(module_gan, args.cr, vs[1], args.dist_type)

        optimizer_d.step()
        optimizer_d.zero_grad()

        for index, batch_data in enumerate(data_loader.train_loader):
            x = batch_data.to(device)
            z = module_ae(x, d, with_decoder=False)
            _, G_loss = module_gan(z)
            G_loss.backward()
            loss_per_epoch_g = loss_per_epoch_g + G_loss.detach().cpu().numpy()
            
            for name, param in module_gan.named_parameters():
                if param.grad is None:
                    break
        sparsify(module_gan, args.cr, vs[2], args.dist_type)

        optimizer_g.step()
        optimizer_g.zero_grad()

        loss_per_epoch_r = loss_per_epoch_r / (index+1)
        loss_per_epoch_d = loss_per_epoch_d / (index+1)
        loss_per_epoch_g = loss_per_epoch_g / (index+1)

        current_time = time.time()
        print_info('Epoch[{}/{}]:\tR Loss {:.4f}\tD Loss {:.4f}\tG Loss {:.4f}\tTime {:.2f}s'.format(
                    epoch, args.num_epochs, 
                    float(loss_per_epoch_r), float(loss_per_epoch_d), float(loss_per_epoch_g), 
                    current_time - start_time))
        start_time = current_time

def test(module_ae, data_loader: htd.Data, device):
    module_ae.eval()
    with torch.no_grad():
        d = torch.tensor(data_loader.dataset.d, dtype=torch.float32).to(device)

        for index, batch_data in enumerate(data_loader.test_loader):
            x = batch_data.to(device)
            y, _, _ = module_ae(x, d)
            y = y.cpu().numpy()
            if index == 0:
                y_pred = y
            else:
                y_pred = np.concatenate((y_pred, y), axis=0)
        y_pred = np.reshape(y_pred, [data_loader.dataset.size[0], data_loader.dataset.size[1], data_loader.dataset.bands])
        save_result(y_pred)

def init_vs(module_ae, module_gan) -> list:
    """初始化 vs - 用于梯度累积"""
    v_ae = {} 
    v_d = {} 
    v_g = {} 
    for _, (name, item) in enumerate(module_ae.named_parameters()):
        v_ae[name] = torch.zeros_like(item)
    
    for _, (name, item) in enumerate(module_gan.named_parameters()):
        v_d[name] = torch.zeros_like(item)
        v_g[name] = torch.zeros_like(item)
    return list([v_ae, v_d, v_g])

def save_result(y_pred: np.ndarray):
    # Save result 
    _path = os.path.join(args.job_dir, args.dataset + "_result.mat")
    if os.path.exists(_path):
        open(_path, 'w').close()
    print_info('Saving result to {} ... '.format(_path))
    savemat(_path, {'y': y_pred}, format='7.3')
    print_info('Save done.')

if __name__ == '__main__':
    main()
