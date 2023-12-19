import time, copy
import torch, math

# TODO reconstruction functions with class

def importance_abs(tensor):
    return tensor.abs()

def _update(grad, indices, name, v):
    '''Update grad with gradient accumulate.
        return: the gradient of updated.
    '''

    v_vec = v[name].view(-1)
    v_vec += grad
    A = torch.zeros_like(v_vec).to(grad.device)

    A = copy.deepcopy(v_vec)
    A.index_fill_(0, indices, 0)
    A = -A
    A += v_vec
    v_vec.index_fill_(0, indices, 0)

    v[name] = v_vec.view(v[name].shape)
    return A.view(v[name].shape)

def sparsify(model, compress_cr, v, importance_fn=importance_abs):
    for name, param in model.named_parameters():
        grad = param.grad.data.view(-1)
        if torch.is_tensor(grad):
            grad_norm = importance_fn(grad)
            numReserved = int(math.ceil(grad_norm.numel() * compress_cr))
            threshold = torch.min(torch.topk(grad_norm, numReserved, 0, largest=True, sorted=False)[0])
            mask = torch.ge(grad_norm, threshold)

            # indices = mask.nonzero().view(-1)
            # indices = indices[:numReserved]
            # indices = mask.nonzero().view(-1)[:numReserved]
            indices = torch.nonzero(mask).view(-1)[:numReserved]

            param.grad.data = _update(grad, indices, name, v)
        else:
            raise "grad must be tensor!" # type: ignore

def sparsify_local(model, compress_cr, v):
    for name, param in model.named_parameters():
        grad = param.grad.data.view(-1)
        if torch.is_tensor(grad):
            # conv_weight = torch.div(module.weight.data, math.pow(numFLOPs, lam))
            # grad_norm = torch.abs(grad)
            grad_norm = grad.abs()
            numReserved = int(math.ceil(grad_norm.numel() * compress_cr))
            # threshold = torch.min(torch.topk(grad_norm.view(-1), numReserved, 0, largest=True, sorted=False)[0])
            threshold = torch.min(torch.topk(grad_norm, numReserved, 0, largest=True, sorted=False)[0])
            mask = torch.ge(grad_norm, threshold)
            indices = mask.nonzero().view(-1)
            num_indices = indices.numel()
            indices = indices[:numReserved]
            values = grad[indices]
            param.grad.data = _update(grad, indices, name, v)
        else:
            raise "grad must be tensor!" # type: ignore
