import copy, torch, math
import numpy as np
from scipy.spatial import distance

# TODO reconstruction functions with class

def importance(grad, dist_type: str="l2"):
    grad_vec = grad.view(-1, 1)
    if dist_type == "l2" or "cos":
        grad_norm = torch.norm(grad_vec, 2, 1, keepdim=True)

        # On CPU
        grad_norm_np = grad_norm.cpu().numpy()
        similar_matrix = distance.cdist(grad_norm_np, grad_norm_np, 'euclidean')

        # On GPU
        # similar_matrix = torch.cdist(grad_norm, grad_norm)

        similar_sum = torch.sum(torch.abs(similar_matrix), axis=0)
        return similar_sum
    elif dist_type == "l1":
        grad_norm = torch.norm(grad_vec, 1, 1, keepdim=True)

        # On CPU
        grad_norm_np = grad_norm.cpu().numpy()
        similar_matrix = 1 - distance.cdist(grad_norm_np, grad_norm_np, 'cosine')

        # On GPU
        # similar_matrix = torch.cdist(grad_norm, grad_norm)

        similar_sum = np.sum(np.abs(similar_matrix), axis=0)
        return torch.from_numpy(similar_sum).cuda()
    elif dist_type == "abs":
        return torch.abs(grad_vec)
    else:
        raise "dist_type must be in [l2, cos, l1, abs]" # type: ignore

def _update(grad, indices, name, v):
    '''Update grad with gradient accumulate.
        return: the gradient of updated.
    '''

    v_vec = v[name].view(-1)
    v_vec += grad.view(-1)
    A = torch.zeros_like(v_vec).to(grad.device)

    A = copy.deepcopy(v_vec)
    A.index_fill_(0, indices, 0)
    A = -A
    A += v_vec
    v_vec.index_fill_(0, indices, 0)

    v[name] = v_vec.view(v[name].shape)
    return A.view(v[name].shape)

def sparsify(model, compress_cr, v, dist_type: str="abs"):
    for name, param in model.named_parameters():
        grad = param.grad.data
        if torch.is_tensor(grad):
            grad_norm = importance(grad, dist_type)
            num_reserved = int(math.ceil(grad_norm.numel() * compress_cr))
            threshold = torch.min(torch.topk(grad_norm, num_reserved, 0, largest=True, sorted=False)[0])
            mask = torch.ge(grad_norm, threshold)
            indices = torch.nonzero(mask).view(-1)[:num_reserved]

            param.grad.data = _update(grad, indices, name, v)
        else:
            raise "grad must be tensor!" # type: ignore

'''
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

def sparsify_gcc(model, cr, v, dist_type="l2"):
    for name, param in model.named_parameters():
        grad = param.grad.data
        if torch.is_tensor(grad):
            grad_norm = importance(grad, dist_type)

            num_reserved = int(math.ceil(grad_norm.numel() * cr))
            threshold = torch.min(torch.topk(grad_norm, num_reserved, 0, largest=True, sorted=False)[0])
            mask = torch.ge(grad_norm, threshold)
            indices = torch.nonzero(mask).view(-1)[:num_reserved].cuda()

            param.grad.data = _update(grad, indices, name, v)
        else:
            raise "grad must be tensor!" # type: ignore
'''