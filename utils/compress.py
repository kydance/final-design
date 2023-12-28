import copy, torch, math
import numpy as np
from scipy.spatial import distance

# TODO reconstruction functions with class

def sparsify(model, compress_cr, v, dist_type: str="abs"):
    for name, param in model.named_parameters():
        grad = param.grad.data
        if torch.is_tensor(grad):
            grad_norm = _importance(grad, dist_type)
            num_reserved = int(math.ceil(grad_norm.numel() * compress_cr))
            threshold = torch.min(torch.topk(grad_norm, num_reserved, 0, largest=True, sorted=False)[0])
            mask = torch.ge(grad_norm, threshold)
            indices = torch.nonzero(mask).view(-1)[:num_reserved]

            param.grad.data = _update(grad, indices, name, v)
        else:
            raise Exception("grad must be tensor!")

def _importance(grad: torch.Tensor, dist_type: str="abs"):
    if dist_type == "gcc":
        grad_vec = grad.view(grad.numel(), -1)
        # FIXME
        grad_norm = torch.norm(grad_vec, 2, 1, keepdim=True) # type: ignore

        # On CPU
        # grad_norm_np = grad_norm.cpu().numpy()
        # similar_matrix = distance.cdist(grad_norm_np, grad_norm_np, 'euclidean')
        # similar_sum = np.sum(np.abs(similar_matrix), axis=0)
        # return torch.from_numpy(similar_sum).to(grad.device)

        # On GPU
        similar_matrix = torch.cdist(grad_norm, grad_norm)
        # FIXME
        similar_sum = torch.sum(torch.abs(similar_matrix), axis=0) # type: ignore
        return similar_sum
    elif dist_type == "l1":
        grad_vec = grad.view(-1, 1)
        # FIXME
        grad_norm = torch.norm(grad_vec, 1, 1, keepdim=True) # type: ignore

        # # On CPU
        # grad_norm_np = grad_norm.cpu().numpy()
        # similar_matrix = 1 - distance.cdist(grad_norm_np, grad_norm_np, 'cosine')
        # similar_sum = np.sum(np.abs(similar_matrix), axis=0)
        # return torch.from_numpy(similar_sum).to(grad.device)

        # On GPU
        similar_matrix = torch.cdist(grad_norm, grad_norm)
        # FIXME
        similar_sum = torch.sum(torch.abs(similar_matrix), axis=0) # type: ignore
        return similar_sum
    elif dist_type == "abs":
        grad_vec = grad.view(-1)
        return torch.abs(grad_vec)
    else:
        raise Exception("dist_type must be in [l2, cos, l1, abs]")

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
