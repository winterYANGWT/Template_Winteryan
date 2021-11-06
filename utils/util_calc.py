import torch

__all__ = ['mask_sum', 'mask_mean']


def mask_sum(data, mask, dim=None, keepdim=False):
    masked_data = data * mask

    if dim != None:
        return torch.sum(masked_data, dim, keepdim=keepdim)
    else:
        return torch.sum(masked_data)


def mask_mean(data, mask, dim=None, keepdim=False):
    if dim != None:
        summed_data = mask_sum(data, mask, dim=dim, keepdim=keepdim)
        summed_mask = torch.sum(mask, dim=dim, keepdim=keepdim)
        return summed_data / summed_mask
    else:
        masked_data = data[mask]
        return torch.mean(masked_data)
