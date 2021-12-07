import torch
import torch.nn as nn

__all__ = ['get_device', 'to_device']


def get_device(data):
    '''
    Get the device of data.

    Type `T`: torch.Tensor|torch.nn.Module

    Args:
        data(T): Data which you want to know its device.
    Returns:
        (torch.device): The device which data is in.
    '''
    if isinstance(data, torch.Tensor):
        return data.device
    elif isinstance(data, nn.Module):
        if isinstance(data, nn.DataParallel):
            para_dict = data.module.state_dict()
        else:
            para_dict = data.state_dict()

        keys = [para_dict.keys()]
        return para_dict[keys[0]].device
    else:
        msg = 'data should be torch.Tensor or torch.nn.Module, but got {}'.format(
            type(data))
        raise TypeError(msg)


def to_device(data, device):
    '''
    Move data to specific device.

    Type `T`: Union[torch.Tensor, torch.nn.Moudle]

    Args:
        data(Union[Sequence[T], Dict[str, T], T]): The data which will be moved to device. It can be dict or sequence.
        device(torch.device): The target device of data.
    Returns:
        (Union[Sequence[T], Dict[str, T], T]): Structure is the same as data. but it is moved to specific device.
    '''
    if isinstance(data, (list, tuple)):
        l = []

        for d in data:
            if isinstance(d, (torch.Tensor, torch.nn.Module)):
                l.append(d.to(device))
            else:
                l.append(d)

        return l
    elif isinstance(data, dict):
        d = {}

        for key, value in data.items():
            if isinstance(value, (torch.Tensor, torch.nn.Module)):
                d[key] = value.to(device)
            else:
                d[key] = value

        return d
    elif isinstance(data, (torch.Tensor, torch.nn.Module)):
        return data.to(device)
    else:
        return data
