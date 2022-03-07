import os
import torch

__all__ = ['load_model', 'save_model']


def load_model(model, load_path):
    '''
    Load weight from file which is on load_path to model.

    Args:
        model(torch.nn.Module)
        load_path(str): the path os model weight file.
    Returns:
        (torch.nn.Module): A model loaded with weight.
    '''
    if not os.path.exists(load_path):
        msg = 'this file {} doesn\'t exist'.format(load_path)
        raise FileExistsError(msg)

    model.load_state_dict(torch.load(load_path, map_location='cpu'))
    return model


def save_model(model, save_dir):
    '''
    Save model's weight in save_dir.

    Args:
        model(torch.nn.Module)
        save_dir(str): The directory which model's weight file saved.
    '''
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(model, torch.nn.Module):
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(),
                       os.path.join(save_dir, model.module.name + '.pth'))
        else:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, model.name + '.pth'))
    else:
        msg = 'model should be torch.nn.Module, but got {}'.format(type(model))
        raise TypeError(msg)
