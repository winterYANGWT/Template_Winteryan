import os
import os.path as path
import torch

__all__ = ['load_model', 'save_model']


def load_model(model, load_dir):
    '''
    Load weight from file which is on load_path to model.

    Args:
        model(torch.nn.Module)
        load_dir(str): the directory of model weight file.
    Returns:
        (torch.nn.Module): A model loaded with weight.
    '''
    if not path.exists(load_dir):
        msg = f'this directory \'{load_dir}\' doesn\'t exist.'
        raise FileExistsError(msg)

    load_path = path.join(load_dir, model.name + '.pth')

    if isinstance(model, torch.nn.Module):
        model.load_state_dict(torch.load(load_path, map_location='cpu'))
    else:
        msg = f'model should be torch.nn.Module, but got \'{type(model)}\'.'
        raise TypeError(msg)

    return model


def save_model(model, save_dir):
    '''
    Save model's weight in save_dir.

    Args:
        model(torch.nn.Module)
        save_dir(str): The directory which model's weight file saved.
    '''
    os.makedirs(save_dir, exist_ok=True)
    save_path = path.join(save_dir, model.name + '.pth')

    if isinstance(model, torch.nn.Module):
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
    else:
        msg = f'model should be torch.nn.Module, but got \'{type(model)}\'.'
        raise TypeError(msg)
