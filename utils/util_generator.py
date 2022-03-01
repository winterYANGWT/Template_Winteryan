import os.path as path
import torch
import torch.utils.data as data
import torch.nn as nn
from .util_io import load_model
from .util_device import to_device
from typing import Dict

__all__ = [
    'generate_dataset', 'generate_data_loader', 'generate_data_loaders',
    'generate_models', 'generate_optimizer'
]


def generate_dataset(dataset,
                     shuffle=False,
                     batch_size=1,
                     num_workers=1,
                     prefetch_factor=2,
                     pin_memory=False,
                     collate_fn=None):
    '''
    Generate a dataset that has some necessary attributes.

    Args:
        dataset(Union[dataset.AbstractDataset, dataset.Dataset, data.ConcatDataset]): A dataset can load data.
        shuffle(bool): To indicate whether to shuffle the order of data.
        batch_size(int): To indicate how many samples a batch contain.
        num_workers(int): To indicate how many subprocesses to use for loading data.
        prefetch_factor(int): To indicate that prefetch_factor * batch_size samples will be prefetched.
        collate_fn(Callable): A function processes samples from a batch.

    Returns:
        (dataset.AbstractDataset): a dataset contains some necessary attributes.
    '''
    if not isinstance(shuffle, bool):
        msg = 'shuffle should be Type bool, but got {}.'.format(type(shuffle))
        raise TypeError(msg)

    if not isinstance(batch_size, int):
        msg = 'batch_size should be Type int, but got {}.'.format(
            type(batch_size))
        raise TypeError(msg)

    if not isinstance(num_workers, int):
        msg = 'num_workers should be Type int, but got {}.'.format(
            type(num_workers))
        raise TypeError(msg)

    if not isinstance(prefetch_factor, int):
        msg = 'prefetch_factor should be Type int, but got {}.'.format(
            type(prefetch_factor))
        raise TypeError(msg)

    if not hasattr(collate_fn, '__call__') and collate_fn is not None:
        msg = 'collate_fn should be Type Callable or NoneType, but got {}.'.format(
            type(collate_fn))
        raise TypeError(msg)

    if not isinstance(pin_memory, bool):
        msg = 'pin_memory should be Type bool, but got {}.'.format(
            type(shuffle))
        raise TypeError(msg)

    dataset.shuffle = shuffle
    dataset.batch_size = batch_size
    dataset.num_workers = num_workers
    dataset.prefetch_factor = prefetch_factor
    dataset.collate_fn = collate_fn
    dataset.pin_memory = pin_memory
    return dataset


def generate_models(models: Dict[str, nn.Module],
                    is_load: bool = False,
                    device: torch.device = torch.device('cpu'),
                    data_parallel: bool = False,
                    *,
                    model_dir: str = '') -> Dict[str, torch.nn.Module]:
    '''
    Preprocess models in dict.

    Args:
        models: A dict of PyTorch models.
        is_load: If True, will load pretrained parameters to models.
        device: The device in which the model is in. It could be cuda device or cpu device.
        data_parallel: If True, the model will be loaded into multi-gpus and process data in parallel.
        model_dir: The dir which the model's parameters are stored in.

    Returns:
        A dict contrains preprocessed models.
    '''
    def generate_model(model, parameters_path, device, data_parallel):
        if parameters_path != '':
            model = load_model(model, parameters_path)

        if data_parallel == True:
            model = nn.DataParallel(module=model)

        model = to_device(model, device)
        return model

    if is_load == True:
        assert model_dir != '', 'model_dir shouldn\'t be \'\' if is_load is True.'

    for key in models.keys():
        if is_load == True:
            parameters_path = path.join(model_dir, models[key].name)
        else:
            parameters_path = ''

        models[key] = generate_model(models[key], parameters_path, device,
                                     data_parallel)

    return models


def generate_optimizer(model, optimizer, learning_rate):
    '''
    Create optimizers for models.

    Type `M`: torch.nn.Module
    Type `O`: torch.optim.Optimizer

    Args:
        model(torch.nn.Module): A model need to update parameters druing training.
        optimizer(torch.optim.Optimizer): A optimizer class to use its instance to update model's parameters.
        learning_rate(float): The step size each time the model's parameters are updated.

    Returns:
        (Union[O, Dict[str, O], Sequence[O]]): A container contains optimizers or a optimizer.
    '''
    model_optimizer = optimizer(model.parameters(), lr=learning_rate)
    return model_optimizer


def generate_data_loaders(datasets):
    '''
    Put datasets into corresponding data loaders.

    Type `DS`: Union[data.dataset.Dataset, data.ConcatDataset, dataset.AbstractDataset]
    Type `DL`: data.dataloader.DataLoader

    Args:
        datasets(Dict[str, Union[DS, Sequence[DS], Dict[DS]]): A dict contains datasets or a dataset, its keys can be train, val or test. If dataset in datasets does not have these attributes: batch_size, shuffle, num_workers, prefetch_factor, collate_fn. Use utils.generate_dataset to generate proper datasets.

    Returns:
        (Dict[str, Union[DL, Sequence[DL], Dict[DL]]): A dict contains data loaders or a data loader, its keys are the same as datasets.
    '''
    data_loaders = {}

    for key, _datasets in datasets.items():
        if isinstance(_datasets, (tuple, list)):
            data_loaders[key] = [
                generate_data_loader(dataset) for dataset in _datasets
            ]
        elif isinstance(_datasets, dict):
            data_loaders = {
                name: generate_data_loader(dataset)
                for name, dataset in _datasets.items()
            }
        else:
            data_loaders[key] = generate_data_loader(_datasets)

    return data_loaders


def generate_data_loader(dataset):
    '''
    Put dataset into corresponding data loader.

    Type `DS`: Union[data.dataset.Dataset, data.ConcatDataset, dataset.AbstractDataset]
    Type `DL`: data.dataloader.DataLoader

    Args:
        dataset(DS): A dataset and its keys can be train, val or test. If dataset does not have these attributes: batch_size, shuffle, num_workers, prefetch_factor, collate_fn. Use utils.generate_dataset to generate proper dataset.

    Returns:
        (DL): A data loader and its keys are the same as dataset.
    '''
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=dataset.batch_size,
                                  shuffle=dataset.shuffle,
                                  num_workers=dataset.num_workers,
                                  prefetch_factor=dataset.prefetch_factor,
                                  collate_fn=dataset.collate_fn,
                                  pin_memory=dataset.pin_memory)
    return data_loader
