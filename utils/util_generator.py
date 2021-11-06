from numpy import mod
import torch
from torch._C import _graph_pool_handle
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from .util_device import to_device

__all__ = [
    'generate_dataset', 'generate_data_loader', 'generate_data_loaders',
    'generate_model', 'generate_optimizers', 'generate_optimizer'
]


def generate_dataset(dataset,
                     shuffle=False,
                     batch_size=1,
                     num_workers=1,
                     prefetch_factor=2,
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

    dataset.shuffle = shuffle
    dataset.batch_size = batch_size
    dataset.num_workers = num_workers
    dataset.prefetch_factor = prefetch_factor
    dataset.collate_fn = collate_fn
    return dataset


def generate_model(model,
                   learning_rate=0,
                   optimizer=None,
                   device=torch.device('cpu'),
                   data_parallel=False):
    '''
    Generate a model that has some necessary attributes.

    Args:
        model(torch.nn.Module): A model can process data.
        learning_rate(float): To indicate the step size of updating parameters in each backward propagation.
        optimizer(str): The optimizer used to update parameters.

    Returns:
        (torch.nn.Module): A model contains some necessary attributes.
    '''
    if data_parallel == True:
        model = nn.DataParallel(module=model)

    model = to_device(model, device)
    model.learning_rate = learning_rate
    model.optimizer = optimizer
    return model


def generate_optimizers(models):
    '''
    Create optimizers for models.

    Type `M`: torch.nn.Module
    Type `O`: torch.optim.Optimizer

    Args:
        models(Union[M, Dict[str, M], Sequence[M]]): A container contains models or a model, if model in models does not have necessary attributes: learning_rate, optimizer. Use generate_model to create proper model.

    Returns:
        (Union[O, Dict[str, O], Sequence[O]]): A container contains optimizers or a optimizer.
    '''
    if isinstance(models, (tuple, list)):
        optimizers = [generate_optimizer(model) for model in models]
    elif isinstance(models, dict):
        optimizers = {
            name: generate_optimizer(model)
            for name, model in models.items()
        }
    elif isinstance(models, torch.nn.Module):
        optimizers = generate_optimizer(models)

    return optimizers


def generate_optimizer(model):
    '''
    Create optimizers for models.

    Type `M`: torch.nn.Module
    Type `O`: torch.optim.Optimizer

    Args:
        models(Union[M, Dict[str, M], Sequence[M]]): A model, if model in models does not have necessary attributes: learning_rate, optimizer. Use generate_model to create proper model.

    Returns:
        (Union[O, Dict[str, O], Sequence[O]]): A optimizer to update the parameters of model.
    '''
    if model.optimizer == 'Adam':
        optim_class = optim.Adam
    elif model.optimizer == 'SGD':
        optim_class = optim.SGD
    elif model.optimizer == None:
        return None
    else:
        msg = 'model\'s optimizer should be Adam, SGD or NoneType. but got {}.'.format(
            model.optimizer)
        raise ValueError(msg)

    optimizer = optim_class(model.parameters(), lr=model.learning_rate)
    return optimizer


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
                                  collate_fn=dataset.collate_fn)
    return data_loader
