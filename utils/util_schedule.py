import torch

__all__ = ['linear_schedule', 'cosine_schedule']


def linear_schedule(timesteps: int) -> torch.Tensor:
    '''
    Return linear schedule range from 0 to 1.

    Args:
        timesteps(int): the number of steps that this schedule has.

    Returns:
        (torch.Tensor): schedule.
    '''
    betas = torch.linspace(0.0, 1.0, timesteps)
    alphas = 1. - betas
    return alphas, betas


def cosine_schedule(timesteps: int) -> torch.Tensor:
    '''
    Return cosine schedule range from 0 to 1.

    Args:
        timesteps(int): the number of steps that this schedule has.
    Returns:
        (torch.Tensor): schedule.
    '''
    s = 0.008
    total_steps = timesteps + 1
    steps = torch.linspace(0, total_steps, total_steps)
    f_steps = torch.cos(0.5 * torch.pi * (steps / total_steps + s) /
                        (1 + s))**2
    a_steps = f_steps / f_steps[0]
    betas = 1 - (a_steps[1:] / a_steps[:-1])
    betas = torch.clip(betas, min=0, max=0.999)
    alphas = 1. - betas
    return alphas, betas
