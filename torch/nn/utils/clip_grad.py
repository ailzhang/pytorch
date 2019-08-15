import warnings
import torch
from torch._six import inf


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    device = parameters[0].device if parameters else torch.device('cpu')
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = torch.zeros([], device=device if parameters else None)
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type) ** norm_type
            total_norm.add_(param_norm)
        total_norm = (total_norm ** (1. / norm_type))
    clip_coef = torch.tensor(max_norm, device=device) / (total_norm + 1e-6)
    for p in parameters:
        p.grad.data.mul_(torch.where(clip_coef < 1, clip_coef, torch.tensor(1., device=device)))
    return total_norm


def clip_grad_norm(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    .. warning::
        This method is now deprecated in favor of
        :func:`torch.nn.utils.clip_grad_norm_`.
    """
    warnings.warn("torch.nn.utils.clip_grad_norm is now deprecated in favor "
                  "of torch.nn.utils.clip_grad_norm_.", stacklevel=2)
    return clip_grad_norm_(parameters, max_norm, norm_type)


def clip_grad_value_(parameters, clip_value):
    r"""Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.clamp_(min=-clip_value, max=clip_value)
