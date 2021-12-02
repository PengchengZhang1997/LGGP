"""
Utilities for randomness.
"""
import torch as th

# ---------- Multiprocessing ----------
def get_truncnorm_tensor(shape, *, mean = 0, std = 1, limit = 2) -> th.Tensor:
    """
    Create and return normally distributed tensor, except values with too much deviation are discarded.

    Args:
        shape: tensor shape
        mean: normal mean
        std: normal std
        limit: which values to discard

    Returns:
        Filled tensor with shape (*shape)
    """
    assert isinstance(shape, (tuple, list)), f"shape {shape} is not a tuple or list of ints"
    num_examples = 8
    tmp = th.empty(shape + (num_examples,)).normal_()
    valid = (tmp < limit) & (tmp > -limit)
    _, ind = valid.max(-1, keepdim=True)
    return tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)


def fill_tensor_with_truncnorm(input_tensor: th.Tensor, *, mean: float = 0, std: float = 1, limit: float = 2) -> None:
    """
    Fill given input tensor with a truncated normal dist.

    Args:
        input_tensor: tensor to be filled
        mean: normal mean
        std: normal std
        limit: which values to discard
    """
    # get truncnorm values
    tmp = get_truncnorm_tensor(input_tensor.shape, mean=mean, std=std, limit=limit)
    # fill input tensor
    input_tensor[...] = tmp[...]