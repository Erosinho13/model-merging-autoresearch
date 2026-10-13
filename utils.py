from inspect import getmembers, isfunction
import random
import numpy as np
import torch
import merging_functions
import masking_ops

def get_merging_fn(name):
    """Get the merging function from name tag."""
    vector_fns = dict([(k.replace('_merging', ''), v) for (k, v) in getmembers(merging_functions, isfunction) if '_merging' in k])
    return vector_fns[name]


def get_mask_fn(name):
    """Get the masking function from name tag."""
    masking_fns = dict([(k.replace('_masking', ''), v) for (k, v) in getmembers(masking_ops, isfunction) if '_masking' in k])
    return masking_fns[name]


def set_seed(seed):
    """Set the seed for reproducibility."""
    print("Setting Seed to {}".format(seed))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
