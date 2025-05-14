import torch
import random
import numpy as np
from tqdm import tqdm

from training import train


def run(tries, setup_fn, visualisers, seed=0, *args, **kwargs):
    kwargs['show_pbar'] = False
    kwargs['verbose'] = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    result = None
    for i in tqdm(range(tries)):
        result = train(*args, **kwargs, **setup_fn())
        for key in visualisers:
            visualisers[key].update(result)
    
    for key in visualisers:
        visualisers[key].display()
    
    return visualisers, result
