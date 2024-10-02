from typing import List

import numpy as np


# def collate_outputs(outputs: List[dict]):
#     """
#     used to collate default train_step and validation_step outputs. If you want something different then you gotta
#     extend this

#     we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
#     """
#     collated = {}
#     for k in outputs[0].keys():
#         if np.isscalar(outputs[0][k]):
#             collated[k] = [o[k] for o in outputs]
#         elif isinstance(outputs[0][k], np.ndarray):
#             collated[k] = np.vstack([o[k][None] for o in outputs])
#         elif isinstance(outputs[0][k], list):
#             collated[k] = [item for o in outputs for item in o[k]]
#         else:
#             raise ValueError(f'Cannot collate input of type {type(outputs[0][k])}. '
#                              f'Modify collate_outputs to add this functionality')
            
    
#     return collated

def collate_outputs(outputs: List[dict]):
    if not outputs:
        return {}

    collated = {k: [] for k in outputs[0].keys()}

    for output in outputs:
        for k, v in output.items():
            if np.isscalar(v) or isinstance(v, (np.float32, np.float64)):
                collated[k].append(v)
            elif isinstance(v, np.ndarray):
                collated[k].append(v[None])
            elif isinstance(v, list):
                collated[k].extend(v)
            else:
                raise ValueError(f'Cannot collate input of type {type(v)} for key {k}. '
                                 f'Modify collate_outputs to add this functionality')

    for k, v in collated.items():
        if isinstance(v[0], np.ndarray):
            collated[k] = np.vstack(v)
        elif np.isscalar(v[0]) or isinstance(v[0], (np.float32, np.float64)):
            collated[k] = np.array(v)

    return collated