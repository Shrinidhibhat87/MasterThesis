"""
The file provides an utility function that is responsible for adaptive
learning rate and weight decay parameters.
This is an essential part of training Mask2Former model
"""
import json

from torch import nn

adaptive_weight_decay = [
    {'first_incl_str': 'norm', 'second_incl_str': '', 'wd_mult': 0.0, 'lr_mult': 1.0},
    {
        'first_incl_str': 'encoder',
        'second_incl_str': 'pixel_level_module',
        'wd_mult': 1.0,
        'lr_mult': 0.1,
    },
    {
        'first_incl_str': 'pixel_level_module.encoder',
        'second_incl_str': 'norm',
        'wd_mult': 0.0,
        'lr_mult': 0.1,
    },
    {
        'first_incl_str': 'hidden_states_norms',
        'second_incl_str': 'pixel_level_module',
        'wd_mult': 0.0,
        'lr_mult': 1.0,
    },
    {
        'first_incl_str': 'relative_position_bias_table',
        'second_incl_str': '',
        'wd_mult': 0.0,
        'lr_mult': 0.1,
    },
    {'first_incl_str': 'queries_embedder', 'second_incl_str': '', 'wd_mult': 0.0, 'lr_mult': 1.0},
    {'first_incl_str': 'queries_features', 'second_incl_str': '', 'wd_mult': 0.0, 'lr_mult': 1.0},
    {'first_incl_str': 'level_embed', 'second_incl_str': '', 'wd_mult': 0.0, 'lr_mult': 1.0},
]


def map_weight_decay_to_params(model: nn.Module, initial_lr: float, weight_decay: float):
    """
    Adaptive mapping of weight decay and learning rate acquires the values.
        Currently these values are hard coded, but this would eventually
        be an option in the configuration file.

    For cases when there are multiple matches for the parameter, only the last
        or the most latest one would be selected.

    Thus the assignment of the weight decay and learning multiplier is done
        sequentially with the last configuration that it matches being chosen

    Args:
        model (nn.Module): The model that needs the weight decay and learning rate
            to be adapted
        initial_lr (float): The initial learning rate
    """
    param_dict = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            opt_config = [weight_decay, initial_lr]
            # Get the matching keys
            matching_keys = [
                wd_lr_config
                for wd_lr_config in adaptive_weight_decay
                if wd_lr_config['first_incl_str'] in name
                and wd_lr_config['second_incl_str'] in name
            ]
            # There are situations where there are more matching keys
            # In this case, use the last one
            if len(matching_keys) >= 1:
                last_matching_config = matching_keys[-1]
                opt_config[0] *= last_matching_config['wd_mult']
                opt_config[1] *= last_matching_config['lr_mult']

            opt_key = json.dumps(opt_config)
            param_dict[opt_key] = param_dict.get(opt_key, []) + [param]

    net_params = []
    for opt_key, param_list in param_dict.items():
        wd, lr = json.loads(opt_key)
        net_params.append({'params': param_list, 'weight_decay': wd, 'lr': lr})

    return net_params
