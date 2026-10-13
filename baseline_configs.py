BASELINES = {
    'task_arithmetic': {
        'representation': 'vector',
        'merge_method': 'tv',
        'merging_type': 'sum',
        'merge_other_params': False,
        'scaling_coeffs': [0.5]
    },
    'ties': {
        'representation': 'vector',
        'sign_resolve_mode': 'sum_of_values',
        'topK': 40,
        'merge_method': 'ties',
        'merging_type': 'mean',
        'scaling_coeffs': [1.0],
        'dare': False,
        'dare_pruning_coeffs': 0.0
    },
    'dare_ties': {
        'representation': 'vector',
        'merge_method': 'ties',
        'sign_resolve_mode': 'sum_of_values',
        'merging_type': 'mean',
        'topK': 100,
        'scaling_coeffs': [0.4],
        'dare': True,
        'dare_pruning_coeffs': 0.8,
        'dare_seed': 420
    },
    'average': {
        'representation': 'vector',
        'merge_method': 'tv',
        'merging_type': 'sum',
        'merge_other_params': False,
        'scaling_coeffs': [0.3333333333333333]
    }
}
