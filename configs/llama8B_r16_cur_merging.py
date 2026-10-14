CACHE_DIR = ''
MODEL_DIR = ''
INGREDIENTS_PATH = None
PTM_PATH = ""

config = {
    'dataset': [
        {'name': 'snli', 'mask_class': None},
        {'name': 'mnli', 'val_fraction': 0.2, 'mask_class': None},
        {'name': 'sick', 'mask_class': None},
        {'name': 'qnli', 'val_fraction': 0.2, 'mask_class': 1},
        {'name': 'rte', 'val_fraction': 0.5, 'mask_class': 1},
        {'name': 'scitail', 'mask_class': 2},
    ],
    'model': {
        'name': 'meta-llama/Meta-Llama-3-8B',
        'ptm_path': PTM_PATH,
        'cachedir': CACHE_DIR,
        'bases': [
            'hoffman-lab/KnOTS-Llama3_8B_lora_R16_snli',
            'hoffman-lab/KnOTS-Llama3_8B_lora_R16_mnli',
            'hoffman-lab/KnOTS-Llama3_8B_lora_R16_sick',
            'hoffman-lab/KnOTS-Llama3_8B_lora_R16_qnli',
            'hoffman-lab/KnOTS-Llama3_8B_lora_R16_rte',
            'hoffman-lab/KnOTS-Llama3_8B_lora_R16_scitail',
        ],
        'ft_config': {
            'type': 'lora',
            'subtype': 'peft',
        },
        'peft_config': {
            'task_type': 'SEQ_CLS',
            'inference_mode': False,
            'r': 16,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        },
    },
    'task_merge_config': {
        'ingredients_path': INGREDIENTS_PATH,
        'representation': 'matrix_per_layer',
        'sign_resolve_mode': 'sum_of_values',
        'topK': 100,
        'merge_method': 'ties',
        'merge_space': 'svdcur',
        'merging_type': 'mean',
        'scaling_coeffs': 0.4,
        'concat_across_output': True,
        'svd_cur_weight_mode': 'energy',
        'svd_cur_svd_scaling_coeffs': 0.4,
        'svd_cur_cur_scaling_coeffs': 0.4,
        'dare': False,
        'dare_pruning_coeffs': 0.0,
    },
    'eval_type': 'logits',
}
