"""Original loader settings for the three selected NLP datasets."""

snli = {
    'dir': './datasets/snli',
    'type' : 'snli',
    'batch_size': 2,
    'num_workers': 16,
    'model_name_or_path' : "meta-llama/Meta-Llama-3-8B",
    'mask_class' : None,
}

mnli = {
    'dir': './datasets/mnli',
    'type' : 'mnli',
    'batch_size': 2,
    'num_workers': 16,
    'val_fraction': 0.2,
    'model_name_or_path' : "meta-llama/Meta-Llama-3-8B",
    'mask_class' : None,
    
}

scitail = {
    'dir': './datasets/scitail',
    'type' : 'scitail',
    'batch_size': 2,
    'num_workers': 16,
    'model_name_or_path' : "meta-llama/Meta-Llama-3-8B",
    'mask_class' : 2,
}
