"""Checked tokenizer loading and targeted filtering of expected offline notices."""
from contextlib import contextmanager
import logging

import datasets
from transformers import AutoConfig, AutoTokenizer

SPECIAL_NOTICE = 'Special tokens have been added in the vocabulary'
CACHE_NOTICES = {
    'datasets.load': "Using the latest cached version of the dataset since",
    'datasets.packaged_modules.cache.cache': 'Found the latest cached dataset configuration',
}


class ExpectedNoticeFilter(logging.Filter):
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def filter(self, record):
        return not record.getMessage().startswith(self.prefix)


@contextmanager
def without_expected_notices(notices):
    filters = [(logging.getLogger(name), ExpectedNoticeFilter(prefix))
               for name, prefix in notices.items()]
    for logger, message_filter in filters:
        logger.addFilter(message_filter)
    try:
        yield
    finally:
        for logger, message_filter in filters:
            logger.removeFilter(message_filter)


def load_tokenizer(model_name_or_path, padding_side):
    config = AutoConfig.from_pretrained(model_name_or_path)
    with without_expected_notices({'transformers.tokenization_utils_base': SPECIAL_NOTICE}):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    ids = list(tokenizer.get_vocab().values())
    if not ids or min(ids) < 0 or max(ids) >= config.vocab_size:
        raise ValueError('Tokenizer IDs exceed the pretrained model vocabulary; evaluation requires matching tokenizer and weights')
    return tokenizer


def load_dataset(*args, **kwargs):
    offline = (getattr(datasets.config, 'HF_DATASETS_OFFLINE', False)
               or getattr(datasets.config, 'HF_HUB_OFFLINE', False))
    with without_expected_notices(CACHE_NOTICES if offline else {}):
        return datasets.load_dataset(*args, **kwargs)
