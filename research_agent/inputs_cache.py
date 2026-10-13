"""Trusted fixed-input cache; only public/ is mounted into candidate sandboxes."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import torch
from safetensors.torch import save_file

SCHEMA = 1
FILES = ('public/model.json', 'public/model.safetensors', 'public/inputs.safetensors',
         'heads.safetensors', 'evaluation.json')
SOURCES = ('research_agent/inputs_cache.py', 'research_agent/bridge.py', 'ft_handlers.py',
           'task_merger.py', 'utils.py', 'merging_functions.py', 'masking_ops.py',
           'merging_types.py', 'baseline_configs.py', 'configs/nlp_benchmark.py',
           'private_eval/model_utils.py', 'eval_scripts/nli_fixed_test.py',
           'configs/llama8B_r16_tv.py', 'requirements-modal.txt')


def cache_key(revisions, repository):
    payload = {'schema': SCHEMA, 'checkpoints': revisions,
               'python': list(sys.version_info[:2]),
               'runtime': {name: importlib.metadata.version(name)
                           for name in ('torch', 'peft', 'transformers', 'numpy', 'safetensors')},
               'sources': {name: hashlib.sha256((Path(repository) / name).read_bytes()).hexdigest()
                           for name in SOURCES}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_folder(root, key):
    if not re.fullmatch('[0-9a-f]{64}', key):
        raise ValueError('Invalid fixed-input cache key')
    return Path(root) / key


def read_manifest(folder):
    folder = Path(folder)
    try:
        manifest = json.loads((folder / 'complete.json').read_text())
        if (manifest['schema'] != SCHEMA or manifest['key'] != folder.name
                or set(manifest['sizes']) != set(FILES)):
            return None
        for name, size in manifest['sizes'].items():
            path = folder / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != size or size <= 0:
                return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=lambda v: sorted(v) if isinstance(v, set) else v.value))
    temporary.replace(path)


def write_tensors(path, tensors):
    temporary = path.with_suffix(path.suffix + '.tmp')
    save_file(tensors, str(temporary))
    temporary.replace(path)


def ensure_inputs(root, key, prepare, progress=print):
    """Load adapters/write tensors only on cache miss. Completion is recorded last."""
    folder = cache_folder(root, key)
    if read_manifest(folder):
        progress(f'Fixed-input cache hit: {key}')
        return folder, True
    progress(f'Fixed-input cache miss: {key}; preparing on CPU')
    # Incomplete entries must never be treated as reusable inputs.
    (folder / 'complete.json').unlink(missing_ok=True)
    public = folder / 'public'
    public.mkdir(parents=True, exist_ok=True)
    model, heads, raw, vectors, adapters = prepare()
    description = {'config': model.config.to_dict(),
                   'adapter_config': model.peft_config['default'].to_dict()}
    description['adapter_config'].pop('base_model_name_or_path', None)
    description['adapter_config'].pop('revision', None)
    write_json(public / 'model.json', description)
    state, seen = {}, set()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        # No trained classification heads are made available to candidate code.
        if '.score.' in name or name.startswith('score.'):
            value = value.new_zeros(value.shape)
        if value.data_ptr() in seen:
            value = value.clone()
        seen.add(value.data_ptr())
        state[name] = value
    size = sum(v.numel() * v.element_size() for v in state.values()) / 1024**3
    progress(f'Caching model tensors ({size:.2f} GiB)')
    write_tensors(public / 'model.safetensors', state)
    del state
    tensors = {f'task{i}/{k}': v.detach().contiguous()
               for i, vector in enumerate(vectors) for k, v in vector.items()}
    for i, adapter in enumerate(adapters):
        for name, factors in adapter.items():
            tensors[f'lora{i}/A/{name}'] = factors['A'].contiguous()
            tensors[f'lora{i}/B/{name}'] = factors['B'].contiguous()
            tensors[f'lora{i}/scaling/{name}'] = torch.tensor(factors['scaling'], dtype=torch.float64)
    progress('Caching task vectors and original LoRA factors')
    write_tensors(public / 'inputs.safetensors', tensors)
    del tensors
    write_tensors(folder / 'heads.safetensors', {k: v.contiguous().clone() for k, v in heads.items()})
    write_json(folder / 'evaluation.json', raw)
    # The trusted evaluator uses this to validate exported candidate weights.
    weight_shapes = {k: list(v.shape) for k, v in vectors[0].items()}
    manifest = {'schema': SCHEMA, 'key': key, 'weight_shapes': weight_shapes,
                'sizes': {name: (folder / name).stat().st_size for name in FILES}}
    write_json(folder / 'complete.json', manifest)
    progress('Fixed inputs cached; private heads remain outside the public mount')
    return folder, False
