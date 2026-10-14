"""CPU cache preparation followed by protected Modal GPU evaluation."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import monotonic
import uuid

import modal
from modal_app import image, cache, results, check_checkpoints_ready, app as baseline_app
from private_eval.scoring import load_reference, relative_scores
from research_agent.repository import public_repository

# Modal may import this module from /root; uploaded runtime assets live in /workspace.
ROOT = Path(__file__).resolve().parent if modal.is_local() else Path('/workspace/research_agent')
app = modal.App('lora-research-agent')
io_volume = modal.Volume.from_name('lora-candidate-io', create_if_missing=True)
inputs_volume = modal.Volume.from_name('lora-research-inputs', create_if_missing=True)
grader_image = image.add_local_dir(ROOT, '/workspace/research_agent',
                                   ignore=['references/**', '**/__pycache__/**'])
worker_image = (modal.Image.debian_slim(python_version='3.10')
                .pip_install('torch==2.2.2', 'numpy==1.26.4', 'safetensors==0.4.1',
                             'transformers==4.40.2', 'peft==0.3.0', 'accelerate==0.26.1',
                             'huggingface-hub==0.20.2')
                .add_local_file(ROOT / 'worker.py', '/worker.py', copy=True))


@app.function(image=grader_image, cpu=8, memory=196608, timeout=3600,
              max_containers=1, scaledown_window=2,
              volumes={'/cache': cache, '/fixed-inputs': inputs_volume})
def prepare_inputs():
    """Cache fixed tensors on CPU before allocating a GPU."""
    started = monotonic()
    os.chdir('/workspace')
    sys.path.insert(0, '/workspace')
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    # ML imports follow the worker's offline/path setup.
    from research_agent.bridge import prepare_inputs as prepare
    from research_agent.inputs_cache import cache_key, ensure_inputs
    from eval_scripts.nli_fixed_test import progress
    cache.reload()
    inputs_volume.reload()
    ready = Path('/cache/checkpoints-ready.json')
    if not ready.exists():
        raise RuntimeError('Run baseline --download first')
    key = cache_key(json.loads(ready.read_text()), ROOT.parent)
    folder, hit = ensure_inputs('/fixed-inputs', key, prepare, progress)
    progress('Committing fixed-input cache before GPU allocation')
    inputs_volume.commit()
    return {'cache_key': key, 'cache_hit': hit, 'preparation_seconds': monotonic() - started}


@app.function(image=grader_image, gpu='A100-40GB', cpu=8, memory=196608,
              timeout=3600, max_containers=1, scaledown_window=2,
              volumes={'/cache': cache, '/results': results, '/candidate-io': io_volume,
                       '/fixed-inputs': inputs_volume.with_mount_options(read_only=True)})
def evaluate_candidate(source: str, split: str = 'val', cache_key: str = '', sandbox_image_id: str = ''):
    # Worker-only imports follow offline/path initialization.
    os.chdir('/workspace')
    sys.path.insert(0, '/workspace')
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    from safetensors.torch import load_file
    from research_agent.bridge import evaluate
    from research_agent.inputs_cache import cache_folder, read_manifest
    from eval_scripts.nli_fixed_test import progress
    reference = load_reference(split)
    folder = cache_folder('/fixed-inputs', cache_key)
    inputs_volume.reload()
    if read_manifest(folder) is None or not sandbox_image_id:
        raise RuntimeError('Prepare the fixed inputs and sandbox image before evaluation')
    experiment = uuid.uuid4().hex
    output = Path('/results/agent', experiment)
    output.mkdir(parents=True)
    (output / 'candidate.py').write_text(source)
    metadata = {'experiment_id': experiment, 'split': split, 'input_cache_key': cache_key,
                'sandbox_image_id': sandbox_image_id,
                'sha256': hashlib.sha256(source.encode()).hexdigest()}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    def execute(code, shapes):
        progress(f'Preparing candidate files; experiment {experiment}')
        attempt = Path('/candidate-io', experiment)
        attempt.mkdir(parents=True)
        (attempt / 'candidate.py').write_text(code)
        public = attempt / 'repository'
        public.mkdir()
        for name, content in public_repository().items():
            (public / name).write_text(content)
        # Only small per-attempt files are written; model inputs remain in their cache.
        io_volume.commit()
        progress('Creating fresh sandbox with cached image and read-only model inputs')
        sandbox = modal.Sandbox.create(
            'python', '/worker.py', '/data', '/inputs', app=app,
            image=modal.Image.from_id(sandbox_image_id),
            cpu=8, memory=98304, timeout=600, block_network=True,
            volumes={'/data': io_volume.with_mount_options(sub_path=f'/{experiment}'),
                     '/inputs': inputs_volume.with_mount_options(
                         sub_path=f'/{cache_key}/public', read_only=True)})
        try:
            progress(f'Sandbox created: {sandbox.object_id}; waiting for candidate execution')
            sandbox.wait()
            returncode = sandbox.returncode
            progress(f'Sandbox exited with code {returncode}; terminating sandbox')
        finally:
            sandbox.terminate(wait=True)
        io_volume.reload()
        if returncode != 0:
            detail = attempt / 'error.json'
            message = ''
            if detail.is_file() and not detail.is_symlink():
                with detail.open() as stream:
                    message = stream.read(2500)
            raise RuntimeError(f'Candidate sandbox exited with code {returncode}: {message}')
        weights = attempt / 'output.safetensors'
        if weights.is_symlink() or not weights.is_file():
            raise ValueError('Missing regular candidate output file')
        maximum_bytes = sum(math.prod(shape) * 8 for shape in shapes.values()) + 1024**2
        if weights.stat().st_size > maximum_bytes:
            raise ValueError('Candidate output exceeds the permitted tensor size')
        # Tensor-only interchange: never deserialize candidate-controlled pickle.
        return load_file(str(weights))

    try:
        report = evaluate(source, str(output), execute, split, inputs_dir=folder)
        report['scores'] = relative_scores(report, reference)
        report['score_type'] = 'reference_relative_percent'
        report.update(metadata)
        (output / 'result.json').write_text(json.dumps(report, indent=2))
        return report
    except Exception as error:
        (output / 'error.json').write_text(json.dumps({**metadata, 'error': str(error)}))
        raise
    finally:
        results.commit()


def prepare_evaluation():
    """Called locally inside a running app: build the image, then prepare CPU inputs."""
    print('Preparing sandbox image before CPU input preparation and GPU allocation', flush=True)
    started = monotonic()
    with modal.enable_output():
        built = worker_image.build(app=app)
    image_seconds = monotonic() - started
    prepared = prepare_inputs.remote()
    return {**prepared, 'sandbox_image_id': built.object_id,
            'image_build_seconds': image_seconds}


def run_cache_check(split, output_dir):
    """Sequential operator-only cache check with separate reports and timings."""
    folder = Path(output_dir) / uuid.uuid4().hex
    folder.mkdir(parents=True)
    summaries = []
    previous = None
    for name in ('average', 'task_arithmetic'):
        source = (ROOT / 'examples' / f'{name}.py').read_text()
        (folder / f'{name}.py').write_text(source)
        started = monotonic()
        prepared = prepare_evaluation()
        prep_seconds = monotonic() - started
        if previous is not None:
            if prepared['cache_key'] != previous['cache_key'] or not prepared['cache_hit']:
                raise RuntimeError('Second preparation did not reuse the same input cache; GPU evaluation skipped')
            if prepared['sandbox_image_id'] != previous['sandbox_image_id']:
                raise RuntimeError('Sandbox image changed between runs; GPU evaluation skipped')
        print(f"{name}: cache_hit={prepared['cache_hit']}, preparation={prep_seconds:.1f}s", flush=True)
        evaluation_started = monotonic()
        report = evaluate_candidate.remote(source, split, prepared['cache_key'], prepared['sandbox_image_id'])
        summary = {'candidate': name, 'split': split, **prepared,
                   'preparation_call_seconds': prep_seconds,
                   'evaluation_call_seconds': monotonic() - evaluation_started,
                   'total_seconds': monotonic() - started,
                   'experiment_id': report['experiment_id'], 'scores': report['scores'],
                   'raw_accuracy_percent': {task: result['accuracy_percent']
                                            for task, result in report['tasks'].items()}}
        (folder / f'{name}.json').write_text(json.dumps({'setup': summary, 'report': report}, indent=2))
        summaries.append(summary)
        (folder / 'summary.json').write_text(json.dumps(summaries, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        previous = prepared
    print(f'Cache-check reports saved to {folder.resolve()}', flush=True)
    return summaries


@app.local_entrypoint()
def main(candidate: str = '', split: str = 'val', cache_check: bool = False, output_dir: str = 'cache-checks'):
    if not candidate and not cache_check:
        raise ValueError('Provide --candidate path/to/candidate.py')
    if candidate and cache_check:
        raise ValueError('Choose either --candidate or --cache-check')
    if split not in {'val', 'test'}:
        raise ValueError('Invalid split')
    if not cache_check:
        source = Path(candidate).read_text()
    with baseline_app.run():
        check_checkpoints_ready.remote()
    if cache_check:
        run_cache_check(split, output_dir)
        return
    prepared = prepare_evaluation()
    print(json.dumps(evaluate_candidate.remote(source, split,
                     prepared['cache_key'], prepared['sandbox_image_id']), indent=2))
