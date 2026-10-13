"""Local interface/integrity tests; no model API or Modal compute calls."""
import ast
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_agent.repository import public_repository, reference_files
from research_agent.task import check_source, research
from private_eval.scoring import load_reference, relative_scores
from research_agent.modal_runner import app, evaluate_candidate
import torch
from transformers import LlamaConfig, LlamaForSequenceClassification
from peft import LoraConfig, get_peft_model
from ft_handlers import LoRAHandler
from safetensors.torch import save_file, load_file

VALID = 'def merge(base_model, task_vectors, lora_adapters):\n    return base_model\n'


class PublicViewTests(unittest.TestCase):
    def test_private_definitions_are_excluded(self):
        files = public_repository()
        self.assertNotIn('eval_scripts/nli_fixed_test.py', files)
        self.assertNotIn('modal_app.py', files)
        self.assertEqual(files['task_merger.py'], Path('task_merger.py').read_text())
        self.assertEqual(files['utils.py'], Path('utils.py').read_text())
        self.assertEqual(files['baseline_configs.py'], Path('baseline_configs.py').read_text())
        for filename, source in files.items():
            if filename.endswith('.py'):
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or '').startswith(('private_eval', 'eval_scripts', 'dataset', 'configs')))
        self.assertNotIn('SVDMerger', files['task_merger.py'])
        self.assertNotIn('evaluate_logits', files['utils.py'])
        self.assertNotIn('prepare_llama', files['utils.py'])
        self.assertNotIn('knots', '\n'.join(files.values()).lower())
        original = ast.parse(Path('task_merger.py').read_text())
        exposed = ast.parse(files['task_merger.py'])
        original_classes = {n.name: ast.dump(n, include_attributes=False)
                            for n in original.body if isinstance(n, ast.ClassDef)}
        for node in exposed.body:
            if isinstance(node, ast.ClassDef):
                self.assertEqual(ast.dump(node, include_attributes=False), original_classes[node.name])

    def test_nested_references_and_symlink_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / 'research_agent/references'
            (refs / 'paper').mkdir(parents=True)
            (refs / 'README.md').write_text('operator instructions')
            (refs / 'paper/paper.md').write_text('research text')
            secret = root / 'secret.md'
            secret.write_text('private')
            (refs / 'leak.md').symlink_to(secret)
            with patch('research_agent.repository.ROOT', root):
                self.assertEqual(reference_files(), {'paper/paper.md': 'research text'})

    def test_signature_and_empty_stub(self):
        check_source(VALID, require_implementation=True)
        with self.assertRaises(ValueError):
            check_source('def merge(base_model, task_vectors):\n    return base_model\n')
        check_source('def merge(base_model, task_vectors, lora_adapters):\n    pass\n')
        with self.assertRaises(ValueError):
            check_source('def merge(base_model, task_vectors, lora_adapters):\n    pass\n', True)
        with self.assertRaises(ValueError):
            check_source('def merge(base_weights, task_vectors):\n    pass\n')


class ScoringTests(unittest.TestCase):
    def test_test_mean_and_validation_raw(self):
        reference = load_reference('test')
        raw = reference['reference_accuracy_percent']
        report = {'tasks': {t: {'accuracy_percent': v} for t, v in raw.items()}}
        self.assertEqual(relative_scores(report, reference), dict(snli=100, mnli=100, scitail=100, avg=100))
        report['tasks']['snli']['accuracy_percent'] *= .5
        scores = relative_scores(report, reference)
        self.assertAlmostEqual(scores['avg'], 250 / 3)
        self.assertIsNone(load_reference('val'))
        scores = relative_scores(report, None)
        self.assertAlmostEqual(scores['avg'], sum(v['accuracy_percent'] for v in report['tasks'].values()) / 3)
        with self.assertRaises(ValueError):
            load_reference('train')


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def setup_run(self, directory):
        task = research(run_dir=directory, max_evaluations=3)
        state = SimpleNamespace(token_usage=100, token_limit=30000, tools=[])
        async def unused(*args, **kwargs):
            raise AssertionError('No model generation expected')
        await task.solver[0](state, unused)
        await task.solver[1](state, unused)
        folder = next(Path(directory).iterdir())
        return task, state.tools, folder

    async def test_reads_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, tools, folder = await self.setup_run(tmp)
            budget, read, papers, write, evaluate, submit = tools
            self.assertIn('candidate.py', json.loads(await read()))
            self.assertIn('pass', await read(path='candidate.py'))
            for private in ('../README.md', '../utils.py', 'eval_scripts/nli_fixed_test.py', 'modal_app.py'):
                self.assertEqual(await read(path=private), 'Invalid file or line range')
            self.assertIn('empty', await evaluate(explanation="# Algorithm\nInitial method"))
            self.assertEqual(json.loads(await budget())['evaluations_used'], 0)
            self.assertIn('Saved', await write(code=VALID))
            self.assertIn('nonempty', await evaluate(explanation='   '))
            self.assertEqual(json.loads(await budget())['evaluations_used'], 0)
            with patch.object(app, 'run'), patch.object(evaluate_candidate, 'remote', return_value={'scores': dict(snli=90, mnli=90, scitail=90, avg=90)}):
                await evaluate(explanation="# Algorithm\nBest method")
            worse = VALID + '\n# lower scoring variant\n'
            await write(code=worse)
            with patch.object(app, 'run'), patch.object(evaluate_candidate, 'remote', return_value={'scores': dict(snli=80, mnli=80, scitail=80, avg=80)}):
                await evaluate(explanation="# Algorithm\nWorse method")
            await task.cleanup(None)
            self.assertEqual((folder / 'submission.py').read_text(), VALID)
            self.assertEqual((folder / 'algorithm.md').read_text(), '# Algorithm\nBest method\n')
            self.assertEqual((folder / 'algorithm_002.md').read_text(), '# Algorithm\nWorse method\n')
            self.assertEqual(json.loads((folder / 'submission.json').read_text())['selection'], 'fallback_best_validation')

    async def test_test_feedback_and_submission_without_tuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, tools, folder = await self.setup_run(tmp)
            await tools[3](code=VALID)
            with patch.object(app, 'run'), patch.object(evaluate_candidate, 'remote',
                              return_value={'scores': dict(snli=101, mnli=102, scitail=100, avg=101),
                                            'average_accuracy_percent': 90}) as remote:
                feedback = json.loads(await tools[4](explanation='# Algorithm\nNo tuning', split='test'))
                remote.assert_called_once_with(VALID, 'test')
            self.assertEqual(feedback['score_type'], 'reference_relative_percent')
            self.assertNotIn('average_accuracy_percent', feedback)
            await task.cleanup(None)
            self.assertEqual(json.loads((folder / 'submission.json').read_text())['selection'], 'fallback_best_test')

    async def test_explicit_submission_and_no_valid_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, tools, folder = await self.setup_run(tmp)
            await tools[3](code=VALID)
            with patch.object(app, 'run'), patch.object(evaluate_candidate, 'remote', return_value={'scores': dict(snli=75, mnli=75, scitail=75, avg=75)}):
                await tools[4](explanation="# Algorithm\nEvaluated method")
            await tools[5](explanation="# Algorithm\nFinal explanation with validation evidence")
            await task.cleanup(None)
            self.assertEqual(json.loads((folder / 'submission.json').read_text())['selection'], 'agent_submission')
            self.assertEqual((folder / 'algorithm.md').read_text(), '# Algorithm\nFinal explanation with validation evidence\n')
            self.assertEqual(await tools[3](code=VALID), 'Submission is frozen')
        with tempfile.TemporaryDirectory() as tmp:
            task, tools, folder = await self.setup_run(tmp)
            await task.cleanup(None)
            self.assertFalse((folder / 'submission.py').exists())
            self.assertEqual(json.loads((folder / 'submission.json').read_text())['selection'], 'none')


class ModelWorkerTests(unittest.TestCase):
    def test_original_factor_extraction(self):
        model = get_peft_model(LlamaForSequenceClassification(LlamaConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, num_labels=3)),
            LoraConfig(task_type='SEQ_CLS', r=2, lora_alpha=6, target_modules=['q_proj']))
        handler = LoRAHandler(model)
        factors = handler.get_lora_factors()
        dense = handler.get_ft_parameters()
        self.assertEqual(set(factors), set(dense))
        for key, factor in factors.items():
            self.assertEqual(factor['scaling'], 3)
            self.assertEqual(factor['A'].shape, (2, 8))
            self.assertEqual(factor['B'].shape, (8, 2))
            torch.testing.assert_close(factor['B'] @ factor['A'], dense[key])
            factor['A'].zero_()
        self.assertTrue(any(torch.count_nonzero(module.lora_A['default'].weight)
                            for module in model.modules() if hasattr(module, 'lora_A')))

    def test_real_model_roundtrip(self):
        model = get_peft_model(LlamaForSequenceClassification(LlamaConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, num_labels=3)),
            LoraConfig(task_type='SEQ_CLS', r=2, lora_alpha=2, target_modules=['q_proj']))
        state = model.state_dict()
        key = next(k for k in state if 'q_proj' in k and 'lora_' not in k)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'repository').mkdir()
            for name, text in public_repository().items():
                (folder / 'repository' / name).write_text(text)
            description = {'config': model.config.to_dict(), 'adapter_config': model.peft_config['default'].to_dict()}
            (folder / 'model.json').write_text(json.dumps(description, default=lambda v: sorted(v) if isinstance(v, set) else v.value))
            save_file({k: v.detach().clone().contiguous() for k, v in state.items()}, str(folder / 'model.safetensors'))
            inputs = {f'task{i}/{key}': torch.ones_like(state[key]) for i in range(3)}
            A = torch.arange(16, dtype=torch.float32).reshape(2, 8)
            for i in range(3):
                inputs[f'lora{i}/A/{key}'] = A.clone()
                inputs[f'lora{i}/B/{key}'] = torch.full((8, 2), float(i + 1))
                inputs[f'lora{i}/scaling/{key}'] = torch.tensor(.5 + i, dtype=torch.float64)
            save_file(inputs, str(folder / 'inputs.safetensors'))
            source = ('import torch\nfrom merging_functions import tv_merging\n'
                      'def merge(base_model, task_vectors, lora_adapters):\n'
                      '    state = base_model.state_dict()\n'
                      '    with torch.no_grad():\n'
                      '        for key in task_vectors[0]:\n'
                      '            updates = [a[key]["scaling"] * (a[key]["B"] @ a[key]["A"]) for a in lora_adapters]\n'
                      '            delta, _, _ = tv_merging([v.flatten() for v in updates], merging_type="sum")\n'
                      '            state[key].add_(delta.reshape(state[key].shape) / 3)\n'
                      '    return base_model\n')
            (folder / 'candidate.py').write_text(source)
            worker = Path('research_agent/worker.py').resolve()
            result = subprocess.run([sys.executable, str(worker), tmp], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = load_file(str(folder / 'output.safetensors'))
            self.assertEqual(set(output), {key})
            expected = state[key] + sum((.5 + i) * (torch.full((8, 2), float(i + 1)) @ A) for i in range(3)) / 3
            torch.testing.assert_close(output[key], expected)


if __name__ == '__main__':
    unittest.main()
