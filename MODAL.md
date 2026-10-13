# Run this repository on Modal

Activate your `eai-tests` environment, where Modal is installed. Run all commands
below from this repository's root directory.

## First GPU check

If this environment is not authenticated yet, run `modal token new`.

```bash
python -m modal run modal_app.py
```

This builds a Linux environment, uploads the repository source, imports the
merging code, and verifies a small merge on an L4 GPU. It downloads no Llama
checkpoints and needs no Hugging Face secret. Expect GPU information and two
`ok` checks. The initial image build can take several minutes.

The application has no continuously running deployment. GPU functions have a
single-container limit and a short idle window. The smoke check has a five-minute
execution timeout; evaluation has a one-hour timeout. These are execution limits,
not a guaranteed dollar spending cap.

## CPU-only preparation check

Create a Modal secret named `huggingface` with key `HF_TOKEN` using the Modal
dashboard. Do not put the token in code or chat. The token must have access to
`meta-llama/Meta-Llama-3-8B`, including acceptance of its access conditions.

```bash
python -m modal run modal_app.py --prepare
```

This command uses **no GPU**. It loads the three raw datasets, checks the gated
Llama tokenizer, and inspects adapter configurations/file listings. It does not
download Llama or adapter weights, tokenize the complete datasets, or evaluate
models. It needs the `huggingface` secret described above. CPU and storage still
incur charges. There is a 15-minute execution timeout and one-container limit.

The benchmark selects **SNLI, MNLI and SciTail**, with matching adapters, in
`configs/nlp_benchmark.py`. Preparation and evaluation use this shared selection.
MNLI retains the repository's deterministic validation partition: 20% for agent
validation and 80% for held-out evaluation. SNLI and SciTail use their separate
validation/test splits. SICK, QNLI and RTE are not loaded or merged.

Successful preparation writes `/preparation.json` in `lora-merging-results`.
The report includes package versions and immutable adapter revisions. It does
not yet verify classification heads or establish baseline accuracy.

## Evaluation is temporarily disabled

The new fixed-baseline runner is separate from the legacy `--module` command:

```bash
python -m modal run modal_app.py --download
python -m modal run modal_app.py --method task_arithmetic
python -m modal run modal_app.py --method average
python -m modal run modal_app.py --method ties
python -m modal run modal_app.py --method dare_ties
python -m modal run modal_app.py --method knots_ties
python -m modal run modal_app.py --method knots_dare_ties
python -m modal run modal_app.py --method base
python -m modal run modal_app.py --method snli
python -m modal run modal_app.py --method mnli
python -m modal run modal_app.py --method scitail
```

Download on CPU first; proceed to GPU runs only after checkpoint/head checks pass.
`base` evaluates the unadapted backbone; task names evaluate that single adapter.
Each is scored on all three tasks using each dataset's original trained head.
`average` uses Task Arithmetic with coefficient 1/3 for each of the three updates.
To compare TIES hyperparameters on validation, use `--method ties --split val
--scale 0.5 --top-k 40` (on one command line). Candidate results go under `/hp`
in the results volume; select settings on validation before running `--split test`.
Each method evaluates the three test subsets with released repository defaults;
DARE methods use `DARE_SEEDS` (currently seed 420). Deterministic methods run once.
KnOTS methods use the released SVD alignment with column concatenation and the
original singular-value threshold; alignment runs on GPU before evaluation.
These three-task fixed-setting runs are not an exact reproduction of the paper's
six-task, validation-tuned table. Results are saved under
`/fixed-test` in `lora-merging-results`. The new runner requests A100-40GB for
unquantized float32 inference; memory fit and numerical results remain unverified.

The `--module` entry point now raises a local error **before allocating a GPU**.
The evaluation script now selects three tasks but still selects KnOTS-TIES and
requires `heads.pt`. We must configure the method and measure memory before enabling
GPU evaluation. The dormant remote runner retains its original A100 resource
declaration; it is not invoked by the preparation command.

HF downloads
persist in `lora-merging-hf-cache`; evaluation CSVs persist in
`lora-merging-results`. For example, download outputs with:

```bash
modal volume get lora-merging-results /csvs_new ./modal-results
```

Only explicitly listed source directories and files are uploaded. Local secrets,
Git metadata, checkpoints and datasets outside those directories are excluded.
Keep these source directories free of credentials and private data.

## Dependency reconciliation

`requirements-modal.txt` restores the original PEFT 0.3.0, Accelerate 0.26.1,
Hugging Face Hub 0.20.2, NumPy 1.26.3 and SentencePiece 0.1.97 pins, alongside
Transformers 4.40.2 and Datasets 2.16.1. Hub 0.20.2 satisfies Transformers 4.40.2's
declared minimum; there is no established conflict between those two pins.

Known differences: Python is 3.10 rather than 3.9.18; scikit-learn is 1.4.2 rather
than 0.24.0 for available Python 3.10 wheels. PyTorch 2.2.2 and torchvision 0.17.2
are explicit choices because the original environment does not pin them. Notebook
and unrelated training/vision dependencies are omitted.

These pins have been defined and locally checked, but the revised image still
needs a cloud build/import check and individual-adapter accuracy verification.
Dependency reconciliation is not evidence of numerical reproduction.
