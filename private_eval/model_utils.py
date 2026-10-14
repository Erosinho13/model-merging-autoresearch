from copy import deepcopy
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, PeftModel


def evaluate_logits(model, loader, device, mask_class = None, eval = True):
    """Evaluate a model trained with standard CE on a dataset."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for step, batch in enumerate(tqdm(loader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
            if mask_class != None:
                if eval:
                    outputs.logits[:, mask_class] = -np.inf
                else:
                    outputs.logits[:, mask_class] = -1e10
            predictions = outputs.logits.argmax(dim=-1)
            total += batch["labels"].size(0)
            correct += (predictions == batch["labels"].to(device)).sum().item()
    return correct / total


def prepare_llama(config, device):
    """Load LLama models from config."""
    bases = []
    peft_config = LoraConfig(task_type=config["peft_config"]["task_type"],
                                inference_mode=config["peft_config"]["inference_mode"],
                                r=config["peft_config"]["r"],
                                lora_alpha=config["peft_config"]["lora_alpha"],
                                lora_dropout=config["peft_config"]["lora_dropout"],
                                target_modules = config["peft_config"]["target_modules"],
                                )
    model_name_or_path = config['name']
    ptm_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path, return_dict=True, cache_dir=config['cachedir'], num_labels = 3)
    base_model = get_peft_model(ptm_model,peft_config)
    for idx, base_path in tqdm(enumerate(config['bases']), desc="Preparing Models", position=0, leave=True):
        if base_path.endswith('.pt'):
            base_model.load_state_dict(torch.load(base_path, map_location='cpu')) # Load fine-tuned model from local directory
        else: 
            base_model = PeftModel.from_pretrained(model = ptm_model, model_id = base_path) # Load model adapter from HF
        bases += [deepcopy(base_model)]
    # ptm_model_path = 'pretrained.pt' # load ptm_model from local directory
    # base_model.load_state_dict(torch.load(ptm_model_path, map_location='cpu'))
    ptm_model_path = 'hoffman-lab/KnOTS-Llama3_8B_lora_R16_pretrained_model'
    base_model = PeftModel.from_pretrained(model = ptm_model, model_id = ptm_model_path) # Load ptm_model from HF
    return {
        'bases': bases,
        'new': base_model
    }
