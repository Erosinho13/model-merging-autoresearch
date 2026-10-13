from collections import defaultdict, OrderedDict
from copy import deepcopy
import torch
from torch import nn
import pdb
from utils import get_merging_fn

class VectorOps(nn.Module):
    def directions_to_reps(self, directions):
        if isinstance(directions, list):
            return [self.directions_to_reps(direction) for direction in directions]
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in directions.items()]
        )
        
    def rep_to_state_dict(self, vector, state_dict, remove_keys=[]):
        if isinstance(vector, list) or len(vector.shape) == 2:
            return [self.rep_to_state_dict(v, state_dict, remove_keys) for v in vector]
        # create a reference dict to define the order of the vector
        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
        sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

        # create a shared state dict using the refence dict
        torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in sorted_reference_dict:
            for key in remove_keys:
                sorted_reference_dict[key] = sorted_reference_dict[
                    "transformer.shared.weight"
                ]
        return sorted_reference_dict
    
    def mask_to_state_dict(self, mask, state_dict, remove_keys=[]):
        if isinstance(mask, list):
            return [self.mask_to_state_dict(m, state_dict, remove_keys) for m in mask]
        return self.rep_to_state_dict(mask, state_dict, remove_keys)
    
    def forward(self, directions, merging_fn, merge_config):
        vectors = self.directions_to_reps(directions)
        merged_vector,rows_to_keep, topk_mask = merging_fn(vectors)
        mask_sd = self.rep_to_state_dict(topk_mask, directions[0])
        
        ties_mask = [dict() for _ in range(len(rows_to_keep))]
        for idx in range(len(rows_to_keep)):
            ties_mask[idx] = self.rep_to_state_dict(rows_to_keep[idx], directions[0])
        sd = self.rep_to_state_dict(merged_vector, directions[0])
        
        return sd, ties_mask


class TaskMerger(nn.Module):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__()
        
        self.device = device
        self.scaling_coeffs = torch.tensor([1.] * len(finetuned_models))
        self.param_handler = param_handler
        self.finetuned_models = finetuned_models
        self.ftms_params = [param_handler(ft_model) for ft_model in finetuned_models]
        self.pretrained_model = pretrained_model.cpu()
        self.pt_params = self.pretrained_model.state_dict()
        self.merge_config = merge_config

    def randbin(self, M, N, P):
        P = 1-P
        return torch.randint(2, size=(M, N), dtype=torch.float32).bernoulli(P)
    
    def apply_dare(self, ftms_params, p, dare_seed = 0):
        print("DARE seed: ", dare_seed)
        torch.manual_seed(dare_seed)
        finetuned_directions = []
        for ftm_params in ftms_params:
            direction_sd = {}
            for key, finetuned_val in ftm_params.items():
                direction_sd[key] = finetuned_val * self.randbin(finetuned_val.shape[0], finetuned_val.shape[1], p) * (1/(1-p))
                # pdb.set_trace()
            finetuned_directions += [OrderedDict(sorted(direction_sd.items()))]
        return finetuned_directions

    def get_task_directions(self, ptm_params, ftms_params):
        finetuned_directions = []
        for ftm_params in ftms_params:
            direction_sd = {}
            
            for key, finetuned_val in ftm_params.items():
                if key not in ptm_params:
                    ptm_val = torch.zeros_like(finetuned_val)
                else:
                    ptm_val = ptm_params[key]
                direction_sd[key] = finetuned_val - ptm_val
            finetuned_directions += [OrderedDict(sorted(direction_sd.items()))]
        return finetuned_directions
    
    def set_scaling_coeffs(self, scaling_coeffs):
        if isinstance(scaling_coeffs, float) or len(scaling_coeffs) == 1:
            self.scaling_coeffs = torch.tensor([scaling_coeffs] * len(self.ftms_params))
        else:
            self.scaling_coeffs = torch.tensor(scaling_coeffs)
        
    def get_layer_names(self, state_dict):
        layer_names = defaultdict(lambda: dict())
        for key in state_dict:
            if ('.weight' in key) or ('_weight' in key):
                strip_key = key.replace('.weight', '').replace('_weight', '')
                layer_names[strip_key]['weight'] = key
            elif ('.bias' in key) or ('_bias' in key):
                strip_key = key.replace('.bias', '').replace('_bias', '')
                layer_names[strip_key]['bias'] = key
            else:
                layer_names[key]['other'] = key + ':other'
        return layer_names
    
    def add_task_parameters(self, base_model, parameters, concat_across_output = True, scaling_coeffs=1.):
        if isinstance(parameters, list):
            return [self.add_task_parameters(
                deepcopy(base_model), 
                parameter,
                concat_across_output=concat_across_output, 
                scaling_coeffs=scaling_coeffs
            ) for parameter in parameters]
        sd = base_model.state_dict()
        for key, val in parameters.items():
            try:
                if (concat_across_output):
                    sd[key].add_(val.cpu() * scaling_coeffs)
                else:
                    sd[key].add_(val.T.cpu() * scaling_coeffs)
            except:
                pdb.set_trace()
        return base_model
    
    def directions_to_matrices(self, directions, reference_layer_names=None):
        if isinstance(directions, list):
            return [self.directions_to_matrices(direction, reference_layer_names) for direction in directions]
        
        if reference_layer_names is None:
            layer_names = self.get_layer_names(directions)
        else:
            layer_names = reference_layer_names

        matrices = {}
        for layer_name, parameter_names in layer_names.items():
            if 'other' in parameter_names:
                other_parameter = directions[parameter_names['other'].replace(':other', '')].to(torch.float32)
                # Ensure parameters are always two dimensional
                if len(other_parameter.shape) == 1: # e.g., class token, positional embeddings
                    other_parameter = other_parameter[None, :]
                elif len(other_parameter.shape) > 2: # e.g., patch embeddings
                    other_parameter = other_parameter.flatten(1)
                matrices[layer_name + ':other'] = other_parameter
            elif 'weight' in parameter_names:
                weight_name = parameter_names['weight']
                weight = directions[weight_name]
                if 'norm' in layer_name or 'ln' in layer_name:
                    weight = torch.diag(weight)
                matrices[layer_name] = weight.flatten(1)
                if 'bias' in parameter_names:
                    bias = directions[parameter_names['bias']]
                    matrices[layer_name] = torch.concat((matrices[layer_name], bias.reshape(-1, 1)), dim=1)
        return matrices
    
    def matrix_to_state_dict(self, matrix, state_dict, remove_keys=[]):
        if isinstance(matrix, list):
            return [self.matrix_to_state_dict(m, state_dict) for m in matrix]
        
        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
                
        layer_names = self.get_layer_names(reference_dict)
        merged_state_dict = {}
        for layer_name, value in matrix.items():
            try:
                parameter_types = layer_names[layer_name.replace(':other', '')]
                if 'other' in parameter_types:
                    name = parameter_types['other'].replace(':other', '')
                    merged_state_dict[name] = value.reshape(reference_dict[name].shape)
                else:
                    if 'bias' in parameter_types: 
                        bias_index = value.shape[1] - 1
                        value, bias = value[:, :bias_index], value[:, -1].flatten()
                        merged_state_dict[parameter_types['bias']] = bias
                    if 'norm' in layer_name or 'ln' in layer_name:
                        value = torch.diagonal(value)
                    name = parameter_types['weight']
                    merged_state_dict[name] = value.reshape(*(reference_dict[name].shape))
            except:
                pdb.set_trace()
                    
        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in merged_state_dict:
            for key in remove_keys:
                merged_state_dict[key] = merged_state_dict[
                    "transformer.shared.weight"
                ]
        return merged_state_dict
    
    def transform(self, *args, **kwargs):
        return


class VectorMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models, 
            pretrained_model=pretrained_model, 
            param_handler=param_handler, 
            device=device,
            merge_config=merge_config
        )
        
        self.representation_helper = VectorOps()
    
    def merge(self, merge_config={'merge_method': 'tv'}):
        print(merge_config['merge_method'])
        merging_fn = lambda x: get_merging_fn(merge_config['merge_method'])(
            x, **merge_config, weights=self.scaling_coeffs
        )

        ptm_reference_params = self.param_handler(self.pretrained_model).get_ft_parameters()
        ftms_relevant_params = [ftm.get_ft_parameters() for ftm in self.ftms_params]
        ftms_task_dirs = self.get_task_directions(ptm_reference_params, ftms_relevant_params)
        
        if merge_config.get('dare', False):
            ftms_task_dirs = self.apply_dare(
                ftms_task_dirs, merge_config['dare_pruning_coeffs'], merge_config['dare_seed']
            )
        
        merged_sd = self.representation_helper(ftms_task_dirs, merging_fn, merge_config)

        merged_base = deepcopy(self.pretrained_model)
        if len(merged_sd) == 2:
            merged_sd, mask = merged_sd
        merged_model = self.add_task_parameters(merged_base, merged_sd)
        
        return merged_model
