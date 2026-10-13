"""Private reference alignment; not exposed to the research agent."""
from copy import deepcopy
from collections import defaultdict
import torch
from tqdm.auto import tqdm
from task_merger import TaskMerger, VectorOps, VectorMerger
from utils import get_mask_fn, get_merging_fn
from masking_ops import masked_merge


class SVDMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models, 
            pretrained_model=pretrained_model, 
            param_handler=param_handler, 
            device=device,
            merge_config=merge_config
        )
        
        self.layer_names = self.get_layer_names(self.ftms_params[0].get_ft_parameters())
        self.representation_helper = VectorOps()
        self.ingredients = None
            
    def variable_extend_dim(self, elements, op_dim):
        if isinstance(elements, list):
            return [self.variable_extend_dim(element, op_dim) for element in elements]
        while len(elements.shape) < (op_dim+1):
            elements = elements.unsqueeze(-1)
        return elements
    
    def dict_of_concat_matrices(self, list_of_dictmatrices, dim=0, concat_across_output = True):
        dict2matrix_stack = defaultdict(lambda: list())
        for dict2matrix in list_of_dictmatrices:
            for key, val in dict2matrix.items():
                if(concat_across_output == True):
                    dict2matrix_stack[key] += [val.to(self.device)]
                else: 
                    dict2matrix_stack[key] += [val.T.to(self.device)]
                
        for key, list_of_vals in dict2matrix_stack.items():
            # Extend dim as necessary
            list_of_vals = self.variable_extend_dim(list_of_vals, op_dim=dim)
            dict2matrix_stack[key] = torch.concat(list_of_vals, dim=dim)
        return dict2matrix_stack
    
    def reconstruct_merged_sd(self, U_sd, sV_sd):
        if isinstance(sV_sd, list):
            if isinstance(U_sd, list):
                return [self.reconstruct_merged_sd(U, sV) for U, sV in zip(U_sd, sV_sd)]
            return [self.reconstruct_merged_sd(U_sd, sV) for sV in sV_sd]
        sd = {}
        for key, U in U_sd.items():
            sd[key] = (U @ sV_sd[key]).to(torch.float32)
        return sd
        
    def apply_svd(self, ft_params, concat_across_output = True):
        UsV_dict = {}
        basis_dict = {} # basis for reconstruction
        s_compositions_dict = [dict() for _ in range(len(ft_params))]
        V_compositions_dict = [dict() for _ in range(len(ft_params))] # basis composition information per task
        
        print(f'Calculating SVD over {len(ft_params)} models. S > 1e-5')
        concated_ft_params = self.dict_of_concat_matrices(ft_params, dim=1, concat_across_output = concat_across_output)
        for key, val in tqdm(concated_ft_params.items(), desc='Obtaining SVDs...'):
            U, s, V = torch.linalg.svd(val.to(torch.float64), full_matrices=False)
            # Keep only supported basis components
            U = U[:, s > 1e-5].type(torch.float32)
            V = V[s > 1e-5].type(torch.float32)
            s = s[s > 1e-5].type(torch.float32)
            UsV_dict[key] = {'U': deepcopy(U), 's':deepcopy(s), 'V':deepcopy(V) }
            # Set all s to be the same scale
            s[s <= 1e-5] = 0
            cat_hidden_dim = V.shape[1] // len(ft_params)

            basis_dict[key] = U.cpu()
            sV_concat = V
            Vs = list(torch.split(sV_concat, cat_hidden_dim, dim=1))
            for idx, V in enumerate(Vs):
                V = torch.diag(s) @ V # Simple and safe for all merging methods we use.
                s_model = s / s

                s_compositions_dict[idx][key] = s_model.cpu()
                V_compositions_dict[idx][key] = V.cpu()
        return basis_dict, s_compositions_dict, V_compositions_dict, UsV_dict
    
    def apply_Ss_on_Vs(self, task_Vs, task_Ss):
        task_sVs = [dict() for i in range(len(task_Vs))]
        for idx, (Vs, Ss) in enumerate(zip(task_Vs, task_Ss)):
            for key, V in Vs.items():
                if len(Ss[key].shape) == 2:
                    task_sVs[idx][key] = Ss[key] @ V
                else:
                    task_sVs[idx][key] = torch.diag(Ss[key]) @ V
        return task_sVs
    
    def remove_others(self, ftms_mats):
        other_mats = [dict() for i in range(len(ftms_mats))]
        transform_mats = [dict() for i in range(len(ftms_mats))]
        
        for m_idx, ftm_mats in enumerate(ftms_mats):
            for key, val in ftm_mats.items():
                if ':other' in key:
                    other_mats[m_idx][key] = val
                elif 'modules_to_save' in key:
                    other_mats[m_idx][key] = val
                else:
                    transform_mats[m_idx][key] = val
        print(f'Len other: {len(other_mats[0])}| len: transform: {len(transform_mats[0])}')
        return other_mats, transform_mats
    
    def add_others(self, ftms_mats, ftms_others):
        if isinstance(ftms_mats, list):
            return [self.add_others(ftms_mat, ftms_other) for ftms_mat, ftms_other in zip(ftms_mats, ftms_others)]
        
        for key, val in ftms_others.items():
            ftms_mats[key] = val
        return ftms_mats
    
    def transform(self, merge_config):
        # Setup parameters
        ptm_reference_params = deepcopy(self.param_handler(self.pretrained_model).get_ft_parameters())
        ftms_relevant_params = [ftm.get_ft_parameters() for ftm in self.ftms_params]
        ftms_task_dirs = self.get_task_directions(ptm_reference_params, ftms_relevant_params)
        
        ftms_task_mats = self.directions_to_matrices(ftms_task_dirs)
        ftms_others, ftms_mats = self.remove_others(ftms_task_mats)

        U, task_Ss, task_sVs, UsV_dict = self.apply_svd(
            ftms_mats,
            concat_across_output = merge_config.get('concat_across_output', True),
        )
            
        self.ingredients = {
            'ftms_relevant_params': ftms_relevant_params,
            'ftms_others': ftms_others,
            'ptm_reference_params': ptm_reference_params,
            'U': U,
            'task_Ss': task_Ss,
            'task_sVs': task_sVs,
            'UsV_dict': UsV_dict,
        }
        
        if merge_config.get('ingredients_path') is not None:
            torch.save(self.ingredients, merge_config['ingredients_path'])
    
    def merge(self, merge_config):
        if merge_config.get('ingredients_path') is not None:
            ingredients = torch.load(merge_config['ingredients_path'])
        else:
            ingredients = deepcopy(self.ingredients)
            
        ftms_others = ingredients['ftms_others']
        ptm_reference_params = ingredients['ptm_reference_params']
        U = ingredients['U']
        task_Ss = ingredients['task_Ss']
        task_sVs = ingredients['task_sVs']
        
        if merge_config.get('dare', False):
            print("Applying DARE")
            task_sVs = self.apply_dare(
                task_sVs, merge_config['dare_pruning_coeffs'], merge_config['dare_seed']
            )
        
        representations = self.representation_helper.directions_to_reps(task_sVs)
        ftms_reps = representations
        
        mask_fn = get_mask_fn(merge_config['merge_method'])
        masks = mask_fn(ftms_reps, **merge_config)
        ftms_reps = torch.vstack(ftms_reps).clone()
        masked_sVs = ftms_reps * masks
        pre_merge_sVs_dict = self.representation_helper.rep_to_state_dict(masked_sVs, task_sVs[0])
        rescaled_Vs = self.apply_Ss_on_Vs(pre_merge_sVs_dict, task_Ss)
        
        rescaled_Vs = torch.stack(self.representation_helper.directions_to_reps(rescaled_Vs), dim=0)
        merged_sV_ = masked_merge(
            merge_func=merge_config.get('merging_type'), vectors=rescaled_Vs, weights=self.scaling_coeffs
        )
        merged_sV_sd = self.representation_helper.rep_to_state_dict(merged_sV_, task_sVs[0])
        
        merged_sd = self.reconstruct_merged_sd(U, merged_sV_sd)
        
        merging_fn = lambda x: get_merging_fn(merge_config['merge_method'])(
            x, **merge_config, weights=self.scaling_coeffs
        )
        if merge_config.get('merge_other_params', False):
            merged_others,_ = self.representation_helper(ftms_others,  merging_fn=merging_fn)
            merged_sd = self.add_others(merged_sd, merged_others)
        
        merged_sd = self.matrix_to_state_dict(merged_sd, ptm_reference_params)
        # Add merged sd to the ptm
        merged_base = deepcopy(self.pretrained_model)
        merged_model = self.add_task_parameters(merged_base, merged_sd,  concat_across_output = merge_config.get('concat_across_output', True))
        return merged_model


def get_merge_handler(rep_type):
    if rep_type == 'svd-vector':
        return SVDMerger
    elif rep_type == 'vector':
        return VectorMerger
