from copy import deepcopy

import torch
from tqdm.auto import tqdm

from merging_functions import ties_merging
from task_merger import TaskMerger

# Use the authors' built-in reshape fallback instead of adding an einops dependency.
einops = None


class MatrixPerLayerMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models,
            pretrained_model=pretrained_model,
            param_handler=param_handler,
            device=device,
            merge_config=merge_config
        )

        self.layer_names = self.get_layer_names(self.ftms_params[0].get_ft_parameters())
        self.ingredients = None
        self.cache = {}
        self.lmc = False  # True when merging in core (LoRA A/B) space

    def set_scaling_coeffs(self, scaling_coeffs):
        if isinstance(scaling_coeffs, float) or len(scaling_coeffs) == 1:
            self.scaling_coeffs = torch.tensor([scaling_coeffs] * len(self.ftms_params))
        else:
            self.scaling_coeffs = torch.tensor(scaling_coeffs)

    def _get_svd_cur_mixing_weights(self, delta_w_svd, delta_w_cur, merge_config):
        """Mixing weights for the SVD + CUR branches (paper Eq. 31-33).

        Energy mode: alpha = ||ΔW_SVD||_F / (||ΔW_SVD||_F + ||ΔW_CUR||_F).
        Fixed mode: use svd_cur_svd_weight / svd_cur_cur_weight from config.
        """
        mode = merge_config.get('svd_cur_weight_mode', 'fixed')
        if mode != 'energy':
            svd_weight = merge_config.get('svd_cur_svd_weight', 0.675)
            cur_weight = merge_config.get('svd_cur_cur_weight', 1.0 - svd_weight)
            weight_sum = svd_weight + cur_weight
            if weight_sum > 0:
                return svd_weight / weight_sum, cur_weight / weight_sum
            return 0.675, 0.325

        m_svd = torch.linalg.norm(delta_w_svd, ord='fro').item()
        m_cur = torch.linalg.norm(delta_w_cur, ord='fro').item()
        denom = m_svd + m_cur
        if denom <= 0:
            svd_weight = merge_config.get('svd_cur_svd_weight', 0.5)
            cur_weight = merge_config.get('svd_cur_cur_weight', 0.5)
            weight_sum = svd_weight + cur_weight
            if weight_sum > 0:
                return svd_weight / weight_sum, cur_weight / weight_sum
            return 0.5, 0.5
        return m_svd / denom, m_cur / denom

    def _apply_delta(self, new_sd, key, delta_w):
        """Add scaled delta_w to the parameter keyed by ``key``."""
        sd = new_sd.state_dict()
        if key in sd:
            # apply scaling_coeffs (same convention as core-space merging)
            # single coeff or first element of list (shared across tasks)
            if isinstance(self.scaling_coeffs, torch.Tensor):
                if len(self.scaling_coeffs) == 1:
                    scale = self.scaling_coeffs[0]
                else:
                    # multiple coeffs: use first (tasks usually share one scale)
                    scale = self.scaling_coeffs[0]
            else:
                scale = self.scaling_coeffs
            sd[key] = sd[key] + scale * delta_w.type_as(sd[key])
            new_sd.load_state_dict(sd)

    def _process_ties(self, tensor_list, topK=10):
        """Run TIES merging on a list of tensors."""
        original_shape = tensor_list[0].shape
        tensor_list = list(map(torch.flatten, tensor_list))
        merged_tv, rows_to_keep, mask = ties_merging(tensor_list, topK=topK)
        return merged_tv.reshape(original_shape)

    def get_knots_components(self, ftms_task_dirs):
        """Get the components in KnOTS space."""
        stack = torch.cat(ftms_task_dirs, dim=1)  # shape: (n, T*r*n)
        U, S, Vh = torch.linalg.svd(stack.to(torch.float64), full_matrices=False)
        # Retain only supported basis components.
        U = U[:, S > 1e-5].type(torch.float32)
        Vh = Vh[S > 1e-5].type(torch.float32)
        S = S[S > 1e-5].type(torch.float32)

        S[S <= 1e-5] = 0
        if einops is not None:
            Vs = einops.rearrange(Vh, 'Tr (b c) -> b Tr c', b=len(ftms_task_dirs))
        else:
            # Reshape manually if einops is unavailable.
            Tr, total = Vh.shape
            c = total // len(ftms_task_dirs)
            Vs = Vh.view(Tr, len(ftms_task_dirs), c).permute(1, 0, 2)
        return U, S, list(Vs)

    def get_cur_delta_w(self, ftms_task_dirs, num_cols=None, num_rows=None, selection_method='leverage_score', merge_config=None):
        """Compute the merged delta using CUR, without SVD.
        
        Args:
            ftms_task_dirs: per-task [m,n] matrices
            num_cols: Number of columns selected for CUR decomposition.
            num_rows: Number of rows selected for CUR decomposition.
            selection_method: CUR column/row selection method.
            merge_config: Merge configuration; use the specified merge_method, such as TSV.
        
        Returns:
            delta_w: Merged update matrix.
        """
        # Step 1: concatenate task updates.
        stack = torch.cat(ftms_task_dirs, dim=1)  # shape: (m, T*n)
        m, n_total = stack.shape
        n_per_task = n_total // len(ftms_task_dirs)
        
        # Step 2: apply CUR to the concatenated stack.
        if num_cols is None:
            num_cols = min(100, n_total // 2)
        if num_rows is None:
            num_rows = min(100, m // 2)
        
        num_cols = min(num_cols, n_total)
        num_rows = min(num_rows, m)
        
        # Select columns and rows based on their norms.
        col_norms = torch.norm(stack, dim=0, p=2)
        row_norms = torch.norm(stack, dim=1, p=2)
        _, col_indices = torch.topk(col_norms, num_cols, largest=True)
        _, row_indices = torch.topk(row_norms, num_rows, largest=True)
        col_indices = torch.sort(col_indices)[0]
        row_indices = torch.sort(row_indices)[0]
        
        # C = X(:, J) - selected columns [m, k].
        C = stack[:, col_indices]
        # R = X(I, :) - selected rows [k', n_total].
        R = stack[row_indices, :]
        
        # connector matrix U
        try:
            C_pinv = torch.linalg.pinv(C)
            R_pinv = torch.linalg.pinv(R)
            U = C_pinv @ stack @ R_pinv
        except:
            try:
                CTC_inv = torch.linalg.pinv(C.T @ C)
                RRT_inv = torch.linalg.pinv(R @ R.T)
                U = CTC_inv @ C.T @ stack @ R.T @ RRT_inv
            except:
                k, k_prime = C.shape[1], R.shape[0]
                U = torch.eye(min(k, k_prime), dtype=stack.dtype, device=stack.device)
                if k != k_prime:
                    U = torch.nn.functional.pad(U, (0, max(0, k_prime - k), 0, max(0, k - k_prime)))
        
        # Step 3: merge in full parameter space if TSV or CART is specified.
        # TSV/CART require full space rather than the reduced UR space.
        if merge_config and merge_config.get('merge_method') in ('tsv', 'cart'):
            # First apply TSV/CART merging in the original task-parameter space.
            merged_delta = self._merge_tensors(ftms_task_dirs, merge_config)
            # TSV/CART: skip CUR projection
            # CUR projection loses information; TSV/CART already optimize merging in full space.
            delta_w = merged_delta
        else:
            # Step 4: split R into blocks and compute UR(i) for each task.
            task_URs = []
            for i in range(len(ftms_task_dirs)):
                R_task = R[:, i * n_per_task:(i + 1) * n_per_task]
                UR_task = U @ R_task
                task_URs.append(UR_task)
            
            # Merge per-task UR using averaging or another method.
            merged_UR = self._merge_tensors(task_URs, merge_config)
            
            # Step 5: reconstruct the delta.
            delta_w = C @ merged_UR
        
        return delta_w.type_as(ftms_task_dirs[0])

    def _merge_tensors(self, tensors, merge_config):
        """Restrict the upstream dispatcher to the published TIES baseline."""
        if merge_config.get('merge_method') != 'ties':
            raise ValueError('CUR-merging supports TIES only')
        return self._process_ties(tensors, merge_config.get('topK', 10))

    def merge(self, merge_config):
        """Run only the upstream hybrid branch using the existing evaluator inputs."""
        if (merge_config.get('merge_space') != 'svdcur'
                or merge_config.get('merge_method') != 'ties'
                or merge_config.get('isotropize', False)):
            raise ValueError('CUR-merging requires svdcur/TIES without isotropization')
        self.lmc = False
        ftms_task_dirs = [ftm.get_ft_parameters() for ftm in self.ftms_params]
        all_keys = self.pretrained_model.state_dict().keys()
        new_sd = deepcopy(self.pretrained_model)
        avg_ranks = []
        weight_mode = merge_config.get('svd_cur_weight_mode', 'fixed')
        print(f"SVD+CUR mixing mode: {weight_mode}")

        svd_scaling_coeffs = merge_config.get('svd_cur_svd_scaling_coeffs', merge_config.get('scaling_coeffs', 0.6))
        cur_scaling_coeffs = merge_config.get('svd_cur_cur_scaling_coeffs', merge_config.get('scaling_coeffs', 0.6))
        layer_alphas = []

        for key in tqdm(all_keys, desc="Merging svdcur space"):
            key_base = key.replace('.base_layer', '')
            if key_base in ftms_task_dirs[0]:
                tensor_list = [deepcopy(ft_dir[key_base]).to(self.device) for ft_dir in ftms_task_dirs]
                
                # 1. SVD branch (knots space)
                if key_base in self.cache:
                    U, S, Vs = self.cache[key_base]
                else:
                    U, S, Vs = self.get_knots_components(tensor_list)
                    self.cache[key_base] = (U, S, Vs)
                
                # set SVD scaling
                original_scaling_coeffs = self.scaling_coeffs
                if isinstance(svd_scaling_coeffs, (int, float)):
                    self.scaling_coeffs = torch.tensor([svd_scaling_coeffs] * len(Vs))
                else:
                    self.scaling_coeffs = torch.tensor(svd_scaling_coeffs) if not isinstance(svd_scaling_coeffs, torch.Tensor) else svd_scaling_coeffs
                
                # SVD merge_config
                svd_merge_config = merge_config.copy()
                svd_merge_config.update({'scaling_coeffs': svd_scaling_coeffs})
                Vs_merged = self._merge_tensors(Vs, svd_merge_config)
                delta_w_svd = U @ torch.diag(S) @ Vs_merged
                if merge_config.get('isotropize', False):
                    delta_w_svd = isotropize_matrix(delta_w_svd)
                
                # restore scaling_coeffs
                self.scaling_coeffs = original_scaling_coeffs
                
                # 2. CUR branch
                if isinstance(cur_scaling_coeffs, (int, float)):
                    self.scaling_coeffs = torch.tensor([cur_scaling_coeffs] * len(tensor_list))
                else:
                    self.scaling_coeffs = torch.tensor(cur_scaling_coeffs) if not isinstance(cur_scaling_coeffs, torch.Tensor) else cur_scaling_coeffs
                
                # CUR merge_config
                cur_merge_config = merge_config.copy()
                cur_merge_config.update({'scaling_coeffs': cur_scaling_coeffs})
                delta_w_cur = self.get_cur_delta_w(
                    tensor_list,
                    num_cols=merge_config.get('cur_num_cols', None),
                    num_rows=merge_config.get('cur_num_rows', None),
                    selection_method=merge_config.get('cur_selection_method', 'leverage_score'),
                    merge_config=cur_merge_config
                )
                if merge_config.get('isotropize', False):
                    delta_w_cur = isotropize_matrix(delta_w_cur)
                
                # restore scaling_coeffs
                self.scaling_coeffs = original_scaling_coeffs

                svd_weight, cur_weight = self._get_svd_cur_mixing_weights(
                    delta_w_svd, delta_w_cur, merge_config
                )
                layer_alphas.append(svd_weight)
                mean_delta = delta_w_svd * svd_weight + delta_w_cur * cur_weight

                rank = torch.linalg.matrix_rank(mean_delta).item()
                avg_ranks.append(rank)
                self._apply_delta(new_sd, key, mean_delta)

        if layer_alphas:
            alpha_tensor = torch.tensor(layer_alphas)
            print(
                f"SVD branch weight (alpha): mean={alpha_tensor.mean():.4f}, "
                f"std={alpha_tensor.std():.4f}, min={alpha_tensor.min():.4f}, max={alpha_tensor.max():.4f}"
            )
        return new_sd


def isotropize_matrix(matrix):
    """Isotropize a matrix: replace all singular values with their mean."""
    U, S, V = torch.linalg.svd(matrix.to(torch.float64), full_matrices=False)
    S_iso = S.mean() * torch.ones_like(S)
    return (U @ torch.diag(S_iso) @ V).type_as(matrix)
