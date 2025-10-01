import functools, os
from typing import Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
import torch.distributed as dist

from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.train_spec import OptimizersBuilder
from torchtitan.tools.logging import logger

class DiLoCoAdapter:
    """DiLoCo optimizer adapter that manages global optimizer and pseudo-gradient computation."""

    def __init__(
        self,
        model_parts: List[nn.Module],
        local_optimizer: Optimizer,
        parallel_dims: ParallelDims,
        num_local_steps: int = 1,
        global_lr: float = 0.7,
        momentum: float = 0.9,
        nesterov: bool = True,
        dump_folder: str = "",
    ):
        # Store references to model parts and parallel_dims
        self._model_parts = model_parts
        self._parallel_dims = parallel_dims

        self._worker_model_params = []
        for model in self._model_parts:
            # Handle distributed and regular parameters
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self._worker_model_params.append(p)

        # Store local SGD specific fields.
        self._num_local_steps = num_local_steps
        self._local_step_counter = 0
        self._global_step_counter = 0
        
        # Initialize L2 norm variables with default None values
        self._calc_metrics = False
        self._calc_debug_metrics = False
        self._local_params_l2_norm = None
        self._local_gradient_l2_norm = None
        self._local_gradient_percentiles = {}
        self._pseudo_gradient_l2_norm = None
        self._pseudo_gradient_cosine_sim = None
        self._global_params_l2_norm = None
        self._pseudo_gradients_buffer = None
        
        # Cache for parameter count
        self._params_numel = None

        # Create global optimizer
        self._global_model_params = [
            p.clone().detach().requires_grad_(False)
            for p in self._worker_model_params
        ]
        global_model_l2_norm = self._calc_l2_norm(self._global_model_params)
        worker_model_params_l2_norm = self._calc_l2_norm(self._worker_model_params)
        logger.info(f"Creating DiloCoAdapter with {len(self._global_model_params)} parameters, {global_model_l2_norm=}, {worker_model_params_l2_norm= }")
        self._global_optimizer = optim.SGD(
            self._global_model_params,
            lr=global_lr,
            momentum=momentum,
            nesterov=nesterov,
        )
        self._local_optimizer = local_optimizer
        
        self.dump_folder = os.path.join(dump_folder, "diloco")
        os.makedirs(self.dump_folder, exist_ok=True)


    @torch.no_grad()
    def _prepare_flat_buffers_from_model_params(
        self,
        params: list[torch.Tensor],
        copy: bool = False,
        device: torch.device | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """
        Prepares the flat buffer based on the worker model parameters.

        Args:
            params (list[torch.Tensor]): List of worker model parameters.
            copy (bool): Whether to copy the parameters into flat buffer.
            device (torch.device | None): Device for flat buffer (default: params device).
        Returns:
            tuple[list[torch.Tensor], torch.Tensor]: Tuple containing the list of
            pseudo-gradient buffers that are views into the flat buffer.
        """

        out_params = [p.clone().detach().requires_grad_(False) for p in params]
        if copy:
            for out_param, param in zip(out_params, params):
                out_param.copy_(param)
        return out_params, None

    @torch.no_grad()
    def _compute_pseudo_gradients(
        self,
        pseudo_gradients: list[torch.Tensor],
        global_model_params: list[torch.Tensor],
        worker_model_params: list[torch.Tensor],
    ) -> None:
        """
        Computes the pseudo-gradients by subtracting the worker model parameters from
        the global model parameters.
        Args:
            global_model_params (list[torch.Tensor]): List of global model parameters.
            worker_model_params (list[torch.Tensor]): List of worker model parameters.
            pseudo_gradients (list[torch.Tensor]): Where computed pseudo-gradients
                must be stored.
        """
        for (pseudo_grad, global_param, worker_param) in zip(pseudo_gradients, global_model_params, worker_model_params):
            pseudo_grad.copy_(global_param)
            pseudo_grad.sub_(worker_param)

    @torch.no_grad()
    def _set_pseudo_gradients(
        self, params: List[torch.Tensor], pseudo_gradients: List[torch.Tensor]
    ) -> None:
        """Set gradients of parameters to pseudo-gradients, handling distributed tensors."""
        for param, pseudo_gradient in zip(params, pseudo_gradients):
            assert hasattr(param, "grad"), f"Parameter must have gradient {param.shape}"
            if hasattr(param, "grad"):
                param.grad = pseudo_gradient

    @torch.no_grad()
    def _calc_l2_norm(self, tensors: List[torch.Tensor]) -> float:
        """Calculate distributed L2 norm over dp_cp mesh."""

        if len(tensors) == 0:
            return 0.0

        l2_norm_squared = sum(getattr(t, "_local_tensor", t).detach().pow(2).sum().item() for t in tensors)
        # For distributed case, need to aggregate across parallel dimensions
        if self._parallel_dims.dp_cp_enabled:
            l2_norm_squared = dist_utils.dist_sum(
                torch.tensor(l2_norm_squared, device=tensors[0].device, dtype=torch.float32),
                self._parallel_dims.world_mesh["dp_cp"]
            )


        total_numel = sum(getattr(p, "_local_tensor", p).detach().numel() for p in self._worker_model_params)
        if self._parallel_dims.dp_cp_enabled:
            if not self._params_numel:
                total_numel = dist_utils.dist_sum(
                    torch.tensor(total_numel, device=tensors[0].device, dtype=torch.float32),
                    self._parallel_dims.world_mesh["dp_cp"]
                )
                self._params_numel = total_numel
            else:
                total_numel = self._params_numel

        import math
        return math.sqrt(l2_norm_squared / max(1.0, total_numel))


    @torch.no_grad()
    def post_step_hook(self, optimizer, args, kwargs):
        """
        Post optimization hook that implements DiLoCo logic.
        Adapted from nested_optimizer.py step() method.
        """
        self._local_step_counter += 1
        trigger_global_step = self._local_step_counter > 0 and (
            self._local_step_counter % self._num_local_steps == 0
        )

        if self._calc_metrics:
            # Calculate L2 norms using member function
            self._local_params_l2_norm = self._calc_l2_norm(self._worker_model_params)
            self._local_gradient_l2_norm = self._calc_l2_norm([(p.grad._local_tensor if hasattr(p, "_local_tensor") else p.grad) for p in self._worker_model_params if p.grad is not None])

            self._global_params_l2_norm = self._calc_l2_norm(self._global_model_params)

            # Step 1: Create buffer for pseudo-gradients
            pseudo_gradients, _ = self._prepare_flat_buffers_from_model_params(
                params=self._global_model_params
            )

            # Step 2: Compute pseudo-gradients and copy global params into worker params
            self._compute_pseudo_gradients(
                pseudo_gradients=pseudo_gradients,
                worker_model_params=self._worker_model_params, # [(p._local_tensor if hasattr(p, "_local_tensor") else p).detach() for p in self._worker_model_params],
                global_model_params=self._global_model_params,
            )
            self._pseudo_gradient_l2_norm = self._calc_l2_norm(pseudo_gradients)

        if self._calc_debug_metrics:
            def _flatten_(tensors: List[torch.Tensor]) -> torch.Tensor:
                return torch.cat([t.view(-1) for t in tensors], dim=0)

            def _calc_percentiles(tensor: torch.Tensor, N: int = 1_000_000) -> dict[str, float]:
                percentiles = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
                indices = torch.randint(0, tensor.numel(), (N,), device=tensor.device)
                samples = torch.index_select(tensor, 0, indices)
                grad_percentiles = torch.quantile(samples.abs(), torch.tensor(percentiles, device=tensor.device))
                return {f"{int(p*1000)/10:.1f}%": v.item() for p, v in zip(percentiles, grad_percentiles)}

            self._local_gradient_percentiles = _calc_percentiles(_flatten_([getattr(p, "_local_tensor", p).grad for p in self._worker_model_params if p.grad is not None]))
            pseudo_gradients_buffer = _flatten_([getattr(g, "_local_tensor", g) - getattr(p, "_local_tensor", p) for p, g in zip(self._global_model_params, self._worker_model_params)])
            if self._pseudo_gradients_buffer is not None:
                # Calculate cosine similarity between previous and current pseudo-gradients
                self._pseudo_gradient_cosine_sim = torch.nn.functional.cosine_similarity(pseudo_gradients_buffer - self._pseudo_gradients_buffer, self._pseudo_gradients_buffer, dim=0).item()
            else:
                self._pseudo_gradient_cosine_sim = 0.0

            self._pseudo_gradients_buffer = pseudo_gradients_buffer

        if trigger_global_step:
            self._global_step_counter += 1

            # Step 1: Create buffer for pseudo-gradients
            pseudo_gradients, _ = self._prepare_flat_buffers_from_model_params(
                params=self._global_model_params
            )

            # Step 2: Compute pseudo-gradients and copy global params into worker params
            self._compute_pseudo_gradients(
                pseudo_gradients=pseudo_gradients,
                worker_model_params=self._worker_model_params, # [(p._local_tensor if hasattr(p, "_local_tensor") else p).detach() for p in self._worker_model_params],
                global_model_params=self._global_model_params,
            )

            # Step 3: Set pseudo-gradients as gradients of the parameters
            self._set_pseudo_gradients(
                params=self._global_model_params,
                pseudo_gradients=pseudo_gradients,
            )
            
            # Step 4: Perform global optimizer step
            self._global_optimizer.step()

            # Step 5: Update worker model parameters copy
            for local_p, global_p in zip(self._worker_model_params, self._global_model_params):
                local_p.data.lerp_(global_p, 1.0)


            self._global_optimizer.zero_grad()
            self._pseudo_gradients_buffer = None

        if self._calc_metrics:
            local_optimizer_exp_avg_l2_norm = self._calc_l2_norm([self._local_optimizer.state[p]["exp_avg"] for p in self._local_optimizer.param_groups[0]["params"] if len(self._local_optimizer.state[p])])
            local_optimizer_exp_avg_sq_l2_norm = self._calc_l2_norm([self._local_optimizer.state[p]["exp_avg_sq"] for p in self._local_optimizer.param_groups[0]["params"] if len(self._local_optimizer.state[p])])
            global_optimizer_mom_l2_norm = self._calc_l2_norm([self._global_optimizer.state[p]["momentum_buffer"] for p in self._global_optimizer.param_groups[0]["params"] if len(self._global_optimizer.state[p])])

            import json
            logger.info("DiLoCo Metrics: " + json.dumps({
                "local_step_counter": self._local_step_counter,
                "global_step_counter": self._global_step_counter,
                "local_params_l2_norm": self._local_params_l2_norm,
                "local_gradient_l2_norm": self._local_gradient_l2_norm,
                "pseudo_gradient_l2_norm": self._pseudo_gradient_l2_norm,
                "pseudo_gradient_cosine_sim": self._pseudo_gradient_cosine_sim,
                "global_params_l2_norm": self._global_params_l2_norm,
                "params_numel": self._params_numel,
                **self._local_gradient_percentiles,
                **{
                    "local_optimizer_param_groups": {k : v for k,v in self._local_optimizer.param_groups[0].items() if k != "params"}, 
                    "local_optimizer_exp_avg_l2_norm": local_optimizer_exp_avg_l2_norm, 
                    "local_optimizer_exp_avg_sq_l2_norm": local_optimizer_exp_avg_sq_l2_norm
                },
                **{
                    "global_optimizer_param_groups": {k : v for k,v in self._global_optimizer.param_groups[0].items() if k != "params"}, 
                    "global_optimizer_mom_l2_norm": global_optimizer_mom_l2_norm, 
                },
            }))


    def state_dict(self) -> dict[str, Any]: 
        """
        Returns state dict containing both local and global optimizer states,
        as well as step counters and global model parameters.
        """
        state_dict = {
            "local_step_counter": self._local_step_counter,
            "global_step_counter": self._global_step_counter,
            "global_optimizer": self._global_optimizer.state_dict(),
            "global_model_params": [p.data for p in self._global_model_params],
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Loads state dict to restore local and global optimizer states,
        step counters, and global model parameters.
        """
        self._local_step_counter = state_dict["local_step_counter"]
        self._global_step_counter = state_dict["global_step_counter"]
        self._global_optimizer.load_state_dict(state_dict["global_optimizer"])
        for p, p_data in zip(self._global_model_params, state_dict["global_model_params"]):
            p.data.copy_(p_data)

    def post_state_dict_hook(self, optimizer: Optimizer, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Post state dict hook to include DiLoCo adapter state in the optimizer's state dict.
        This hook is called after the optimizer's state_dict() method.
        """
        global_model_l2_norm = self._calc_l2_norm(self._global_model_params)
        worker_model_params_l2_norm = self._calc_l2_norm(self._worker_model_params)
        local_optimizer_exp_avg_l2_norm = self._calc_l2_norm([self._local_optimizer.state[p]["exp_avg"] for p in self._local_optimizer.param_groups[0]["params"] if len(self._local_optimizer.state[p])])
        local_optimizer_exp_avg_sq_l2_norm = self._calc_l2_norm([self._local_optimizer.state[p]["exp_avg_sq"] for p in self._local_optimizer.param_groups[0]["params"] if len(self._local_optimizer.state[p])])
        global_optimizer_mom_l2_norm = self._calc_l2_norm([self._global_optimizer.state[p]["momentum_buffer"] for p in self._global_optimizer.param_groups[0]["params"] if len(self._global_optimizer.state[p])])

        state_dict[f"diloco_adapter"] = self.state_dict()

        # from tools.debugging.pdb import vscode_debug_for_rank
        # vscode_debug_for_rank()

        diloco_adapter_state_dict = state_dict[f"diloco_adapter"]
        logger.info(f"Saving DiLoCo adapter state to optimizer state for step {self._local_step_counter}: {list(diloco_adapter_state_dict.keys())}, {global_model_l2_norm=}, {worker_model_params_l2_norm=} {local_optimizer_exp_avg_l2_norm=} {local_optimizer_exp_avg_sq_l2_norm=} {local_optimizer_exp_avg_sq_l2_norm=} {global_optimizer_mom_l2_norm=}")

        return state_dict

    def pre_load_state_dict_hook(self, optimizer: Optimizer, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Pre load state dict hook to extract DiLoCo adapter state from the optimizer's state dict.
        This hook is called before the optimizer's load_state_dict() method.
        """

        # from tools.debugging.pdb import vscode_debug_for_rank
        # vscode_debug_for_rank()

        assert f"diloco_adapter" in state_dict, f"DiLoCo adapter state not found in state dict: {list(state_dict.keys())}"
        diloco_adapter_state_dict = state_dict.pop(f"diloco_adapter")
        self.load_state_dict(diloco_adapter_state_dict)

        global_model_l2_norm = self._calc_l2_norm(self._global_model_params)
        worker_model_params_l2_norm = self._calc_l2_norm(self._worker_model_params)
        logger.info(f"Loading DiLoCo adapter state from optimizer state for step {diloco_adapter_state_dict['local_step_counter']}, global_step {diloco_adapter_state_dict['global_step_counter']}, {global_model_l2_norm=} {worker_model_params_l2_norm=}")
        return state_dict

    def post_load_state_dict_hook(self, optimizer: Optimizer) -> None:
        """
        Post load state dict hook to extract DiLoCo adapter state from the optimizer's state dict.
        This hook is called after the optimizer's load_state_dict() method.
        """

        # from tools.debugging.pdb import vscode_debug_for_rank
        # vscode_debug_for_rank()

        local_optimizer_exp_avg_l2_norm = self._calc_l2_norm([self._local_optimizer.state[p]["exp_avg"] for p in self._local_optimizer.param_groups[0]["params"] if len(self._local_optimizer.state[p])])
        local_optimizer_exp_avg_sq_l2_norm = self._calc_l2_norm([self._local_optimizer.state[p]["exp_avg_sq"] for p in self._local_optimizer.param_groups[0]["params"] if len(self._local_optimizer.state[p])])
        global_optimizer_mom_l2_norm = self._calc_l2_norm([self._global_optimizer.state[p]["momentum_buffer"] for p in self._global_optimizer.param_groups[0]["params"] if len(self._global_optimizer.state[p])])

        logger.info(f"Post Load DiLoCo adapter state from optimizer state for step {self._local_step_counter}, global_step {self._global_step_counter}, {local_optimizer_exp_avg_l2_norm=} {local_optimizer_exp_avg_sq_l2_norm=} {global_optimizer_mom_l2_norm=}")

def build_diloco_optimizers(
    build_local_optimizers_fn: OptimizersBuilder,
    model_parts: List[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    """Build DiLoCo optimizer by wrapping the regular optimizer with DiLoCo adapter."""

    logger.info(f"Building DiLoCo Optimizers: {optimizer_config.__dict__=}")
    # Build regular optimizers first
    optimizers = build_local_optimizers_fn(
        model_parts, optimizer_config, parallel_dims, ft_manager
    )

    # Create DiLoCo adapter with default values (these could be made configurable)
    diloco_adapter = DiLoCoAdapter(
        model_parts, 
        optimizers.optimizers[0], 
        parallel_dims,
        num_local_steps=optimizer_config.num_local_steps, 
        global_lr=optimizer_config.global_lr, 
        momentum=optimizer_config.global_momentum, 
        nesterov=optimizer_config.global_nesterov,
        dump_folder=optimizer_config.dump_folder,
    )

    # Register DiLoCo hooks with the optimizers container
    optimizers.register_step_post_hook(diloco_adapter.post_step_hook)
    optimizers.register_state_dict_post_hook(diloco_adapter.post_state_dict_hook)
    optimizers.register_load_state_dict_pre_hook(diloco_adapter.pre_load_state_dict_hook)
    optimizers.register_load_state_dict_post_hook(diloco_adapter.post_load_state_dict_hook)

    return optimizers
