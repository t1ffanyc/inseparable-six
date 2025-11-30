"""
Signal Strength Allocation Strategies for Layer-Adaptive DP

Computes per-layer clipping bounds {C_l} where:
- C_l = global_clipping_bound * alpha_l (per-layer clipping bound)
- Constraint: sqrt(sum(C_l^2)) = global_clipping_bound

Each strategy implements a different approach to allocating the privacy budget
across layers based on "signal strength" heuristics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple, Callable
from collections import OrderedDict
from enum import Enum
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from .config import PrivacyDataSource
from .metrics import AllocationMetrics


class AllocationStrategy(ABC):
    """
    Abstract base class for computing per-layer clipping bound allocation.
    
    Subclasses implement different strategies for distributing the global
    clipping bound across layers based on various heuristics.
    
    Returns allocation ratios {alpha_l} where:
    - C_l = global_clipping_bound * alpha_l (per-layer clipping bound)
    - Constraint: sqrt(sum(alpha_l^2)) = 1.0 (so sqrt(sum(C_l^2)) = global_clipping_bound)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for the strategy"""
        pass
    
    @abstractmethod
    def compute_allocation(
        self,
        model: nn.Module,
        global_clipping_bound: float,
        data_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """
        Compute per-layer clipping bounds.
        
        Args:
            model: The model (potentially with LoRA adapters)
            global_clipping_bound: Global clipping bound C_global
            data_loader: Optional dataloader for gradient-based estimation
            device: Device to run computations on
            
        Returns:
            Dictionary mapping layer_group_name -> clipping_bound_l
            where sqrt(sum(clipping_bound_l^2)) = global_clipping_bound
        """
        pass
    
    def get_metrics(self) -> Optional[AllocationMetrics]:
        """
        Get allocation metrics after compute_allocation has been called.
        
        Returns:
            AllocationMetrics if available, None otherwise
        """
        return getattr(self, '_metrics', None)
    
    def get_parameter_groups(
        self, 
        model: nn.Module,
        block_pattern: str = ".h."  # Default for GPT-2 style transformers
    ) -> Dict[str, List[nn.Parameter]]:
        """
        Group trainable parameters by transformer block.
        
        For LoRA models: Groups LoRA parameters per block.
        For full models: Groups all trainable parameters per block.
        
        Args:
            model: The model to analyze
            block_pattern: Pattern to identify transformer blocks (e.g., ".h." for GPT-2)
            
        Returns:
            OrderedDict mapping group_name -> list of parameters
        """
        param_groups = OrderedDict()
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Extract block identifier from parameter name
            group_name = self._extract_group_name(name, block_pattern)
            
            if group_name not in param_groups:
                param_groups[group_name] = []
            param_groups[group_name].append(param)
        
        return param_groups
    
    def _extract_group_name(self, param_name: str, block_pattern: str) -> str:
        """
        Extract the group name from a parameter name.
        
        Examples:
            "base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight" -> "block_0"
            "transformer.h.3.mlp.c_fc.weight" -> "block_3"
            "model.embed_tokens.weight" -> "other"
        """
        if block_pattern in param_name:
            parts = param_name.split(block_pattern)
            if len(parts) > 1:
                block_num = parts[1].split(".")[0]
                try:
                    int(block_num)  # Verify it's a valid number
                    return f"block_{block_num}"
                except ValueError:
                    pass
        return "other"
    
    def _normalize_allocations(
        self, 
        raw_allocations: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize allocation ratios to satisfy: sqrt(sum(alpha_l^2)) = 1.0
        
        Args:
            raw_allocations: Unnormalized allocation values (must be non-negative)
            
        Returns:
            Normalized allocations where sqrt(sum(alpha_l^2)) = 1.0
        """
        # Ensure non-negative
        raw_allocations = {k: max(v, 1e-8) for k, v in raw_allocations.items()}
        
        # Compute L2 norm
        sum_squares = sum(v**2 for v in raw_allocations.values())
        norm_factor = np.sqrt(sum_squares)
        
        if norm_factor == 0:
            # Fallback to uniform if all zeros
            n = len(raw_allocations)
            return {k: 1.0 / np.sqrt(n) for k in raw_allocations.keys()}
        
        # Normalize
        return {k: v / norm_factor for k, v in raw_allocations.items()}
    
    def _to_clipping_bounds(
        self, 
        normalized_allocations: Dict[str, float],
        global_clipping_bound: float
    ) -> Dict[str, float]:
        """
        Convert normalized allocation ratios to actual clipping bounds.
        
        C_l = global_clipping_bound * alpha_l
        """
        return {
            k: global_clipping_bound * alpha 
            for k, alpha in normalized_allocations.items()
        }
    
    def _verify_constraint(
        self, 
        clipping_bounds: Dict[str, float], 
        global_clipping_bound: float,
        tolerance: float = 1e-3
    ) -> Tuple[bool, float]:
        """
        Verify that sqrt(sum(C_l^2)) = global_clipping_bound
        
        Returns:
            Tuple of (is_valid, computed_value)
        """
        sum_squares = sum(c_l**2 for c_l in clipping_bounds.values())
        computed = np.sqrt(sum_squares)
        is_valid = np.isclose(computed, global_clipping_bound, rtol=tolerance)
        return is_valid, computed


class UniformStrategy(AllocationStrategy):
    """
    Uniform allocation across all parameter groups.
    
    Each layer group receives equal allocation: alpha_l = 1/sqrt(n_groups)
    This is equivalent to standard DP-SGD behavior.
    """
    
    @property
    def name(self) -> str:
        return "uniform"
    
    def compute_allocation(
        self,
        model: nn.Module,
        global_clipping_bound: float,
        data_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ) -> Dict[str, float]:
        
        param_groups = self.get_parameter_groups(model)
        n_groups = len(param_groups)
        
        # Uniform: alpha_l = 1/sqrt(n) for each group
        alpha = 1.0 / np.sqrt(n_groups)
        raw_allocations = {name: alpha for name in param_groups.keys()}
        
        normalized = self._normalize_allocations(raw_allocations)
        clipping_bounds = self._to_clipping_bounds(normalized, global_clipping_bound)
        
        # Store metrics
        is_valid, constraint_value = self._verify_constraint(
            clipping_bounds, global_clipping_bound
        )
        self._metrics = AllocationMetrics(
            strategy_name=self.name,
            layer_clipping_bounds=clipping_bounds,
            allocation_ratios=normalized,
            global_clipping_bound=global_clipping_bound,
            constraint_value=constraint_value,
        )
        
        return clipping_bounds


class GradientNormStrategy(AllocationStrategy):
    """
    Signal-proportional allocation based on observed gradient magnitudes.
    
    Layers with larger gradients receive larger clipping bounds (more "budget").
    Hypothesis: Layers with stronger gradient signals benefit more from looser clipping.
    
    Privacy considerations:
    - If using public_data: No privacy cost for gradient estimation
    - If using private_data: Gradient estimation on training data may leak information.
      The samples used for estimation should be accounted for in privacy analysis.
    """
    
    def __init__(
        self,
        num_estimation_samples: int = 100,
        data_source: PrivacyDataSource = PrivacyDataSource.PRIVATE,
        public_data_loader: Optional[DataLoader] = None,
        estimation_privacy_epsilon: float = 0.1,  # If using private data
        verbose: bool = True,
    ):
        """
        Args:
            num_estimation_samples: Number of samples for gradient estimation
            data_source: Whether to use public or private data for estimation
            public_data_loader: DataLoader for public estimation data (if data_source=PUBLIC)
            estimation_privacy_epsilon: Privacy budget for estimation (if data_source=PRIVATE)
            verbose: Whether to print progress information
        """
        self.num_estimation_samples = num_estimation_samples
        self.data_source = data_source
        self.public_data_loader = public_data_loader
        self.estimation_privacy_epsilon = estimation_privacy_epsilon
        self.verbose = verbose
        
        # Validate configuration
        if data_source == PrivacyDataSource.PUBLIC and public_data_loader is None:
            raise ValueError(
                "public_data_loader must be provided when data_source is PUBLIC"
            )
    
    @property
    def name(self) -> str:
        source = "public" if self.data_source == PrivacyDataSource.PUBLIC else "private"
        return f"gradient_norm_{source}"
    
    def compute_allocation(
        self,
        model: nn.Module,
        global_clipping_bound: float,
        data_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ) -> Dict[str, float]:
        
        # Select data source for estimation
        if self.data_source == PrivacyDataSource.PUBLIC:
            estimation_loader = self.public_data_loader
            used_public_data = True
            privacy_cost = 0.0
        else:
            if data_loader is None:
                raise ValueError(
                    "data_loader required for gradient estimation with private data"
                )
            estimation_loader = data_loader
            used_public_data = False
            privacy_cost = self.estimation_privacy_epsilon
        
        param_groups = self.get_parameter_groups(model)
        
        # Compute gradient norms
        gradient_norms = self._estimate_gradient_norms(
            model, param_groups, estimation_loader, device
        )
        
        # Normalize and convert to clipping bounds
        normalized = self._normalize_allocations(gradient_norms)
        clipping_bounds = self._to_clipping_bounds(normalized, global_clipping_bound)
        
        # Store metrics
        is_valid, constraint_value = self._verify_constraint(
            clipping_bounds, global_clipping_bound
        )
        self._metrics = AllocationMetrics(
            strategy_name=self.name,
            layer_clipping_bounds=clipping_bounds,
            allocation_ratios=normalized,
            global_clipping_bound=global_clipping_bound,
            constraint_value=constraint_value,
            estimated_gradient_norms=gradient_norms,
            estimation_samples=self.num_estimation_samples,
            privacy_cost_epsilon=privacy_cost,
            used_public_data=used_public_data,
        )
        
        return clipping_bounds
    
    def _estimate_gradient_norms(
        self,
        model: nn.Module,
        param_groups: Dict[str, List[nn.Parameter]],
        data_loader: DataLoader,
        device: str
    ) -> Dict[str, float]:
        """Estimate average gradient norms per parameter group"""
        
        # Initialize accumulators
        group_grad_norms = {name: 0.0 for name in param_groups.keys()}
        
        model.train()
        model.to(device)
        
        # Determine samples to use
        dataset_size = len(data_loader.dataset)
        n_samples = min(self.num_estimation_samples, dataset_size)
        
        if self.verbose:
            print(f"  Estimating gradient norms on {n_samples} samples...")
            print(f"  Data source: {'public' if self.data_source == PrivacyDataSource.PUBLIC else 'private'}")
        
        samples_processed = 0
        iterator = tqdm(data_loader, desc="  Gradient estimation", disable=not self.verbose)
        
        for batch in iterator:
            if samples_processed >= n_samples:
                break
            
            # Handle batch format (supports HuggingFace datasets format)
            batch_tensors = self._prepare_batch(batch, device)
            if batch_tensors is None:
                continue
                
            input_ids, attention_mask, labels = batch_tensors
            
            # Limit batch if needed
            batch_size = input_ids.size(0)
            remaining = n_samples - samples_processed
            if batch_size > remaining:
                input_ids = input_ids[:remaining]
                attention_mask = attention_mask[:remaining]
                labels = labels[:remaining]
                batch_size = remaining
            
            # Forward pass
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Forward pass failed: {e}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradient norms per group
            for group_name, params in param_groups.items():
                group_norm_sq = 0.0
                for param in params:
                    if param.grad is not None:
                        group_norm_sq += param.grad.norm(2).item() ** 2
                group_grad_norms[group_name] += np.sqrt(group_norm_sq)
            
            # Clear gradients
            model.zero_grad()
            
            samples_processed += batch_size
        
        # Average across samples
        if samples_processed > 0:
            for group_name in group_grad_norms:
                group_grad_norms[group_name] /= samples_processed
        
        if self.verbose:
            print(f"  ✓ Processed {samples_processed} samples")
            
        return group_grad_norms
    
    def _prepare_batch(
        self, 
        batch: Dict, 
        device: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Prepare batch tensors from different data formats"""
        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Handle both "labels" and "label" keys
            labels = batch.get("labels", batch.get("label"))
            if labels is None:
                return None
            labels = labels.to(device)
            return input_ids, attention_mask, labels
        except (KeyError, AttributeError):
            return None


class DepthBasedStrategy(AllocationStrategy):
    """
    Allocation based on layer depth in the transformer architecture.
    
    Supports different distribution patterns:
    - 'uniform': Equal allocation (same as UniformStrategy)
    - 'linear_increasing': Linearly increases with depth (later layers get more)
    - 'linear_decreasing': Linearly decreases with depth (earlier layers get more)
    - 'gaussian_peak': Gaussian centered at middle layers
    - 'u_shaped': Higher at edges (early/late layers), lower in middle
    """
    
    VALID_PATTERNS = [
        'uniform', 'linear_increasing', 'linear_decreasing',
        'gaussian_peak', 'u_shaped'
    ]
    
    def __init__(
        self, 
        pattern: str = "linear_increasing",
        gaussian_sigma_factor: float = 0.25,  # sigma = n_blocks * factor
    ):
        """
        Args:
            pattern: Distribution pattern for allocation
            gaussian_sigma_factor: Controls spread for gaussian_peak pattern
        """
        if pattern not in self.VALID_PATTERNS:
            raise ValueError(
                f"Pattern must be one of {self.VALID_PATTERNS}, got '{pattern}'"
            )
        
        self.pattern = pattern
        self.gaussian_sigma_factor = gaussian_sigma_factor
    
    @property
    def name(self) -> str:
        return f"depth_{self.pattern}"
    
    def compute_allocation(
        self,
        model: nn.Module,
        global_clipping_bound: float,
        data_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ) -> Dict[str, float]:
        
        param_groups = self.get_parameter_groups(model)
        
        # Separate block parameters from other parameters
        block_items, other_items = self._separate_blocks(param_groups)
        
        if len(block_items) == 0:
            # No transformer blocks found, fall back to uniform
            print("  Warning: No transformer blocks found, using uniform allocation")
            alpha = 1.0 / np.sqrt(len(param_groups))
            clipping_bounds = {name: global_clipping_bound * alpha for name in param_groups.keys()}
            normalized = {name: alpha for name in param_groups.keys()}
        else:
            # Compute allocations based on pattern
            raw_allocations = self._compute_pattern_allocations(block_items, other_items)
            normalized = self._normalize_allocations(raw_allocations)
            clipping_bounds = self._to_clipping_bounds(normalized, global_clipping_bound)
        
        # Store metrics
        is_valid, constraint_value = self._verify_constraint(
            clipping_bounds, global_clipping_bound
        )
        self._metrics = AllocationMetrics(
            strategy_name=self.name,
            layer_clipping_bounds=clipping_bounds,
            allocation_ratios=normalized,
            global_clipping_bound=global_clipping_bound,
            constraint_value=constraint_value,
        )
        
        return clipping_bounds
    
    def _separate_blocks(
        self, 
        param_groups: Dict[str, List[nn.Parameter]]
    ) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Separate block parameters from other parameters"""
        block_items = []
        other_items = []
        
        for name in param_groups.keys():
            if name.startswith("block_"):
                try:
                    block_idx = int(name.split("_")[1])
                    block_items.append((name, block_idx))
                except (ValueError, IndexError):
                    other_items.append(name)
            else:
                other_items.append(name)
        
        # Sort blocks by index
        block_items.sort(key=lambda x: x[1])
        return block_items, other_items
    
    def _compute_pattern_allocations(
        self,
        block_items: List[Tuple[str, int]],
        other_items: List[str]
    ) -> Dict[str, float]:
        """Compute raw allocations based on the selected pattern"""
        n_blocks = len(block_items)
        raw_allocations = {}
        
        for name, idx in block_items:
            if self.pattern == 'uniform':
                raw_allocations[name] = 1.0
                
            elif self.pattern == 'linear_increasing':
                # Later layers get more
                raw_allocations[name] = (idx + 1) / n_blocks
                
            elif self.pattern == 'linear_decreasing':
                # Earlier layers get more
                raw_allocations[name] = (n_blocks - idx) / n_blocks
                
            elif self.pattern == 'gaussian_peak':
                # Gaussian centered at middle
                center = (n_blocks - 1) / 2
                sigma = n_blocks * self.gaussian_sigma_factor
                raw_allocations[name] = np.exp(-((idx - center) ** 2) / (2 * sigma ** 2))
                
            elif self.pattern == 'u_shaped':
                # Higher at edges, lower in middle
                center = (n_blocks - 1) / 2
                max_dist = center if center > 0 else 1
                dist_from_center = abs(idx - center) / max_dist
                raw_allocations[name] = 0.1 + 0.9 * dist_from_center  # Minimum 0.1
        
        # Handle non-block parameters with average allocation
        if other_items and raw_allocations:
            avg_allocation = np.mean(list(raw_allocations.values()))
            for name in other_items:
                raw_allocations[name] = avg_allocation
        elif other_items:
            for name in other_items:
                raw_allocations[name] = 1.0
        
        return raw_allocations


class CubeRootGradientNormStrategy(GradientNormStrategy):
    """
    Signal-proportional allocation based on cube root of gradient magnitudes.
    
    Similar to GradientNormStrategy, but applies cube root transformation to
    gradient norms before normalization. This reduces the influence of outlier
    layers with very large gradients, leading to a more balanced allocation.
    
    The cube root transformation compresses the range of gradient norms,
    giving smaller-gradient layers relatively more budget compared to
    the standard gradient norm strategy.
    """
    
    @property
    def name(self) -> str:
        source = "public" if self.data_source == PrivacyDataSource.PUBLIC else "private"
        return f"cube_root_gradient_norm_{source}"
    
    def compute_allocation(
        self,
        model: nn.Module,
        global_clipping_bound: float,
        data_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ) -> Dict[str, float]:
        
        # Select data source for estimation
        if self.data_source == PrivacyDataSource.PUBLIC:
            estimation_loader = self.public_data_loader
            used_public_data = True
            privacy_cost = 0.0
        else:
            if data_loader is None:
                raise ValueError(
                    "data_loader required for gradient estimation with private data"
                )
            estimation_loader = data_loader
            used_public_data = False
            privacy_cost = self.estimation_privacy_epsilon
        
        param_groups = self.get_parameter_groups(model)
        
        # Compute gradient norms
        gradient_norms = self._estimate_gradient_norms(
            model, param_groups, estimation_loader, device
        )
        
        # Apply cube root transformation
        cube_root_norms = {
            k: np.cbrt(v) for k, v in gradient_norms.items()
        }
        
        # Normalize and convert to clipping bounds
        normalized = self._normalize_allocations(cube_root_norms)
        clipping_bounds = self._to_clipping_bounds(normalized, global_clipping_bound)
        
        # Store metrics
        is_valid, constraint_value = self._verify_constraint(
            clipping_bounds, global_clipping_bound
        )
        self._metrics = AllocationMetrics(
            strategy_name=self.name,
            layer_clipping_bounds=clipping_bounds,
            allocation_ratios=normalized,
            global_clipping_bound=global_clipping_bound,
            constraint_value=constraint_value,
            estimated_gradient_norms=gradient_norms,
            estimation_samples=self.num_estimation_samples,
            privacy_cost_epsilon=privacy_cost,
            used_public_data=used_public_data,
        )
        
        return clipping_bounds


class InverseDepthStrategy(DepthBasedStrategy):
    """
    Inverse of depth-based strategy for comparison experiments.
    
    Automatically inverts the pattern:
    - linear_increasing <-> linear_decreasing
    - gaussian_peak <-> u_shaped
    - uniform stays uniform
    """
    
    INVERSE_PATTERNS = {
        'linear_increasing': 'linear_decreasing',
        'linear_decreasing': 'linear_increasing',
        'gaussian_peak': 'u_shaped',
        'u_shaped': 'gaussian_peak',
        'uniform': 'uniform'
    }
    
    def __init__(self, base_pattern: str = "linear_increasing"):
        inverse_pattern = self.INVERSE_PATTERNS.get(base_pattern, 'uniform')
        super().__init__(pattern=inverse_pattern)
        self.base_pattern = base_pattern
    
    @property
    def name(self) -> str:
        return f"inverse_depth_{self.base_pattern}"


class CustomAllocationStrategy(AllocationStrategy):
    """
    Custom allocation strategy using a user-provided function.
    
    Allows experiments with arbitrary allocation schemes.
    """
    
    def __init__(
        self,
        allocation_fn: Callable[[int, int], float],
        name: str = "custom",
    ):
        """
        Args:
            allocation_fn: Function(block_idx, n_blocks) -> allocation_weight
            name: Name for this custom strategy
        """
        self._name = name
        self.allocation_fn = allocation_fn
    
    @property
    def name(self) -> str:
        return self._name
    
    def compute_allocation(
        self,
        model: nn.Module,
        global_clipping_bound: float,
        data_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ) -> Dict[str, float]:
        
        param_groups = self.get_parameter_groups(model)
        
        # Separate blocks and other params
        block_items = []
        other_items = []
        
        for name in param_groups.keys():
            if name.startswith("block_"):
                try:
                    block_idx = int(name.split("_")[1])
                    block_items.append((name, block_idx))
                except (ValueError, IndexError):
                    other_items.append(name)
            else:
                other_items.append(name)
        
        block_items.sort(key=lambda x: x[1])
        n_blocks = len(block_items)
        
        # Apply custom function
        raw_allocations = {}
        for name, idx in block_items:
            raw_allocations[name] = self.allocation_fn(idx, n_blocks)
        
        # Average for other params
        if other_items and raw_allocations:
            avg = np.mean(list(raw_allocations.values()))
            for name in other_items:
                raw_allocations[name] = avg
        elif other_items:
            for name in other_items:
                raw_allocations[name] = 1.0
        
        normalized = self._normalize_allocations(raw_allocations)
        clipping_bounds = self._to_clipping_bounds(normalized, global_clipping_bound)
        
        is_valid, constraint_value = self._verify_constraint(
            clipping_bounds, global_clipping_bound
        )
        self._metrics = AllocationMetrics(
            strategy_name=self.name,
            layer_clipping_bounds=clipping_bounds,
            allocation_ratios=normalized,
            global_clipping_bound=global_clipping_bound,
            constraint_value=constraint_value,
        )
        
        return clipping_bounds


# =============================================================================
# Strategy Registry and Factory
# =============================================================================

STRATEGY_REGISTRY = {
    "uniform": UniformStrategy,
    "gradient_norm": GradientNormStrategy,
    "cube_root_gradient_norm": CubeRootGradientNormStrategy,
    "depth_uniform": lambda: DepthBasedStrategy(pattern="uniform"),
    "depth_linear_increasing": lambda: DepthBasedStrategy(pattern="linear_increasing"),
    "depth_linear_decreasing": lambda: DepthBasedStrategy(pattern="linear_decreasing"),
    "depth_gaussian_peak": lambda: DepthBasedStrategy(pattern="gaussian_peak"),
    "depth_u_shaped": lambda: DepthBasedStrategy(pattern="u_shaped"),
}


def get_strategy(name: str, **kwargs) -> AllocationStrategy:
    """
    Factory function to create allocation strategies by name.
    
    Args:
        name: Strategy name (see STRATEGY_REGISTRY)
        **kwargs: Additional arguments for strategy initialization
        
    Returns:
        Initialized AllocationStrategy instance
    """
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    
    strategy_cls = STRATEGY_REGISTRY[name]
    
    if callable(strategy_cls) and not isinstance(strategy_cls, type):
        # Lambda factory function
        return strategy_cls()
    else:
        # Class constructor
        return strategy_cls(**kwargs)


def list_strategies() -> List[str]:
    """List all available strategy names"""
    return list(STRATEGY_REGISTRY.keys())


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_strategies(
    model: nn.Module,
    strategies: List[Tuple[str, AllocationStrategy]],
    global_clipping_bound: float = 1.0,
    data_loader: Optional[DataLoader] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, AllocationMetrics]:
    """
    Compare allocation results across different strategies.
    
    Args:
        model: The model to analyze
        strategies: List of (display_name, strategy) tuples
        global_clipping_bound: Global clipping bound C_global
        data_loader: Optional data for gradient-based strategies
        device: Device for computations
        verbose: Print comparison results
        
    Returns:
        Dictionary mapping strategy name to AllocationMetrics
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ALLOCATION STRATEGY COMPARISON")
        print("=" * 70)
    
    results = {}
    
    for display_name, strategy in strategies:
        if verbose:
            print(f"\n[{display_name}]")
        
        allocations = strategy.compute_allocation(
            model, global_clipping_bound, data_loader, device
        )
        
        metrics = strategy.get_metrics()
        results[display_name] = metrics
        
        if verbose:
            is_valid, constraint_val = strategy._verify_constraint(
                allocations, global_clipping_bound
            )
            
            print(f"  Constraint: sqrt(sum(C_l^2)) = {constraint_val:.6f} "
                  f"(target: {global_clipping_bound:.6f}) {'✓' if is_valid else '✗'}")
            print(f"  Groups: {len(allocations)}")
            print(f"  C_l range: [{min(allocations.values()):.4f}, {max(allocations.values()):.4f}]")
            
            # Show sample allocations
            print(f"  Sample allocations:")
            for i, (name, c_l) in enumerate(list(allocations.items())[:5]):
                print(f"    {name}: {c_l:.4f}")
            if len(allocations) > 5:
                print(f"    ... ({len(allocations) - 5} more)")
    
    if verbose:
        print("\n" + "=" * 70)
    
    return results


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Use absolute imports when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from inseparable_six.config import PrivacyDataSource
    from inseparable_six.metrics import AllocationMetrics
    
    print("Testing Allocation Strategies...")
    print("Available strategies:", list_strategies())
    
    # Example: Create strategies
    uniform = get_strategy("uniform")
    depth_inc = get_strategy("depth_linear_increasing")
    
    # Example: Custom strategy (exponential decay)
    custom = CustomAllocationStrategy(
        allocation_fn=lambda idx, n: np.exp(-idx / n),
        name="exponential_decay"
    )
    
    print(f"\nCreated strategies: {uniform.name}, {depth_inc.name}, {custom.name}")
