"""
Metrics collection and logging for DP training experiments.

Provides structured data collection for:
- Training metrics (loss, accuracy, per-step data)
- Privacy accounting (epsilon over time)
- Layer-wise statistics (gradient norms, clipping ratios)
- Allocation strategy data
- Resource usage (time, memory)

All data is stored in formats suitable for analysis and plotting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import torch

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class StepMetrics:
    """Metrics for a single training step"""
    step: int
    epoch: int
    loss: float
    accuracy: float
    batch_size: int
    learning_rate: float
    
    # Privacy metrics
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    
    # Gradient statistics (optional, for detailed analysis)
    gradient_norm: Optional[float] = None
    clipped_fraction: Optional[float] = None
    
    # Layer-wise statistics (optional)
    layer_gradient_norms: Optional[Dict[str, float]] = None
    layer_clipping_fractions: Optional[Dict[str, float]] = None
    
    # Timing
    step_time_ms: Optional[float] = None
    
    # Memory
    gpu_memory_mb: Optional[float] = None


@dataclass
class EpochMetrics:
    """Aggregated metrics for an epoch"""
    epoch: int
    
    # Training metrics
    train_loss: float
    train_accuracy: float
    
    # Evaluation metrics
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    
    # Privacy metrics
    epsilon: float = 0.0
    delta: float = 0.0
    
    # Aggregated gradient stats
    mean_gradient_norm: Optional[float] = None
    mean_clipped_fraction: Optional[float] = None
    
    # Resource usage
    epoch_time_seconds: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None
    
    # Step-level data (for detailed analysis)
    steps: List[StepMetrics] = field(default_factory=list)


@dataclass
class AllocationMetrics:
    """Metrics for allocation strategy analysis"""
    strategy_name: str
    
    # Per-layer clipping bounds
    layer_clipping_bounds: Dict[str, float] = field(default_factory=dict)
    
    # Normalized allocation ratios
    allocation_ratios: Dict[str, float] = field(default_factory=dict)
    
    # Constraint verification
    global_clipping_bound: float = 1.0
    constraint_value: float = 0.0  # sqrt(sum(C_l^2)), should equal global_clipping_bound
    
    # For gradient-based strategies
    estimated_gradient_norms: Optional[Dict[str, float]] = None
    estimation_samples: int = 0
    privacy_cost_epsilon: float = 0.0  # Privacy spent on estimation (if using private data)
    used_public_data: bool = False
    
    # Timing for allocation computation
    allocation_time_seconds: float = 0.0


@dataclass
class ExperimentResult:
    """Complete results from a training experiment"""
    experiment_name: str
    config: Dict[str, Any]  # Serialized config
    
    # Training results
    epochs: List[EpochMetrics] = field(default_factory=list)
    
    # Final metrics
    final_train_loss: float = 0.0
    final_train_accuracy: float = 0.0
    final_eval_accuracy: float = 0.0
    final_epsilon: float = 0.0
    final_delta: float = 0.0
    
    # Allocation strategy info
    allocation_metrics: Optional[AllocationMetrics] = None
    
    # Model info
    model_name: str = ""
    total_parameters: int = 0
    trainable_parameters: int = 0
    lora_enabled: bool = False
    
    # Resource usage
    total_training_time_seconds: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    
    # Timing
    start_time: str = ""
    end_time: str = ""
    
    # Experiment type for comparison
    experiment_type: str = ""  # "no_dp", "uniform_dp", "layer_adaptive_dp"
    strategy_name: str = ""    # Strategy used (if applicable)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_peak_memory_stats():
    """Reset peak memory statistics"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class MetricsCollector:
    """
    Collects and manages training metrics throughout an experiment.
    
    Usage:
        collector = MetricsCollector(experiment_name="my_experiment")
        collector.set_config(config.to_dict())
        
        for epoch in range(epochs):
            collector.start_epoch(epoch)
            for step, batch in enumerate(dataloader):
                # ... training step ...
                collector.log_step(step, epoch, loss, accuracy, batch_size)
            collector.end_epoch(train_loss, train_acc, eval_acc, epsilon)
        
        results = collector.finalize()
        collector.save("results.json")
    """
    
    def __init__(self, experiment_name: str = "experiment"):
        self.experiment_name = experiment_name
        self.config: Dict[str, Any] = {}
        
        # Current state
        self._current_epoch: int = 0
        self._current_steps: List[StepMetrics] = []
        self._epoch_start_time: Optional[float] = None
        
        # Collected data
        self.epochs: List[EpochMetrics] = []
        self.allocation_metrics: Optional[AllocationMetrics] = None
        
        # Model info (set later)
        self.model_name: str = ""
        self.total_parameters: int = 0
        self.trainable_parameters: int = 0
        self.lora_enabled: bool = False
        
        # Experiment type
        self.experiment_type: str = ""
        self.strategy_name: str = ""
        
        # Timing
        self._experiment_start_time: Optional[float] = None
        self._start_timestamp: str = ""
        
    def set_config(self, config: Dict[str, Any]):
        """Set the experiment configuration"""
        self.config = config
        
    def set_model_info(
        self, 
        model_name: str, 
        total_params: int, 
        trainable_params: int,
        lora_enabled: bool = False
    ):
        """Set model metadata"""
        self.model_name = model_name
        self.total_parameters = total_params
        self.trainable_parameters = trainable_params
        self.lora_enabled = lora_enabled
        
    def set_experiment_type(self, experiment_type: str, strategy_name: str = ""):
        """Set experiment type for comparison purposes"""
        self.experiment_type = experiment_type
        self.strategy_name = strategy_name
        
    def set_allocation_metrics(self, metrics: AllocationMetrics):
        """Set allocation strategy metrics"""
        self.allocation_metrics = metrics
        
    def start_experiment(self):
        """Mark the start of the experiment"""
        import time
        self._experiment_start_time = time.time()
        self._start_timestamp = datetime.now().isoformat()
        reset_peak_memory_stats()
        
    def start_epoch(self, epoch: int):
        """Mark the start of an epoch"""
        import time
        self._current_epoch = epoch
        self._current_steps = []
        self._epoch_start_time = time.time()
        
    def log_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        accuracy: float,
        batch_size: int,
        learning_rate: float = 0.0,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        clipped_fraction: Optional[float] = None,
        layer_gradient_norms: Optional[Dict[str, float]] = None,
        layer_clipping_fractions: Optional[Dict[str, float]] = None,
        step_time_ms: Optional[float] = None,
    ):
        """Log metrics for a single training step"""
        step_metrics = StepMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epsilon=epsilon,
            delta=delta,
            gradient_norm=gradient_norm,
            clipped_fraction=clipped_fraction,
            layer_gradient_norms=layer_gradient_norms,
            layer_clipping_fractions=layer_clipping_fractions,
            step_time_ms=step_time_ms,
            gpu_memory_mb=get_gpu_memory_mb(),
        )
        self._current_steps.append(step_metrics)
        
    def end_epoch(
        self,
        train_loss: float,
        train_accuracy: float,
        eval_accuracy: Optional[float] = None,
        eval_loss: Optional[float] = None,
        epsilon: float = 0.0,
        delta: float = 0.0,
    ):
        """Finalize metrics for the current epoch"""
        import time
        
        epoch_time = None
        if self._epoch_start_time is not None:
            epoch_time = time.time() - self._epoch_start_time
            
        # Compute aggregated gradient stats
        mean_grad_norm = None
        mean_clipped_frac = None
        
        grad_norms = [s.gradient_norm for s in self._current_steps if s.gradient_norm is not None]
        if grad_norms:
            mean_grad_norm = np.mean(grad_norms)
            
        clipped_fracs = [s.clipped_fraction for s in self._current_steps if s.clipped_fraction is not None]
        if clipped_fracs:
            mean_clipped_frac = np.mean(clipped_fracs)
        
        epoch_metrics = EpochMetrics(
            epoch=self._current_epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            eval_accuracy=eval_accuracy,
            eval_loss=eval_loss,
            epsilon=epsilon,
            delta=delta,
            mean_gradient_norm=mean_grad_norm,
            mean_clipped_fraction=mean_clipped_frac,
            epoch_time_seconds=epoch_time,
            peak_gpu_memory_mb=get_peak_gpu_memory_mb(),
            steps=self._current_steps.copy(),
        )
        self.epochs.append(epoch_metrics)
        
    def finalize(self) -> ExperimentResult:
        """Finalize the experiment and return results"""
        import time
        
        total_time = 0.0
        if self._experiment_start_time is not None:
            total_time = time.time() - self._experiment_start_time
            
        end_timestamp = datetime.now().isoformat()
        
        # Get final metrics from last epoch
        final_epoch = self.epochs[-1] if self.epochs else None
        
        return ExperimentResult(
            experiment_name=self.experiment_name,
            config=self.config,
            epochs=self.epochs,
            final_train_loss=final_epoch.train_loss if final_epoch else 0.0,
            final_train_accuracy=final_epoch.train_accuracy if final_epoch else 0.0,
            final_eval_accuracy=final_epoch.eval_accuracy if final_epoch else 0.0,
            final_epsilon=final_epoch.epsilon if final_epoch else 0.0,
            final_delta=final_epoch.delta if final_epoch else 0.0,
            allocation_metrics=self.allocation_metrics,
            model_name=self.model_name,
            total_parameters=self.total_parameters,
            trainable_parameters=self.trainable_parameters,
            lora_enabled=self.lora_enabled,
            total_training_time_seconds=total_time,
            peak_gpu_memory_mb=get_peak_gpu_memory_mb(),
            start_time=self._start_timestamp,
            end_time=end_timestamp,
            experiment_type=self.experiment_type,
            strategy_name=self.strategy_name,
        )
        
    def save(self, filepath: str):
        """Save results to JSON file"""
        results = self.finalize()
        self._save_results(results, filepath)
        
    @staticmethod
    def _save_results(results: ExperimentResult, filepath: str):
        """Save ExperimentResult to JSON"""
        from dataclasses import asdict
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict with proper handling of nested dataclasses
        def convert_to_serializable(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert_to_serializable(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        data = convert_to_serializable(results)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    @staticmethod
    def load(filepath: str) -> ExperimentResult:
        """Load results from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Reconstruct nested dataclasses
        epochs = []
        for epoch_data in data.get('epochs', []):
            steps = [StepMetrics(**s) for s in epoch_data.pop('steps', [])]
            epochs.append(EpochMetrics(**epoch_data, steps=steps))
            
        allocation_data = data.get('allocation_metrics')
        allocation_metrics = AllocationMetrics(**allocation_data) if allocation_data else None
        
        return ExperimentResult(
            experiment_name=data['experiment_name'],
            config=data['config'],
            epochs=epochs,
            final_train_loss=data.get('final_train_loss', 0.0),
            final_train_accuracy=data.get('final_train_accuracy', 0.0),
            final_eval_accuracy=data.get('final_eval_accuracy', 0.0),
            final_epsilon=data.get('final_epsilon', 0.0),
            final_delta=data.get('final_delta', 0.0),
            allocation_metrics=allocation_metrics,
            model_name=data.get('model_name', ''),
            total_parameters=data.get('total_parameters', 0),
            trainable_parameters=data.get('trainable_parameters', 0),
            lora_enabled=data.get('lora_enabled', False),
            total_training_time_seconds=data.get('total_training_time_seconds', 0.0),
            peak_gpu_memory_mb=data.get('peak_gpu_memory_mb', 0.0),
            start_time=data.get('start_time', ''),
            end_time=data.get('end_time', ''),
            experiment_type=data.get('experiment_type', ''),
            strategy_name=data.get('strategy_name', ''),
            metadata=data.get('metadata', {}),
        )


# =============================================================================
# Utility functions for extracting plotting data
# =============================================================================

def extract_training_curves(results: ExperimentResult) -> Dict[str, List[float]]:
    """Extract data for training curve plots"""
    return {
        'epochs': [e.epoch for e in results.epochs],
        'train_loss': [e.train_loss for e in results.epochs],
        'train_accuracy': [e.train_accuracy for e in results.epochs],
        'eval_accuracy': [e.eval_accuracy for e in results.epochs if e.eval_accuracy is not None],
        'epsilon': [e.epsilon for e in results.epochs],
        'epoch_time': [e.epoch_time_seconds for e in results.epochs if e.epoch_time_seconds is not None],
    }


def extract_step_level_data(results: ExperimentResult) -> Dict[str, List[float]]:
    """Extract step-level data for detailed analysis"""
    all_steps = []
    for epoch in results.epochs:
        all_steps.extend(epoch.steps)
        
    return {
        'step': [s.step for s in all_steps],
        'epoch': [s.epoch for s in all_steps],
        'loss': [s.loss for s in all_steps],
        'accuracy': [s.accuracy for s in all_steps],
        'gradient_norm': [s.gradient_norm for s in all_steps if s.gradient_norm is not None],
        'gpu_memory_mb': [s.gpu_memory_mb for s in all_steps if s.gpu_memory_mb is not None],
    }


def extract_allocation_data(results: ExperimentResult) -> Optional[Dict[str, Any]]:
    """Extract allocation strategy data for visualization"""
    if results.allocation_metrics is None:
        return None
        
    alloc = results.allocation_metrics
    
    # Sort by layer/block name
    sorted_layers = sorted(alloc.layer_clipping_bounds.keys())
    
    return {
        'strategy_name': alloc.strategy_name,
        'layers': sorted_layers,
        'clipping_bounds': [alloc.layer_clipping_bounds[l] for l in sorted_layers],
        'allocation_ratios': [alloc.allocation_ratios.get(l, 0) for l in sorted_layers],
        'global_clipping_bound': alloc.global_clipping_bound,
        'constraint_value': alloc.constraint_value,
        'estimated_gradient_norms': (
            [alloc.estimated_gradient_norms.get(l, 0) for l in sorted_layers]
            if alloc.estimated_gradient_norms else None
        ),
    }


def compare_experiments(
    results_list: List[ExperimentResult],
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple experiments for analysis and plotting.
    
    Returns a dictionary with experiment names as keys and summary data as values.
    """
    comparison = {}
    
    for results in results_list:
        name = results.experiment_name
        comparison[name] = {
            'experiment_type': results.experiment_type,
            'strategy_name': results.strategy_name,
            'final_eval_accuracy': results.final_eval_accuracy,
            'final_train_accuracy': results.final_train_accuracy,
            'final_epsilon': results.final_epsilon,
            'total_time_seconds': results.total_training_time_seconds,
            'peak_memory_mb': results.peak_gpu_memory_mb,
            'trainable_params': results.trainable_parameters,
            'lora_enabled': results.lora_enabled,
            'training_curves': extract_training_curves(results),
        }
        
    return comparison


def save_comparison(comparison: Dict[str, Dict[str, Any]], filepath: str):
    """Save experiment comparison to JSON file"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(comparison, f, indent=2)


def load_comparison(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Load experiment comparison from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


# =============================================================================
# Summary table generation
# =============================================================================

def generate_summary_table(results_list: List[ExperimentResult]) -> str:
    """Generate a formatted summary table of experiment results"""
    headers = [
        "Experiment", "Type", "Strategy", "LoRA", 
        "Eval Acc", "Train Acc", "Epsilon", "Time (s)", "Memory (MB)"
    ]
    
    rows = []
    for r in results_list:
        rows.append([
            r.experiment_name[:20],
            r.experiment_type,
            r.strategy_name or "N/A",
            "Yes" if r.lora_enabled else "No",
            f"{r.final_eval_accuracy:.2f}%",
            f"{r.final_train_accuracy:.2f}%",
            f"{r.final_epsilon:.2f}" if r.final_epsilon > 0 else "N/A",
            f"{r.total_training_time_seconds:.1f}",
            f"{r.peak_gpu_memory_mb:.0f}",
        ])
    
    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 
                  for i in range(len(headers))]
    
    # Format table
    lines = []
    header_line = "|".join(h.center(w) for h, w in zip(headers, col_widths))
    separator = "+".join("-" * w for w in col_widths)
    
    lines.append(separator)
    lines.append(header_line)
    lines.append(separator)
    
    for row in rows:
        line = "|".join(str(cell).center(w) for cell, w in zip(row, col_widths))
        lines.append(line)
    
    lines.append(separator)
    
    return "\n".join(lines)


# =============================================================================
# Plotting utilities
# =============================================================================

def _ensure_matplotlib():
    """Check if matplotlib is available"""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )


def plot_training_curves(
    results_list: List[ExperimentResult],
    output_dir: str,
    timestamp: str = "",
) -> List[str]:
    """
    Generate training curve plots comparing all experiments.
    
    Creates plots for:
    - Accuracy over epochs (train and eval)
    - Loss over epochs
    - Privacy budget (epsilon) over epochs
    - Training time comparison
    
    Args:
        results_list: List of experiment results to compare
        output_dir: Directory to save plots
        timestamp: Timestamp string for filenames
        
    Returns:
        List of saved plot filepaths
    """
    _ensure_matplotlib()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Set up style
    plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else None
    colors = plt.cm.tab10.colors
    
    # 1. Accuracy over epochs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, results in enumerate(results_list):
        curves = extract_training_curves(results)
        epochs = curves['epochs']
        color = colors[i % len(colors)]
        label = f"{results.experiment_name}"
        
        if epochs:
            # Adjust epochs to be 1-indexed for display
            display_epochs = [e + 1 for e in epochs]
            ax1.plot(display_epochs, curves['train_accuracy'], 
                    marker='o', color=color, label=label, linewidth=2)
            ax2.plot(display_epochs, curves['eval_accuracy'], 
                    marker='s', color=color, label=label, linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax1.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_ylim([0, 105])
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Evaluation Accuracy (%)', fontsize=12)
    ax2.set_title('Evaluation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    filepath = output_path / f"accuracy_curves_{timestamp}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(str(filepath))
    
    # 2. Loss over epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, results in enumerate(results_list):
        curves = extract_training_curves(results)
        epochs = curves['epochs']
        color = colors[i % len(colors)]
        label = f"{results.experiment_name}"
        
        if epochs:
            display_epochs = [e + 1 for e in epochs]
            ax.plot(display_epochs, curves['train_loss'], 
                   marker='o', color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    filepath = output_path / f"loss_curves_{timestamp}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(str(filepath))
    
    # 3. Privacy budget (epsilon) over epochs
    dp_results = [r for r in results_list if r.final_epsilon > 0]
    if dp_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, results in enumerate(dp_results):
            curves = extract_training_curves(results)
            epochs = curves['epochs']
            color = colors[i % len(colors)]
            label = f"{results.experiment_name}"
            
            if epochs and any(e > 0 for e in curves['epsilon']):
                display_epochs = [e + 1 for e in epochs]
                ax.plot(display_epochs, curves['epsilon'], 
                       marker='o', color=color, label=label, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Privacy Budget (ε)', fontsize=12)
        ax.set_title('Privacy Budget Consumed Over Epochs', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        plt.tight_layout()
        filepath = output_path / f"privacy_budget_{timestamp}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(filepath))
    
    # 4. Final metrics comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = [r.experiment_name.replace('_', '\n') for r in results_list]
    x = np.arange(len(names))
    width = 0.6
    
    # Accuracy comparison
    eval_accs = [r.final_eval_accuracy for r in results_list]
    bars = axes[0].bar(x, eval_accs, width, color=[colors[i % len(colors)] for i in range(len(results_list))])
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Final Evaluation Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylim([0, 105])
    for bar, acc in zip(bars, eval_accs):
        axes[0].annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)
    
    # Epsilon comparison
    epsilons = [r.final_epsilon if r.final_epsilon > 0 else 0 for r in results_list]
    bars = axes[1].bar(x, epsilons, width, color=[colors[i % len(colors)] for i in range(len(results_list))])
    axes[1].set_ylabel('Privacy Budget (ε)', fontsize=12)
    axes[1].set_title('Final Privacy Budget', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=9)
    for bar, eps in zip(bars, epsilons):
        label = f'{eps:.2f}' if eps > 0 else 'No DP'
        axes[1].annotate(label, xy=(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0.1)),
                        ha='center', va='bottom', fontsize=9)
    
    # Training time comparison
    times = [r.total_training_time_seconds for r in results_list]
    bars = axes[2].bar(x, times, width, color=[colors[i % len(colors)] for i in range(len(results_list))])
    axes[2].set_ylabel('Time (seconds)', fontsize=12)
    axes[2].set_title('Total Training Time', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, fontsize=9)
    for bar, t in zip(bars, times):
        axes[2].annotate(f'{t:.1f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filepath = output_path / f"metrics_comparison_{timestamp}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(str(filepath))
    
    return saved_files


def plot_allocation_comparison(
    results_list: List[ExperimentResult],
    output_dir: str,
    timestamp: str = "",
) -> List[str]:
    """
    Generate plots comparing allocation strategies across experiments.
    
    Creates plots for:
    - Per-layer clipping bounds comparison (bar chart)
    - Per-layer clipping bounds line chart (for trend visualization)
    - Allocation ratios heatmap
    
    Args:
        results_list: List of experiment results with allocation metrics
        output_dir: Directory to save plots
        timestamp: Timestamp string for filenames
        
    Returns:
        List of saved plot filepaths
    """
    _ensure_matplotlib()
    
    # Filter results that have allocation metrics
    results_with_alloc = [r for r in results_list if r.allocation_metrics is not None]
    if not results_with_alloc:
        return []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    colors = plt.cm.tab10.colors
    
    # Get all layer names (assuming same layers across experiments)
    all_layers = set()
    for r in results_with_alloc:
        all_layers.update(r.allocation_metrics.layer_clipping_bounds.keys())
    
    # Sort layers naturally (block_0, block_1, ..., other)
    def sort_key(name):
        if name.startswith('block_'):
            try:
                return (0, int(name.split('_')[1]))
            except:
                return (1, name)
        return (2, name)
    
    sorted_layers = sorted(all_layers, key=sort_key)
    x = np.arange(len(sorted_layers))
    
    # 1. Plot clipping bounds comparison (bar chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    width = 0.8 / len(results_with_alloc)
    
    for i, results in enumerate(results_with_alloc):
        alloc = results.allocation_metrics
        bounds = [alloc.layer_clipping_bounds.get(l, 0) for l in sorted_layers]
        offset = (i - len(results_with_alloc)/2 + 0.5) * width
        label = results.strategy_name or results.experiment_name
        ax.bar(x + offset, bounds, width, 
               label=label, color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Clipping Bound (C_l)', fontsize=12)
    ax.set_title('Per-Layer Clipping Bounds by Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('block_', 'B').replace('other', 'Other') for l in sorted_layers], 
                       rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=9, title='Strategy')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = output_path / f"allocation_bounds_bar_{timestamp}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(str(filepath))
    
    # 2. Plot clipping bounds as line chart (trend visualization)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, results in enumerate(results_with_alloc):
        alloc = results.allocation_metrics
        bounds = [alloc.layer_clipping_bounds.get(l, 0) for l in sorted_layers]
        label = results.strategy_name or results.experiment_name
        ax.plot(x, bounds, marker='o', linewidth=2, markersize=8,
                label=label, color=colors[i % len(colors)])
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Clipping Bound (C_l)', fontsize=12)
    ax.set_title('Per-Layer Clipping Bounds: Strategy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('block_', 'B').replace('other', 'Other') for l in sorted_layers], 
                       rotation=45, ha='right', fontsize=10)
    ax.legend(loc='best', fontsize=9, title='Strategy')
    ax.grid(True, alpha=0.3)
    
    # Add annotation for uniform strategy reference
    if sorted_layers:
        uniform_bound = 1.0 / np.sqrt(len(sorted_layers))
        ax.axhline(y=uniform_bound, color='gray', linestyle='--', alpha=0.5, 
                   label=f'Uniform ref: {uniform_bound:.3f}')
    
    plt.tight_layout()
    filepath = output_path / f"allocation_bounds_line_{timestamp}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(str(filepath))
    
    # 3. Allocation ratios comparison (normalized view)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, results in enumerate(results_with_alloc):
        alloc = results.allocation_metrics
        ratios = [alloc.allocation_ratios.get(l, 0) for l in sorted_layers]
        label = results.strategy_name or results.experiment_name
        ax.plot(x, ratios, marker='s', linewidth=2, markersize=7,
                label=label, color=colors[i % len(colors)])
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Allocation Ratio (α_l)', fontsize=12)
    ax.set_title('Normalized Allocation Ratios by Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('block_', 'B').replace('other', 'Other') for l in sorted_layers], 
                       rotation=45, ha='right', fontsize=10)
    ax.legend(loc='best', fontsize=9, title='Strategy')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = output_path / f"allocation_ratios_{timestamp}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(str(filepath))
    
    return saved_files


def plot_utility_privacy_tradeoff(
    results_list: List[ExperimentResult],
    output_dir: str,
    timestamp: str = "",
) -> List[str]:
    """
    Generate utility-privacy trade-off plots.
    
    Creates:
    - Accuracy vs Privacy Budget scatter plot
    - Utility gap from non-DP baseline bar chart
    
    Args:
        results_list: List of experiment results
        output_dir: Directory to save plots
        timestamp: Timestamp string for filenames
        
    Returns:
        List of saved plot filepaths
    """
    _ensure_matplotlib()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    colors = plt.cm.tab10.colors
    
    # Separate DP and non-DP results
    dp_results = [r for r in results_list if r.final_epsilon > 0]
    non_dp_results = [r for r in results_list if r.final_epsilon == 0 or r.experiment_type == 'no_dp']
    
    # Get baseline accuracy (non-DP)
    baseline_accuracy = max([r.final_eval_accuracy for r in non_dp_results], default=100.0)
    
    if dp_results:
        # 1. Utility Gap Bar Chart (difference from baseline)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = []
        gaps = []
        accuracies = []
        
        for r in dp_results:
            names.append(r.strategy_name or r.experiment_name)
            gap = baseline_accuracy - r.final_eval_accuracy
            gaps.append(gap)
            accuracies.append(r.final_eval_accuracy)
        
        x = np.arange(len(names))
        bars = ax.bar(x, gaps, color=[colors[i % len(colors)] for i in range(len(names))], 
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Accuracy Gap from Baseline (%)', fontsize=12)
        ax.set_title(f'Utility Gap: Accuracy Loss vs Non-DP Baseline ({baseline_accuracy:.1f}%)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, gap, acc in zip(bars, gaps, accuracies):
            ax.annotate(f'-{gap:.1f}%\n({acc:.1f}%)', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add baseline reference line at 0
        ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Baseline (No DP)')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        filepath = output_path / f"utility_gap_{timestamp}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(filepath))
        
        # 2. Strategy Comparison Radar/Spider Chart (if enough metrics)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Compare strategies on multiple dimensions
        metrics_to_compare = ['final_eval_accuracy', 'final_train_accuracy']
        
        strategy_data = {}
        for r in dp_results:
            name = r.strategy_name or r.experiment_name
            strategy_data[name] = {
                'Eval Accuracy': r.final_eval_accuracy,
                'Train Accuracy': r.final_train_accuracy,
                'Training Time': r.total_training_time_seconds,
            }
        
        # Simple grouped bar for comparison
        metrics = list(list(strategy_data.values())[0].keys())
        x = np.arange(len(metrics))
        width = 0.8 / len(strategy_data)
        
        for i, (strategy, data) in enumerate(strategy_data.items()):
            values = [data[m] for m in metrics]
            # Normalize time (invert so lower is better, scale to 0-100)
            if 'Training Time' in metrics:
                max_time = max(d['Training Time'] for d in strategy_data.values())
                time_idx = metrics.index('Training Time')
                values[time_idx] = 100 * (1 - values[time_idx] / max_time) if max_time > 0 else 100
            
            offset = (i - len(strategy_data)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=strategy, 
                   color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value (% or normalized)', fontsize=12)
        ax.set_title('Strategy Comparison Across Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(loc='best', fontsize=9, title='Strategy')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        filepath = output_path / f"strategy_comparison_{timestamp}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(filepath))
    
    return saved_files


def generate_all_plots(
    results_list: List[ExperimentResult],
    output_dir: str,
    timestamp: str = "",
) -> List[str]:
    """
    Generate all available plots for experiment results.
    
    Args:
        results_list: List of experiment results
        output_dir: Directory to save plots
        timestamp: Timestamp string for filenames
        
    Returns:
        List of all saved plot filepaths
    """
    if not HAS_MATPLOTLIB:
        print("  [Warning] matplotlib not installed, skipping plot generation")
        return []
    
    saved_files = []
    
    try:
        # Training curves (accuracy, loss, privacy over epochs)
        files = plot_training_curves(results_list, output_dir, timestamp)
        saved_files.extend(files)
        
        # Allocation comparison (per-layer clipping bounds)
        files = plot_allocation_comparison(results_list, output_dir, timestamp)
        saved_files.extend(files)
        
        # Utility-privacy trade-off plots
        files = plot_utility_privacy_tradeoff(results_list, output_dir, timestamp)
        saved_files.extend(files)
        
    except Exception as e:
        print(f"  [Warning] Error generating plots: {e}")
        import traceback
        traceback.print_exc()
    
    return saved_files

