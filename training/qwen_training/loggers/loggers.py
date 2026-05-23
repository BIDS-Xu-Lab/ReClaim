import os
import csv
import numpy as np
import torch.distributed as dist
from collections import deque
from transformers import TrainerCallback

def is_main_process():
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class LossSpikeException(Exception):
    """Exception raised when a loss spike is detected."""
    def __init__(self, current_loss, avg_loss, step):
        self.current_loss = current_loss
        self.avg_loss = avg_loss
        self.step = step
        super().__init__(
            f"Loss spike detected at step {step}: "
            f"current_loss={current_loss:.4f} > 2x avg_loss={avg_loss:.4f}"
        )


class LossSpikeMonitorCallback(TrainerCallback):
    """
    Monitors training loss and detects spikes.
    If current loss > spike_threshold * average of last window_size steps,
    raises LossSpikeException to trigger checkpoint reload.
    
    Args:
        window_size: Number of recent steps to compute rolling average (default: 100)
        spike_threshold: Multiplier for spike detection (default: 2.0, i.e., 2x average)
        min_steps: Minimum steps before spike detection starts (default: 100)
        cooldown_steps: Steps to wait after spike before checking again (default: 50)
    """
    def __init__(
        self, 
        window_size: int = 100, 
        spike_threshold: float = 2.0,
        min_steps: int = 100,
        cooldown_steps: int = 50
    ):
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.min_steps = min_steps
        self.cooldown_steps = cooldown_steps
        self.loss_history = deque(maxlen=window_size)
        self.last_spike_step = -1
        self.spike_detected = False
        self.spike_info = None
        
    def reset(self):
        """Reset the monitor state after reloading from checkpoint."""
        self.loss_history.clear()
        self.spike_detected = False
        self.spike_info = None
        # Keep last_spike_step to maintain cooldown
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
            
        current_loss = logs["loss"]
        current_step = state.global_step
        
        # Skip if loss is None or NaN
        if current_loss is None or np.isnan(current_loss):
            return
            
        # Check for spike if we have enough history
        if len(self.loss_history) >= self.min_steps:
            # Check cooldown period
            if current_step - self.last_spike_step > self.cooldown_steps:
                avg_loss = np.mean(list(self.loss_history))
                
                if current_loss > self.spike_threshold * avg_loss:
                    self.spike_detected = True
                    self.spike_info = {
                        "current_loss": current_loss,
                        "avg_loss": avg_loss,
                        "step": current_step
                    }
                    self.last_spike_step = current_step
                    
                    if is_main_process():
                        print(f"\n{'='*60}")
                        print(f"[LOSS SPIKE DETECTED] Step {current_step}")
                        print(f"  Current loss: {current_loss:.4f}")
                        print(f"  Average loss (last {len(self.loss_history)} steps): {avg_loss:.4f}")
                        print(f"  Threshold: {self.spike_threshold}x = {self.spike_threshold * avg_loss:.4f}")
                        print(f"  Triggering checkpoint reload...")
                        print(f"{'='*60}\n")
                    
                    # Stop training to trigger reload
                    control.should_training_stop = True
                    return
        
        # Add current loss to history
        self.loss_history.append(current_loss)
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Log monitoring configuration at training start."""
        if is_main_process():
            print(f"\n[LossSpikeMonitor] Enabled with:")
            print(f"  - Window size: {self.window_size} steps")
            print(f"  - Spike threshold: {self.spike_threshold}x average")
            print(f"  - Min steps before monitoring: {self.min_steps}")
            print(f"  - Cooldown after spike: {self.cooldown_steps} steps\n")

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, filename = "training_log.csv"):
        self.filename = filename
        with open(self.filename, "w", newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "grad_norm", "learning_rate", "eval_loss"])

    def on_log(self, args, state, control, logs = None, **kwargs):
        if not is_main_process():
            return
        if logs is not None:
            with open(self.filename, "a", newline = "") as f:
                writer = csv.writer(f)
                writer.writerow([
                    state.global_step,
                    logs.get("loss"),
                    logs.get("grad_norm"),
                    logs.get("learning_rate"),
                    logs.get("eval_loss")
                ])

class LightweightSaveCallBack(TrainerCallback):
    def __init__(self, output_dir, save_every_steps=1000):
        self.output_dir = os.path.join(output_dir, "lightweight_models")
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_every_steps = save_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_every_steps == 0:
            model = kwargs["model"]
            save_path = os.path.join(self.output_dir, f"model_step_{state.global_step}")
            
            model.save_pretrained(save_path)
            self.trainer.tokenizer.save_pretrained(save_path)
