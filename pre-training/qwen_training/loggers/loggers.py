import os
import csv
import torch.distributed as dist
from transformers import TrainerCallback

def is_main_process():
    # When distributed is initialized, trust the process group rank.
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    # Callbacks are constructed before trainer.train() initializes the process
    # group, so dist may not be ready yet. torchrun always sets RANK, so fall
    # back to it to avoid every rank thinking it is the main process.
    return int(os.environ.get("RANK", "0")) == 0

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, filename = "training_log.csv"):
        self.filename = filename
        # Only rank 0 creates/truncates the log file; otherwise every rank races
        # to write the header to the same path.
        if is_main_process():
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
            # Only rank 0 writes the checkpoint. Under ZeRO stage 2 the model
            # params are replicated on every rank, so rank 0 holds a complete
            # model; letting all ranks save concurrently to the same path just
            # races and corrupts the files.
            if not is_main_process():
                return
            model = kwargs["model"]
            save_path = os.path.join(self.output_dir, f"model_step_{state.global_step}")

            model.save_pretrained(save_path)
            self.trainer.tokenizer.save_pretrained(save_path)
