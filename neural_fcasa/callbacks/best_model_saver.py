import lightning as lt
import os
from pathlib import Path
import shutil

class BestModelSaverCallback(lt.Callback):
    def __init__(self, best_name="best.ckpt"):
        super().__init__()
        self.best_name = best_name

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            checkpoint_callback = trainer.checkpoint_callback
            if checkpoint_callback:
                best_model_path = Path(checkpoint_callback.best_model_path)
                if best_model_path.exists():
                    shutil.copy(best_model_path, best_model_path.parent / self.best_name)
                    print(f"{best_model_path.name} copied to {self.best_name}")