import hydra
from hydra.utils import instantiate

import lightning as L

import memray
import os
from hydra.core.hydra_config import HydraConfig


@hydra.main(version_base="1.3", config_path=".", config_name="config_debug.yaml")
def main(cfg) -> None:
    # output_dir = HydraConfig.get().runtime.output_dir
    # memray_output_path = os.path.join(output_dir, "memray_train.bin")
    # with memray.Tracker(memray_output_path, native_traces=False, follow_fork=True):
        # train
    trainer: L.Trainer = instantiate(cfg.trainer)
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.datamodule)
    # Resume from checkpoint if provided
    ckpt_path = getattr(cfg, "ckpt_path", None)

    fit_kwargs = dict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    trainer.fit(**fit_kwargs)


if __name__ == "__main__":
    # import debugpy
    # try:
    #     debugpy.listen(('localhost', 9502))
    #     print('Waiting for debugger attach')
    #     debugpy.wait_for_client()
    # except Exception as e:
    #     pass
    main()