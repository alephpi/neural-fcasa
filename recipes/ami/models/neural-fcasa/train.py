from argparse import ArgumentParser
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

import lightning as L


# import debugpy
# try:
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Config file in YAML format")
    parser.add_argument("--working_directory", type=Path, default=Path.cwd(), help="Working directory")
    parser.add_argument("--resume_ckpt_dir", type=Path, default=None, help="Path to the dir that stores the checkpoint")
    args, unk_args = parser.parse_known_args()

    # load config
    config = oc.merge(
        {
            "base_config_path": str(Path(__file__).parent / "config"),
            "base_config": "${base_config_path}/train_base.yaml",
        },
        vars(args),
        oc.load(args.config),
        oc.from_cli(unk_args),
    )

    config = oc.merge(
        oc.load(config.base_config),
        config,
    )

    # train
    trainer: L.Trainer = instantiate(config.trainer)
    model = instantiate(config.task)
    datamodule = instantiate(config.datamodule)

    # Resume from checkpoint if provided
    fit_kwargs = dict(model=model, datamodule=datamodule)
    if args.resume_ckpt_dir is not None:
        fit_kwargs["ckpt_path"] = str(args.resume_ckpt_dir)+'/checkpoints/last.ckpt'
        fit_kwargs["default_root_dir"] = str(args.resume_ckpt_dir)

    trainer.fit(**fit_kwargs)


if __name__ == "__main__":
    main()
