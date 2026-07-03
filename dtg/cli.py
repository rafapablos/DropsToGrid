import torch
from jsonargparse import ActionYesNo
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.tuner.tuning import Tuner

from dtg.callbacks.save_config import SaveConfigCallback
from dtg.data.datamodule import DtGDataModule
from dtg.trainer import DtGTrainer


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "run_name",
            type=str,
            help="Name of the run.",
        )
        parser.add_argument(
            "--compile",
            action=ActionYesNo,
            default=True,
            help="Whether to compile the model using torch.compile.",
        )
        parser.add_argument(
            "--find_lr",
            action=ActionYesNo,
            default=False,
            help="Whether to run the learning rate finder.",
        )

        parser.link_arguments("run_name", "trainer.run_name")
        parser.link_arguments(
            "data.sources",
            "model.init_args.sources",
            apply_on="instantiate",
        )
        parser.link_arguments("trainer.max_epochs", "model.init_args.max_epochs")

    def before_instantiate_classes(self) -> None:
        config = self.config[self.config["subcommand"]]

        # Add input and target sources to sources if not already there
        input_source = config["model"]["init_args"]["input_source"]
        target_source = config["model"]["init_args"]["target_source"]
        sources = config["data"]["sources"]
        for source in [input_source, target_source]:
            if source not in sources:
                sources.append(source)
            if source == "wumap_4km" and "wumap_mask_4km" not in sources:
                sources.append("wumap_mask_4km")

    def after_instantiate_classes(self) -> None:
        self.datamodule.set_transforms(self.model.transforms)

        compile = self.config[self.config["subcommand"]]["compile"]
        if compile and _is_compile_supported():
            self.model: LightningModule = torch.compile(self.model)  # type: ignore

    def _add_instantiators(self):
        # temporary patch until https://github.com/Lightning-AI/pytorch-lightning/issues/20311 is fixed
        pass

    def before_fit(self):
        config = self.config[self.config["subcommand"]]
        if config["find_lr"]:
            tuner = Tuner(self.trainer)
            lr_finder = tuner.lr_find(
                self.model,
                self.datamodule,
                min_lr=1e-6,
                max_lr=1e-2,
            )
            fig = lr_finder.plot(suggest=True)
            fig.savefig(f"runs/{config['run_name']}/lr.png")
            print("[LRFinder]: Suggested LR", lr_finder.suggestion())


def _is_compile_supported():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (7, 0)


def _register_classes():
    # To select classes from the CLI using only the class name instead of the full path
    from dtg.modules.dtg.dtg import DtGModule  # noqa: F401


def cli():
    _register_classes()
    CLI(
        model_class=LightningModule,
        datamodule_class=DtGDataModule,
        trainer_class=DtGTrainer,
        seed_everything_default=1337,
        save_config_callback=SaveConfigCallback,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
    )
