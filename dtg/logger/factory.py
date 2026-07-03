from typing import Literal

from lightning.pytorch.loggers import Logger

LOGGERS = Literal["clearml"]


def create_logger(
    logger_type: LOGGERS | None,
    root_dir: str,
    run_name: str,
    project: str,
    task_type: str,
    offline: bool,
) -> Logger | None:
    if logger_type is None:
        return None

    match logger_type:
        case "clearml":
            from dtg.logger.clearml import ClearMlLogger

            return ClearMlLogger(
                save_dir=root_dir,
                name=run_name,
                project=project,
                task_type=task_type,
                offline=offline,
            )
