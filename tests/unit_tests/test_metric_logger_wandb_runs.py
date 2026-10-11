# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression check for independent W&B main and worker runs."""

import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from rlinf.utils.metric_logger import MetricLogger


def test_wandb_runs_are_independent() -> None:
    """Worker creation cannot finish or redirect the main run's metrics."""
    runs = [Mock(), Mock(), Mock()]
    wandb = SimpleNamespace(init=Mock(side_effect=runs), Table=Mock())
    with tempfile.TemporaryDirectory() as directory:
        cfg = OmegaConf.create(
            {
                "runner": {
                    "per_worker_log": True,
                    "logger": {
                        "log_path": directory,
                        "logger_backends": ["wandb"],
                    },
                }
            }
        )
        with patch.dict(sys.modules, {"wandb": wandb}):
            logger = MetricLogger(cfg)
            logger.log({"loss": 1}, 0)
            logger.log({"worker": 0}, 0, worker_group_name="Actor", rank=0)
            logger.log({"worker": 1}, 0, worker_group_name="Actor", rank=1)
            logger.log({"loss": 2}, 1)
            assert runs[0].log.call_count == 2
            runs[1].log.assert_called_once_with(data={"worker": 0}, step=0)
            runs[2].log.assert_called_once_with(data={"worker": 1}, step=0)
            for call in wandb.init.call_args_list:
                assert call.kwargs["reinit"] == "create_new"
            for run in runs:
                run.finish.assert_not_called()
            logger.log_table("dataframe", "table", 1)
            wandb.Table.assert_called_once_with(dataframe="dataframe")
            logger.finish()
            logger.finish()  # Explicit shutdown followed by __del__ is harmless.
            for run in runs:
                run.finish.assert_called_once()


if __name__ == "__main__":
    test_wandb_runs_are_independent()
    print("Independent W&B run check passed")
