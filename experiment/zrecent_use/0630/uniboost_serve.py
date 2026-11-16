#!/usr/bin/env python3
"""[WT] `vllm serve` launcher that injects the UniBoost baseline scheduler.

The two baselines must stay isolated: stock vLLM is one of them, so it must
not contain any UniBoost code or code path. The UniBoost arm therefore runs
the *same* stock vLLM with the scheduler class injected at runtime here, by
wrapping EngineArgs.create_engine_config. When this launcher is not used,
`vllm serve` behaves exactly as before.

Usage (identical arguments to `vllm serve`):

    python uniboost_serve.py serve --model /path/to/model --port 22020 ...

Environment: WT_UNIBOOST_* (see vllm/v1/core/sched/uniboost_scheduler.py).
"""

import sys

from vllm.engine.arg_utils import EngineArgs

_ORIGINAL_CREATE_ENGINE_CONFIG = EngineArgs.create_engine_config

UNIBOOST_SCHEDULER_CLS = (
    "vllm.v1.core.sched.uniboost_scheduler.UniBoostScheduler")


def _create_engine_config_with_uniboost(self, *args, **kwargs):
    config = _ORIGINAL_CREATE_ENGINE_CONFIG(self, *args, **kwargs)
    scheduler_config = config.scheduler_config
    if scheduler_config.scheduler_cls not in (
            "vllm.v1.core.sched.scheduler.Scheduler", None):
        raise SystemExit(
            "[WT] refusing to override an already selected scheduler class: "
            f"{scheduler_config.scheduler_cls!r}. Use UniBoost alone, not "
            "together with --geometric-window-scheduler/--pastfuture-scheduler.")
    scheduler_config.scheduler_cls = UNIBOOST_SCHEDULER_CLS
    return config


def main() -> None:
    EngineArgs.create_engine_config = _create_engine_config_with_uniboost
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.exit(vllm_main())


if __name__ == "__main__":
    main()
