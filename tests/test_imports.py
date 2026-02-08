import importlib

MODULES = [
    "src.main_sim",
    "src.station",
    "src.controllers.baseline_controller",
    "src.envs.drake_gym_env",
    "src.train_rl",
    "src.utils.paths",
    "src.utils.randomization",
]


def main() -> int:
    for name in MODULES:
        importlib.import_module(name)
    print("imports ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
