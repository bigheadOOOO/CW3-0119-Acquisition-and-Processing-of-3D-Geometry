import yaml, torch
import argparse
import os

class TaskAddress:
    def __init__(self):
        self.meshplypath = "data"
        self.outputpath = "results"
        self.configpath = "experiments"
        self.reconstruct_sourcepath = "results"
        self.reconstruct_resultpath = "results"

def get_args(object: str):
    args = TaskAddress()
    args.meshplypath += f"/{object}_curvs.ply"
    args.outputpath += f"/{object}_curvs"
    args.configpath += f"/{object}_curvature.yaml"
    args.reconstruct_sourcepath += f"/{object}_curvs/best.pth"
    args.reconstruct_resultpath += f"/{object}_curvs/best.ply"
    return args

def load_config(args, sampling_mode = "curvature"):
    if not os.path.exists(args.configpath):
        raise FileNotFoundError(
            f"Experiment configuration file \"{args.configpath}\" not found."
        )

    if not os.path.exists(args.meshplypath):
        raise FileNotFoundError(
            f"Mesh file \"{args.meshplypath}\" not found."
        )

    with open(args.configpath, "r") as fin:
        config = yaml.safe_load(fin)

    print(f"Saving results in {args.outputpath}")
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # GPU or CPU
    devstr = "cuda:0"
    if "cuda" in devstr and not torch.cuda.is_available():
        devstr = "cpu"
        print("No CUDA available devices found on system. Using CPU.")
    config["device"] = torch.device(devstr)

    # sampling mode
    config["use_curv_sampling"] = False
    if "sampling" not in config:
        config["sampling"] = {"type": "uniform"}
    elif config["sampling"]["type"] == "curvature":
        print("select curvature sampling")
        config["use_curv_sampling"] = True

    if sampling_mode == "curvature":
        config["use_curv_sampling"] = True
        config["sampling"]["type"] = "curvature"


    if config["use_curv_sampling"]:
        config["sampling"]["curv_fractions"] = config["sampling"].get(
            "curv_fractions", [0.2, 0.6, 0.2]
        )
        config["sampling"]["curv_percentiles"] = config["sampling"].get(
            "curv_percentiles", [0.7, 0.95]
        )
    else:
        config["sampling"]["curv_fractions"] = []
        config["sampling"]["curv_percentiles"] = []

    return config