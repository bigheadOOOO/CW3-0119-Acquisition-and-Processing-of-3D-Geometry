#!/usr/bin/env python
# coding: utf-8

"""
Experiments with calculating the SDF for a batch of points and reusing it for N
iterations.
"""

import argparse
import copy
import os
import os.path as osp
import time
import yaml
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from i3d.dataset import PointCloudDeferredSampling
from i3d.loss_functions import loss_true_sdf
from i3d.model import SIREN

# TODO: NFGP IMPORT
from NFGP import single_shape_sdf_datasets, igp_wrapper, siren_mlp, log
from i3d import loss_functions
from torch.utils.tensorboard import SummaryWriter

#TODO: NFGP END

def get_args(): 
    parser = argparse.ArgumentParser(
        description="Experiments with SDF querying at regular intervals."
    )
    # Define groups
    required_parser = parser.add_argument_group('required arguments')
    optional_parser = parser.add_argument_group('optional arguments')
    siren_parser = parser.add_argument_group('SIREN parameters')
    sampling_parser = parser.add_argument_group('Sampling parameters')
    # Define parameter lists
    required_args = ["meshplypath", "outputpath", "configpath"]
    # Add mandatory parameters of string type
    for arg in required_args:
        required_parser.add_argument(arg, help=arg)
    # Add optional parameters
    optional_parser.add_argument(
        "--device", "-d", type=str, default="cuda:0",
        help="The device to perform the training on. Uses CUDA:0 by default."
    )
    optional_parser.add_argument(
        "--nepochs", "-n", type=int, default=0,
        help="Number of training epochs for each mesh."
    )
    optional_parser.add_argument(
        "--batchsize", "-b", type=int, default=0,
        help="# of points to fetch per iteration. By default, uses the # of"
        " mesh vertices."
    )
    optional_parser.add_argument(
        "--resample-sdf-at", "-r", type=int, default=0,
        help="Recalculates the SDF for off-surface points at every N epochs."
        " By default (0) we calculate the SDF at every iteration."
    )
    optional_parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed to use."
    )
    # Add SIREN parameters
    siren_parser.add_argument(
        "--omega0", "-o", type=int, default=0,
        help="SIREN Omega 0 parameter."
    )
    siren_parser.add_argument(
        "--omegaW", "-w", type=int, default=0,
        help="SIREN Omega 0 parameter for hidden layers."
    )
    siren_parser.add_argument(
        "--hidden-layer-config", type=int, nargs='+', default=[],
        help="SIREN neurons per layer. By default we fetch it from the"
        " configuration file."
    )
    # Add Sampling parameters
    sampling_parser.add_argument(
        "--sampling", "-s", type=str, default="uniform",
        help="Uniform (\"uniform\", default value) or curvature-based"
        " (\"curvature\") sampling."
    )
    sampling_parser.add_argument(
        "--curvature-fractions", type=float, nargs='+', default=[],
        help="Fractions of data to fetch for each curvature bin. Only used"
        " with \"--sampling curvature\" argument, or when sampling type is"
        " \"curvature\" in the configuration file."
    )
    sampling_parser.add_argument(
        "--curvature-percentiles", type=float, nargs='+', default=[],
        help="The curvature percentiles to use when defining the bins. Only"
        " used with \"--sampling curvature\" argument, or when sampling type"
        " is \"curvature\" in the configuration file."
    )
    # NFGP
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")
    return parser.parse_args()


if __name__ == "__main__":
    # The generated convolutional algorithm will remain consistent every time the same code is run, ensuring consistency in the results
    torch.backends.cudnn.deterministic = True
    # Ensure consistent results between different runs.
    torch.backends.cudnn.benchmark = False

    args = get_args()
    
    if not osp.exists(args.configpath):
        raise FileNotFoundError(
            f"Experiment configuration file \"{args.configpath}\" not found."
        )

    if not osp.exists(args.meshplypath):
        raise FileNotFoundError(
            f"Mesh file \"{args.meshplypath}\" not found."
        )

    with open(args.configpath, "r") as fin:
        config = yaml.safe_load(fin)

    print(f"Saving results in {args.outputpath}")
    if not osp.exists(args.outputpath):
        os.makedirs(args.outputpath)

    seed = args.seed if args.seed else config["training"].get("seed", 668123)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    config["training"]["seed"] = seed

    trainingcfg = config["training"]
    EPOCHS = trainingcfg.get("epochs", 100)
    if args.nepochs:
        EPOCHS = args.nepochs
        config["training"]["epochs"] = args.nepochs

    BATCH = trainingcfg.get("batchsize", 0)
    if args.batchsize:
        BATCH = args.batchsize
        config["training"]["batchsize"] = args.batchsize

    REFRESH_SDF_AT_PERC_STEPS = trainingcfg.get("resample_sdf_at", 1)
    if args.resample_sdf_at:
        REFRESH_SDF_AT_PERC_STEPS = args.resample_sdf_at
        config["training"]["resample_sdf_at"] = args.resample_sdf_at

    REFRESH_SDF_AT_PERC_STEPS /= EPOCHS

    devstr = args.device
    if "cuda" in devstr and not torch.cuda.is_available():
        devstr = "cpu"
        print("No CUDA available devices found on system. Using CPU.")

    device = torch.device(devstr)

    withcurvature = False
    if "sampling" not in config:
        config["sampling"] = {"type": "uniform"}
    elif config["sampling"]["type"] == "curvature":
        withcurvature = True

    if args.sampling == "curvature":
        withcurvature = True
        config["sampling"]["type"] = "curvature"

    curv_fractions = []
    curv_percentiles = []
    if withcurvature:
        curv_fractions = config["sampling"].get(
            "curv_fractions", [0.2, 0.6, 0.2]
        )
        curv_percentiles = config["sampling"].get(
            "curv_percentiles", [0.7, 0.95]
        )
        if args.curvature_fractions:
            curv_fractions = [float(f) for f in args.curvature_fractions]
            config["sampling"]["curv_fractions"] = curv_fractions
        if args.curvature_percentiles:
            curv_percentiles = \
                [float(p) for p in args.curvature_percentiles]
            config["sampling"]["curv_percentiles"] = curv_percentiles

    # TODO: NFGP BEGIN
    start_epoch = 0
    start_time = time.time()

    boundary_loss_weight = float(getattr(
        config["NFGP"], "boundary_weight", 1.))
    boundary_loss_num_points = int(getattr(
        config["NFGP"], "boundary_num_points", 0))
    boundary_loss_points_update_step = int(getattr(
        config["NFGP"], "boundary_loss_points_update_step", 1))
    boundary_loss_use_surf_points = int(getattr(
        config["NFGP"], "boundary_loss_use_surf_points", True))
    lap_loss_weight = float(getattr(
        config["NFGP"], "lap_loss_weight", 1e-4))
    lap_loss_threshold = int(getattr(
        config["NFGP"], "lap_loss_threshold", 50))
    lap_loss_num_points = int(getattr(
        config["NFGP"], "lap_loss_num_points", 5000))
    grad_norm_weight = float(getattr(
        config["NFGP"], "grad_norm_weight", 1e-2))
    grad_norm_num_points = int(getattr(
        config["NFGP"], "grad_norm_num_points", 5000))
    beta = float(getattr(
        config["NFGP"], "beta", 1e-2))

    original_decoder = siren_mlp.Net(config, config["NFGP"]["models"]["decoder"])
    original_decoder.cuda()
    original_decoder.load_state_dict(
        torch.load(config["NFGP"]["models"]["decoder"]["path"])['net'])
    print("Original Decoder:")
    print(original_decoder)
    wrapper_type = getattr(
        config["NFGP"], "wrapper_type", "distillation")

    if not hasattr(config["NFGP"]["models"], "net"):
            config["NFGP"]["models"]["net"] = config["NFGP"]["models"]["decoder"]

    if wrapper_type in ['distillation']:
        decoder, opt_dec, scheduler_dec = igp_wrapper.distillation(
            config, original_decoder,
            reload=getattr(config["NFGP"], "reload_decoder", True))
    writer = SummaryWriter(log_dir=getattr(
        config["NFGP"], "log_name", None))

    # TODO: NFGP END

    dataset = PointCloudDeferredSampling(
        args.meshplypath, config, batch_size=BATCH, use_curvature=withcurvature,
        device=device, curv_fractions=curv_fractions,
        curv_percentiles=curv_percentiles
    )
    N = dataset.vertices.shape[0]

    # Fetching batch_size again since we may have passed 0, meaning that we
    # will use all mesh vertices at each iteration.
    BATCH = dataset.batch_size
    nsteps = round(EPOCHS * (2 * N / BATCH))
    warmup_steps = nsteps // 10
    resample_sdf_nsteps = max(1, round(REFRESH_SDF_AT_PERC_STEPS * nsteps))
    print(f"Resampling SDF at every {resample_sdf_nsteps} training steps")
    print(f"Total # of training steps = {nsteps}")

    netcfg = config["network"]
    hidden_layer_config = netcfg["hidden_layers"]
    if args.hidden_layer_config:
        hidden_layer_config = [int(n) for n in args.hidden_layer_config]
        config["network"]["hidden_layers"] = hidden_layer_config

    # Create the model and optimizer
    model = SIREN(
        netcfg["in_coords"],
        netcfg["out_coords"],
        hidden_layer_config=hidden_layer_config,
        w0=netcfg["omega_0"] if not args.omega0 else args.omega0,
        ww=netcfg["omega_w"] if not args.omegaW else args.omegaW
    ).to(device)
    print(model)
    print("# parameters =", parameters_to_vector(model.parameters()).numel())
    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
    training_loss = {}

    best_loss = torch.inf
    best_weights = None
    best_step = warmup_steps

    config["network"]["omega_0"] = model.w0
    config["network"]["omega_w"] = model.ww

    with open(osp.join(args.outputpath, "config.yaml"), 'w') as fout:
        yaml.dump(config, fout)

    # Training loop
    start_training_time = time.time()
    for step in range(nsteps):
        # We will recalculate the SDF points at this # of steps
        if not step % resample_sdf_nsteps:
            dataset.refresh_sdf = True

        samples = dataset[0]
        gt = samples[1]

        optim.zero_grad(set_to_none=True)
        # train
        y = model(samples[0]["coords"])
        loss = loss_true_sdf(y, gt)

        running_loss = torch.zeros((1, 1), device=device)
        for k, v in loss.items():
            running_loss += v
            if k not in training_loss:
                training_loss[k] = [v.detach()]
            else:
                training_loss[k].append(v.detach())

        running_loss.backward()
        optim.step()

        if step > warmup_steps and running_loss.item() < best_loss:
            best_step = step
            best_weights = copy.deepcopy(model.state_dict())
            best_loss = running_loss.item()

        if not step % 100 and step > 0:
            print(f"Step {step} --- Loss {running_loss.item()}")

    # TODO: NFGP, UPDATE LOSS FOR SMOOTHING
    sdf_data = y
    loaders = single_shape_sdf_datasets.get_data_loaders(config["data"], args)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    for epoch in range(start_epoch, config["NFGP"]["epochs"] + start_epoch):
        # train for one epoch
        print(epoch)
        loader_start = time.time()
        for bidx, data in enumerate(train_loader):
            print(bidx)
            loader_duration = time.time() - loader_start
            step = bidx + len(train_loader) * epoch + 1
            logs_info = loss_functions.smoothing_loss(original_decoder, decoder, opt_dec, beta,
                   num_update_step=epoch, boundary_loss_weight=boundary_loss_weight,
                   boundary_loss_num_points=boundary_loss_num_points,
                   boundary_loss_points_update_step=boundary_loss_points_update_step,
                   boundary_loss_use_surf_points=boundary_loss_use_surf_points,
                   grad_norm_weight=grad_norm_weight,
                   grad_norm_num_points=grad_norm_num_points,
                   lap_loss_weight=lap_loss_weight,
                   lap_loss_threshold=lap_loss_threshold,
                   lap_loss_num_points=lap_loss_num_points
                   )
            # if step % int(cfg.viz.log_freq) == 0 and int(cfg.viz.log_freq) > 0:
            duration = time.time() - start_time
            start_time = time.time()
            print("Epoch %d Batch [%2d/%2d] "
                  " Loss %2.5f"
                  % (epoch, bidx, len(train_loader), logs_info['loss']))
            visualize = step % int(config["viz"]["viz_freq"]) == 0 and \
                        int(config["viz"]["viz_freq"]) > 0
            log.log_train(
                logs_info, config["NFGP"], data, decoder, epoch,
                writer=writer, epoch=epoch, step=step, visualize=visualize)

    #         # Reset loader time
            loader_start = time.time()
    #
        # Save first so that even if the visualization bugged,
        # we still have something
        # if (epoch + 1) % int(config["NFGP"]["viz"]["save_freq"]) == 0 and \
                # int(config["NFGP"]["viz"]["save_freq"]) > 0:
            # trainer.save(epoch=epoch, step=step)
    #
    #     if (epoch + 1) % int(cfg.viz.val_freq) == 0 and \
    #             int(cfg.viz.val_freq) > 0:
    #         val_info = trainer.validate(test_loader, epoch=epoch)
    #         trainer.log_val(val_info, writer=writer, epoch=epoch)
    #
    #     # Signal the trainer to cleanup now that an epoch has ended
    #     trainer.epoch_end(epoch, writer=writer)
    # trainer.save(epoch=epoch, step=step)
    # # TODO: NFGP END
    training_time = time.time() - start_training_time
    print(f"Training took {training_time} s")
    print(f"Best loss value {best_loss} at step {best_step}")
    torch.save(
        model.state_dict(), osp.join(args.outputpath, "weights_with_w0.pth")
    )
    model.update_omegas(w0=1, ww=None)
    torch.save(
        model.state_dict(), osp.join(args.outputpath, "weights.pth")
    )
    torch.save(
        best_weights, osp.join(args.outputpath, "best_with_w0.pth")
    )

    model.w0 = netcfg["omega_0"] if not args.omega0 else args.omega0
    model.ww = netcfg["omega_w"] if not args.omegaW else args.omegaW
    model.load_state_dict(best_weights)
    model.update_omegas(w0=1, ww=None)
    torch.save(
        model.state_dict(), osp.join(args.outputpath, "best.pth")
    )
