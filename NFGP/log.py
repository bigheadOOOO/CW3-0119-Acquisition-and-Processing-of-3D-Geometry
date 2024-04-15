from vis_utils import imf2mesh
import os

def log_train(train_info, cfg, train_data, decoder, num_update_step, writer=None,
              step=None, epoch=None, visualize=False, **kwargs):
    if writer is None:
        return
    writer_step = step if step is not None else epoch

    # Log training information to tensorboard
    train_info = {k: (v.cpu() if not isinstance(v, float) else v)
                  for k, v in train_info.items()}
    for k, v in train_info.items():
        ktype = k.split("/")[0]
        kstr = "/".join(k.split("/")[1:])
        if ktype == 'scalar':
            writer.add_scalar(kstr, v, writer_step)


    if visualize:
        # NOTE: trainer sub class should implement this function
        visualize(train_info, cfg, train_data, decoder, step=step,
                       epoch=epoch, visualize=visualize)

def visualize(train_info, cfg, train_data, decoder, num_update_step,
              step=None, epoch=None, visualize=False, ):
    print("Visualize: %s" % step)
    res = int(getattr(cfg, "vis_mc_res", 256))
    thr = float(getattr(cfg, "vis_mc_thr", 0.))
    mesh = imf2mesh(lambda x: decoder(x), res=res, threshold=thr)
    if mesh is not None:
        save_name = "mesh_%diters.obj" % num_update_step
        mesh.export(os.path.join(cfg["save_dir"], "val", save_name))
        mesh.export(os.path.join(cfg["save_dir"], "latest_mesh.obj"))
