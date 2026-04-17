"""
Draw a single PSLD posterior sample and save it as a .pt tensor.
Called once per SLURM array task by run_psld_array.sbatch.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import matplotlib.pyplot as plt

from ldm.util import instantiate_from_config
from ldm.models.diffusion.psld import DDIMSampler


def load_model(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    pl_sd  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd     = pl_sd["state_dict"]
    model  = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx",  type=int, required=True,
                        help="Array task ID; determines output filename.")
    parser.add_argument("--save_dir",    type=str, required=True,
                        help="Root directory for this run's outputs.")
    parser.add_argument("--file_path",   type=str, required=True)
    parser.add_argument("--task_config", type=str,
                        default="configs/center_inpainting_config_psld.yaml")
    parser.add_argument("--dps_path",    type=str,
                        default="../diffusion-posterior-sampling/")
    parser.add_argument("--config",      type=str,
                        default="configs/stable-diffusion/v1-inference.yaml")
    parser.add_argument("--ckpt",        type=str,
                        default="/lustre/fswork/projects/rech/ynx/uxl64xr/models/sd15/v1-5-pruned-emaonly.ckpt")
    parser.add_argument("--ddim_steps",  type=int,   default=1000)
    parser.add_argument("--ddim_eta",    type=float, default=0.0)
    parser.add_argument("--scale",       type=float, default=7.5)
    parser.add_argument("--gamma",       type=float, default=0.01)
    parser.add_argument("--omega",       type=float, default=0.5)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--H",           type=int,   default=512)
    parser.add_argument("--C",           type=int,   default=4)
    parser.add_argument("--f",           type=int,   default=8)
    parser.add_argument("--precision",   type=str,   default="autocast",
                        choices=["full", "autocast"])
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device  = torch.device("cuda")
    model   = load_model(opt.config, opt.ckpt).to(device)
    sampler = DDIMSampler(model)

    samples_dir = os.path.join(opt.save_dir, "samples")
    shared_dir  = os.path.join(opt.save_dir, "shared")
    for d in (samples_dir, shared_dir):
        os.makedirs(d, exist_ok=True)

    # DPS setup
    sys.path.append(opt.dps_path)
    import yaml
    from guided_diffusion.measurements import get_noise, get_operator
    from util.img_utils import mask_generator

    with open(opt.dps_path + opt.task_config) as fh:
        task_config = yaml.load(fh, Loader=yaml.FullLoader)

    measure_config = task_config["measurement"]
    operator = get_operator(device=device, **measure_config["operator"])
    noiser   = get_noise(**measure_config["noise"])
    mask_gen = mask_generator(**measure_config["mask_opt"])

    # Load image → [1,C,H,W] in [-1,1]
    raw = plt.imread(opt.file_path).astype(np.float32)
    raw = (raw - raw.min()) / (raw.max() - raw.min())
    img = torch.from_numpy(raw).unsqueeze(0).permute(0, 3, 1, 2)[:, :3].to(device)
    img = F.interpolate(img, opt.H)
    org_image = ((img - 0.5) / 0.5)  # [1,C,H,W] in [-1,1]

    mask = mask_gen(org_image)          # [1,C,H,W] in {0,1}
    mask = mask[:, :1, :, :]           # [1,1,H,W]

    y   = operator.forward(org_image, mask=mask)
    y_n = noiser(y)                    # [1,C,H,W] in [-1,1]

    # Task 0 writes the shared tensors (gt, measurement, mask).
    # All tasks use the same seed so these are identical across tasks.
    if opt.sample_idx == 0:
        torch.save(org_image[0].cpu(), os.path.join(shared_dir, "gt.pt"))
        torch.save(y_n[0].cpu(),       os.path.join(shared_dir, "yn.pt"))
        torch.save(mask[0].cpu(),      os.path.join(shared_dir, "mask.pt"))
        print("Task 0: saved shared tensors (gt, yn, mask).")

    # Sample
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            uc = model.get_learned_conditioning([""]) if opt.scale != 1.0 else None
            c  = model.get_learned_conditioning([""])
            shape = [opt.C, opt.H // opt.f, opt.H // opt.f]

            s_ddim, _ = sampler.sample(
                S=opt.ddim_steps,
                conditioning=c,
                batch_size=1,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                unconditional_conditioning=uc,
                eta=opt.ddim_eta,
                x_T=None,        # fresh random start → iid sample
                ip_mask=mask,
                measurements=y_n,
                operator=operator,
                gamma=opt.gamma,
                inpainting=1,
                omega=opt.omega,
                general_inverse=0,
                noiser=noiser,
            )

    x0 = model.decode_first_stage(s_ddim)  # [1,C,H,W] in ~[-1,1]
    out_path = os.path.join(samples_dir, f"sample_{opt.sample_idx:03d}.pt")
    torch.save(x0[0].detach().cpu(), out_path)
    print(f"Saved sample {opt.sample_idx:03d} → {out_path}")


if __name__ == "__main__":
    main()
