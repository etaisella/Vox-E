from pathlib import Path
import click
import imageio
import torch

from thre3d_atom.thre3d_reprs.sd_attn import StableDiffusion
from thre3d_atom.modules.volumetric_model import (
    create_volumetric_model_from_saved_model, create_volumetric_model_from_saved_model_attn
)
from thre3d_atom.thre3d_reprs.voxels import create_voxel_grid_from_saved_info_dict, \
    create_voxel_grid_from_saved_info_dict_attn
from thre3d_atom.utils.constants import HEMISPHERICAL_RADIUS, CAMERA_INTRINSICS
from thre3d_atom.utils.imaging_utils import (
    get_thre360_animation_poses,
    get_thre360_spiral_animation_poses,
)
from thre3d_atom.visualizations.animations import (
    render_camera_path_for_volumetric_model, render_camera_path_for_volumetric_model_with_attention_and_diffusion,
    render_camera_path_for_volumetric_model_attn, render_camera_path_for_volumetric_model_attn_blend
)
import numpy as np
from easydict import EasyDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------------------
#  Command line configuration for the script                                          |
# -----------------------------------------------------------------------c--------------
# fmt: off
# noinspection PyUnresolvedReferences
@click.command()
# Required arguments:
@click.option("-i1", "--first_model_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the trained (reconstructed) model")
@click.option("-i2", "--second_model_path", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the trained (reconstructed) model")
@click.option("-o", "--output_path", type=click.Path(file_okay=False, dir_okay=True),
              required=True, help="path for saving rendered output")
# Non-required Render configuration options:
@click.option("--overridden_num_samples_per_ray", type=click.IntRange(min=1), default=512,
              required=False, help="overridden (increased) num_samples_per_ray for beautiful renders :)")
@click.option("--render_scale_factor", type=click.FLOAT, default=2.0,
              required=False, help="overridden (increased) resolution (again :D) for beautiful renders :)")
@click.option("--camera_path", type=click.Choice(["thre360", "spiral"]), default="thre360",
              required=False, help="which camera path to use for rendering the animation")
# thre360_path options
@click.option("--camera_pitch", type=click.FLOAT, default=60.0,
              required=False, help="pitch-angle value for the camera for 360 path animation")
@click.option("--num_frames", type=click.IntRange(min=1), default=180,
              required=False, help="number of frames in the video")
# spiral path options
@click.option("--vertical_camera_height", type=click.FLOAT, default=3.0,
              required=False, help="height at which the camera spiralling will happen")
@click.option("--num_spiral_rounds", type=click.IntRange(min=1), default=2,
              required=False, help="number of rounds made while transitioning between spiral radii")
# Non-required video options:
@click.option("--fps", type=click.IntRange(min=1), default=60,
              required=False, help="frames per second of the video")
@click.option("--timestamp", type=click.INT, default=0,
              required=False, help="diffusion_timestamp")
@click.option("--use_sd", type=click.BOOL, default=False,
              required=False, help="render with stable diffusion")
@click.option("--load_attention", type=click.BOOL, default=False,
              required=False, help="render with attention features")
@click.option("--prompt", type=click.STRING, required=False, default='',
              help="prompt for attention focus")
@click.option("--index_to_attn", type=click.INT, required=False, default=11,
              help="index to apply attention to", show_default=True)
# fmt: on
# -------------------------------------------------------------------------------------
def main(**kwargs) -> None:
    # load the requested configuration for the training
    config = EasyDict(kwargs)

    # parse os-checked path-strings into Pathlike Paths :)
    first_model_path = Path(config.first_model_path)
    second_model_path = Path(config.second_model_path)
    output_path = Path(config.output_path)
    # create the output path if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)

    first_vol_mod, extra_info = create_volumetric_model_from_saved_model_attn(
        model_path=first_model_path,
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict_attn,
        device=device, load_attn=config.load_attention
    )
    second_vol_mod, extra_info = create_volumetric_model_from_saved_model_attn(
        model_path=second_model_path,
        thre3d_repr_creator=create_voxel_grid_from_saved_info_dict_attn,
        device=device, load_attn=config.load_attention
    )

    attn = first_vol_mod.thre3d_repr.attn.detach() - second_vol_mod.thre3d_repr.attn.detach()

    #####################################
    import numpy as np
    import imcut.pycut as pspc
    import matplotlib.pyplot as plt

    # create data
    data = attn.detach().cpu().numpy().squeeze(-1)
        # Make seeds
    seeds = np.zeros(data.shape)
    seeds[data >= 0] = 2
    seeds[data<0] = 1
    # max_ind = np.unravel_index(data.argmax(), data.shape)
    # maxes = [max_ind, (max_ind[0] +1, max_ind[1], max_ind[2])]
    # min_ind = np.unravel_index(data.argmin(), data.shape)
    # mins = [min_ind, (min_ind[0] + 1, min_ind[1], min_ind[2])]
    # seeds[maxes] = 2
    # seeds[mins] = 1

    # Run
    igc = pspc.ImageGraphCut(data, voxelsize=[1, 1, 1])
    igc.set_seeds(seeds)
    igc.run()
    first_vol_mod.thre3d_repr.attn = torch.nn.Parameter(torch.tensor(igc.segmentation.astype(np.float32)).unsqueeze(-1).to(device))
    hemispherical_radius = extra_info[HEMISPHERICAL_RADIUS]
    camera_intrinsics = extra_info[CAMERA_INTRINSICS]

    # generate animation using the newly_created vol_mod :)
    if config.camera_path == "thre360":
        camera_pitch, num_frames = config.camera_pitch, config.num_frames
        animation_poses = get_thre360_animation_poses(
            hemispherical_radius=hemispherical_radius,
            camera_pitch=camera_pitch,
            num_poses=num_frames,
        )
    elif config.camera_path == "spiral":
        vertical_camera_height, num_frames = (
            config.vertical_camera_height,
            config.num_frames,
        )
        animation_poses = get_thre360_spiral_animation_poses(
            horizontal_radius_range=(hemispherical_radius / 8.0, hemispherical_radius),
            vertical_camera_height=vertical_camera_height,
            num_rounds=config.num_spiral_rounds,
            num_poses=num_frames,
        )
    else:
        raise ValueError(
            f"Unknown camera_path ``{config.camera_path}'' requested."
            f"Only available options are: ['thre360' and 'spiral']"
        )
    animation_frames, attn = render_camera_path_for_volumetric_model_attn_blend(
        first_vol_mod,
        camera_path=animation_poses,
        camera_intrinsics=camera_intrinsics,
        overridden_num_samples_per_ray=config.overridden_num_samples_per_ray,
        render_scale_factor=config.render_scale_factor,
    )
    imageio.mimwrite(
            output_path / "rendered_video.mp4",
            animation_frames,
            fps=config.fps,
        )
if __name__ == "__main__":
    main()
