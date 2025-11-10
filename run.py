import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from unilat3d.pipelines import UniLatImageTo3DPipeline
from unilat3d.utils import render_utils, postprocessing_utils
from time import time
import argparse
# Load a pipeline from a model folder or a Hugging Face model hub.


# Load an image
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="assets/images")
parser.add_argument('--output_dir', type=str, default="demo_output/")
# parser.add_argument("--pipeline_path", type=str, default="SmallGuanjun/UniLat3D")
parser.add_argument("--pipeline_path", type=str, default="/data_hdd3/users/yangchen/trellis/guanjunwu/UniLat_ckpts/ckpts") # cy: remove this later
parser.add_argument("--resolution",type=int,default=1024)
parser.add_argument("--num_samples",type=int,default=-1)
parser.add_argument("--cfg_strength",type=float,default=5)
parser.add_argument("--simplify",type=float,default=0.95)
parser.add_argument("--texture_size",type=float,default=1024)

parser.add_argument("--steps",type=int,default=50)
parser.add_argument("--pipeline_name",type=str, default="ours")

parser.add_argument('--formats', action='append', help='return formats, can be gaussian and mesh', default=['gaussian'])
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--save_mp4',  action="store_true")


args = parser.parse_args()

pipeline = UniLatImageTo3DPipeline.from_pretrained(args.pipeline_path)
pipeline.cuda()
# Load an image
input_dir = args.input_dir
output_dir = args.output_dir
image_paths = os.listdir(input_dir)
os.makedirs(output_dir,exist_ok=True)
os.makedirs(os.path.join(f"{output_dir}/gaussian_video/"),exist_ok=True)
os.makedirs(os.path.join(f"{output_dir}/gaussian_gif/"),exist_ok=True)
save_mesh = True
image_paths.sort()
if args.num_samples != -1:
    image_paths = image_paths[:args.num_samples]

for i in image_paths:
    if "png" not in i and "jpg" not in i:
        continue

    image = Image.open(os.path.join(input_dir, i))
    print("current name:", os.path.join(input_dir, i))
    outputs = pipeline.run(
        image,
        seed=1,
        # Optional parameters
        unilat_sampler_params={
            "steps": args.steps,
            "cfg_strength": args.cfg_strength,
        },
        formats=args.formats
    )

    if args.save_mp4:
        video = render_utils.render_video(outputs['gaussian'][0], bg_color=(1, 1, 1), num_frames=300,resolution=args.resolution)['color']
        imageio.mimsave(f"{output_dir}/gaussian_video/{i.split('.')[0]}.mp4", video, fps=60)
        imageio.mimsave(f"{output_dir}/gaussian_gif/{i.split('.')[0]}.gif", video, fps=60)

    if 'mesh' in args.formats:
        mesh = outputs['mesh'][0]
        mesh.export(f"{output_dir}/mesh/{i.split('.')[0]}.ply")

    break # cy: remove this later



