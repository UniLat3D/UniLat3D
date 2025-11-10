from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class UniLatImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        unilat_sampler (samplers.Sampler): The sampler for the sparse structure.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        unilat_sampler: samplers.Sampler = None,
        image_cond_model: str = None,
        unilat_normalization: dict = None
    ):
        if models is None:
            return
        super().__init__(models)
        self.unilat_sampler = unilat_sampler
        self.unilat_sampler_params = {}

        self.unilat_normalization = unilat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "UniLatImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(UniLatImageTo3DPipeline, UniLatImageTo3DPipeline).from_pretrained(path)
        new_pipeline = UniLatImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.unilat_sampler = getattr(samplers, args['unilat_sampler']['name'])(**args['unilat_sampler']['args'])
        new_pipeline.unilat_sampler_params = args['unilat_sampler']['params']


        new_pipeline.unilat_normalization = args['unilat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        
        if "dinov3" in name:
            self.dino_github = "external/dinov3"
            self.dino_weight = "external/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
            self.image_resolution = 592
            dino_model = torch.hub.load(self.dino_github, name, source = "local", weights = self.dino_weight, pretrained=True)
        elif "dinov2" in name:
            self.dino_github = "dinov2"
            self.image_resolution = 518
            dino_model = torch.hub.load(self.dino_github, name, source="local", pretrained=True)
        dino_model.eval()
        self.models['image_cond_model'] = dino_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((self.image_resolution, self.image_resolution), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_resolution, self.image_resolution), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.unilat_sampler_params, **sampler_params}
        z_s = self.unilat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )
        z_sample = z_s.samples
        pred_x_t = z_s["pred_x_t"]
        


        return z_sample, pred_x_t

    def decode_unilat(
        self,
        unilat: torch.Tensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the UniLat

        Args:
            unilat (torch.Tensor): The UniLat
            formats (List[str]): The formats to decode the UniLat latent to.

        Returns:
            dict: The decoded UniLat latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['unilat_decoder_mesh'].decode_mesh(self.models['unilat_decoder_gs'].compute_slats(unilat))
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['unilat_decoder_gs'](unilat)

        return ret


    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        unilat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        sample: bool = False
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            unilat_sampler_params (dict): Additional parameters for the sparse structure sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        unilat, pred_x_t = self.sample_sparse_structure(cond, num_samples, unilat_sampler_params)
        std = torch.tensor(self.unilat_normalization['std'])[None].to(unilat.device)
        mean = torch.tensor(self.unilat_normalization['mean'])[None].to(unilat.device)
        unilat = unilat * std.view(*std.shape,1,1,1).repeat(1,1,16,16,16) + mean.view(*mean.shape,1,1,1).repeat(1,1,16,16,16)
        if sample:
            return unilat
        return self.decode_unilat(unilat, formats)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        unilat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            unilat_sampler_params (dict): Additional parameters for the sparse structure sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.unilat_sampler_params, **unilat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('unilat_sampler', len(images), ss_steps, mode=mode):
            unilat, _ = self.sample_sparse_structure(cond, num_samples, unilat_sampler_params)
        std = torch.tensor(self.unilat_normalization['std'])[None].to(unilat.device)
        mean = torch.tensor(self.unilat_normalization['mean'])[None].to(unilat.device)
        unilat = unilat * std.view(*std.shape,1,1,1).repeat(1,1,16,16,16) + mean.view(*mean.shape,1,1,1).repeat(1,1,16,16,16)

        return self.decode_unilat(unilat, formats)