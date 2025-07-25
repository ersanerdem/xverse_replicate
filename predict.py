# predict.py
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from peft import PeftModel
from PIL import Image
from typing import List
import os
import random
from cog import BasePredictor, Input, Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info(f"Loading base model on device: {self.device}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=self.dtype,
            safety_checker=None,
            use_safetensors=True
        ).to(self.device)

        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

        self.base_components = {
            "vae": self.pipe.vae,
            "text_encoder": self.pipe.text_encoder,
            "tokenizer": self.pipe.tokenizer,
            "unet": self.pipe.unet,
            "scheduler": self.pipe.scheduler,
            "feature_extractor": self.pipe.feature_extractor,
            "safety_checker": None,
            "requires_safety_checker": False
        }

    def _create_pipeline(self, img2img=False):
        cls = StableDiffusionImg2ImgPipeline if img2img else StableDiffusionPipeline
        return cls(**self.base_components).to(self.device)

    def apply_lora(self, pipe, lora_path: str, weight: float = 0.8):
        if not lora_path or not os.path.exists(lora_path):
            logger.warning(f"LoRA path not found: {lora_path}")
            return False
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path).merge_and_unload(scale=weight)
        logger.info(f"LoRA applied: {lora_path}")
        return True

    def predict(
        self,
        prompt: str = Input(description="Text prompt for image generation."),
        seed: int = Input(default=-1, description="Seed value for reproducibility"),
        target_height: int = Input(default=512, description="Output image height"),
        target_width: int = Input(default=512, description="Output image width"),
        num_images: int = Input(default=1, ge=1, le=4, description="Number of images to generate"),
        init_image: Path = Input(default=None, description="Optional input image for img2img"),
        strength: float = Input(default=0.7, description="Strength for img2img (if init_image provided)"),
        lora_model: Path = Input(default=None, description="Optional LoRA model file (.safetensors)"),
        lora_weight: float = Input(default=0.8, description="LoRA influence scale (0.0 to 1.0)"),
    ) -> List[Path]:

        pipe = self._create_pipeline(img2img=init_image is not None)

        if lora_model and os.path.exists(str(lora_model)):
            self.apply_lora(pipe, str(lora_model), lora_weight)
        generator = torch.Generator(self.device)
        try:
            seed = int(seed)
        except Exception:
            seed = -1
        
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        generator.manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "guidance_scale": 7.5,
            "generator": generator,
            "num_inference_steps": 30,
            "num_images_per_prompt": num_images,
        }

        if init_image:
            image = Image.open(str(init_image)).convert("RGB").resize((target_width, target_height))
            kwargs["image"] = image
            kwargs["strength"] = strength
        else:
            kwargs["width"] = target_width
            kwargs["height"] = target_width

        with torch.inference_mode():
            output = pipe(**kwargs).images

        output_paths = []
        for i, img in enumerate(output):
            out_path = f"/tmp/output_{i}.jpg"
            img.save(out_path)
            output_paths.append(Path(out_path))

        return output_paths
