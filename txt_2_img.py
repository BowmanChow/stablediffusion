from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import config
import os
from pathlib import Path

output_path = os.path.join(config.output_dir, "txt_2_img")
Path(output_path).mkdir(parents=True, exist_ok=True) 

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(
    config.model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(config.device)

prompt = "a photo of an astronaut riding a horse on mars"
images = pipe(prompt, num_images_per_prompt=2).images

for i, img in enumerate(images):
    img.save(
        os.path.join(output_path, f"astronaut_riding_horse_{i}.png")
    )
