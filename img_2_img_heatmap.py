from scripts.img2img import load_img
import config
from diffusers import StableDiffusionImg2ImgPipeline
from daam import trace, set_seed
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import requests
from PIL import Image
from pathlib import Path
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


output_path = os.path.join(config.output_dir, "img_2_img_heatmap")
Path(output_path).mkdir(parents=True, exist_ok=True)

# load the pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    config.model_id,
    torch_dtype=torch.float32
).to(
    config.device
)


class ImageDataset(Dataset):
    def __init__(self, img_filename_list, transform=None) -> None:
        self.img_filename_list = img_filename_list
        self.transform = transform

    def __len__(self):
        return len(self.img_filename_list)

    def __getitem__(self, index):
        img = load_img(self.img_filename_list[index])
        img = self.transform(img)
        return img[0], {"name": self.img_filename_list[index]}


# image_file_list = [850, 832, 867, 889, 912]
image_file_list = list(range(460, 1117))
image_file_list = [f'../video_output/{i}.png' for i in image_file_list]

dataset = ImageDataset(
    img_filename_list=image_file_list,
    transform=transforms.CenterCrop(768),
)
loader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
)

prompt = "heavy smoke on a dirt road, rising smoke, yellow smoke"
keyword = 'smoke'

with torch.cuda.amp.autocast(dtype=torch.float32), torch.no_grad():
    for i, (init_imgs, names) in enumerate(loader):
        print(f"batch {i}")
        img_num = init_imgs.shape[0]
        with trace(pipe) as tc:
            images = pipe(
                prompt=prompt, init_image=init_imgs,
                num_images_per_prompt=img_num,
                num_inference_steps=40,
                strength=0.88, guidance_scale=7.5
            ).images

            heat_map = tc.compute_global_heat_map()
            heat_map = heat_map.compute_word_heat_map(word=keyword)
            for i, img in enumerate(images):
                save_file_name = f"{Path(names['name'][i]).stem}_{keyword}"
                print(f"saving {save_file_name}")
                img.save(
                    os.path.join(output_path, f"{save_file_name}.png"))
                np.save(
                    os.path.join(
                        output_path, f"{save_file_name}_heatmap.npy"),
                    heat_map.value.squeeze().cpu().numpy()
                )
                heat_map.plot_overlay(
                    img,
                    out_file=os.path.join(
                        output_path, f"{save_file_name}_heatmap.png"),
                )
