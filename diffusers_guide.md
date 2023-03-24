## Download the [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)

> download is not necessary, but considering the network issue, you'd better download it

Firstly check your `git-lfs` installation

```bash
git lfs install
```

If not, install `git-lfs`

Then pull the model

```bash
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1
```

## Use it with diffusers lib

Install necessary packages

```bash
# If your environment is CUDA 10.2
conda env create -v -f environment-cu10.2.yaml

# If your environment is CUDA 11.X
conda env create -v -f environment-cu11.yaml
```

specify `model_id` in `config.py` with the path where you clone [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)

run `txt_2_img.py`