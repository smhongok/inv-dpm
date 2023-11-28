from statistics import mean, stdev
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.functional import to_pil_image, rgb_to_grayscale

from src.stable_diffusion.inverse_stable_diffusion import InversableStableDiffusionPipeline 
from diffusers import DPMSolverMultistepScheduler
from src.utils.optim_utils import *
from src.utils.io_utils import *
import torch
import textwrap

    
def stable_diffusion_pipe(
        solver_order=1,
        model_id='stabilityai/stable-diffusion-2-1-base',
):
    # load stable diffusion pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler(
        beta_end=0.012,
        beta_schedule='scaled_linear',
        beta_start=0.00085,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        steps_offset=1, 
        trained_betas=None,
        solver_order=solver_order,
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32,
    )
    pipe = pipe.to(device)

    return pipe

def generate(     
        image_num=0,
        prompt=None,
        guidance_scale=3.0,
        num_inference_steps=50,
        solver_order=1,
        image_length=512,
        datasets='Gustavosta/Stable-Diffusion-Prompts',
        model_id='stabilityai/stable-diffusion-2-1-base',
        gen_seed=0,
        pipe=None,
        init_latents=None,
):
    # load stable diffusion pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if pipe is None:
        scheduler = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule='scaled_linear',
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            steps_offset=1, 
            trained_betas=None,
            solver_order=solver_order,
        )
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float32,
        )
    pipe = pipe.to(device)

    # load dataset and prompt
    if prompt is None:
        dataset, prompt_key = get_dataset(datasets)
        prompt = dataset[image_num][prompt_key]

    # generate init latent
    seed = gen_seed + image_num
    set_random_seed(seed)
    
    if init_latents is None:
        init_latents = pipe.get_random_latents()

    # generate image
    output, _ = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=image_length,
        width=image_length,
        latents=init_latents,
    )
    image = output.images[0]

    return image, prompt, init_latents

def exact_inversion(
        image,
        prompt='',
        guidance_scale=3.0,
        num_inference_steps=50,
        solver_order=1,
        test_num_inference_steps=50,
        inv_order=1,
        decoder_inv = True,
        model_id='stabilityai/stable-diffusion-2-1-base',
        pipe=None,
):
    # load stable diffusion pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if pipe is None:
        scheduler = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule='scaled_linear',
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            steps_offset=1, 
            trained_betas=None,
            solver_order=solver_order,
        )
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float32,
        )
    pipe = pipe.to(device)
    
    # prompt to text embeddings
    text_embeddings_tuple = pipe.encode_prompt(
        prompt, 'cuda', 1, guidance_scale > 1.0, None
    )
    text_embeddings = torch.cat([text_embeddings_tuple[1], text_embeddings_tuple[0]])

    # image to latent
    image = transform_img(image).unsqueeze(0).to(text_embeddings.dtype).to(device)
    if decoder_inv:
        image_latents = pipe.decoder_inv(image)
    else:
        image_latents = pipe.get_image_latents(image, sample=False)

    # forward diffusion : image to noise
    reversed_latents = pipe.forward_diffusion(
        latents=image_latents,
        text_embeddings=text_embeddings,
        guidance_scale=guidance_scale,
        num_inference_steps=test_num_inference_steps,
        inverse_opt=(inv_order!=0),
        inv_order=inv_order
    )

    return reversed_latents

def error_nmse(orig,recon):
    orig = to_numpy(orig).astype(np.float32)
    recon = to_numpy(recon).astype(np.float32)
    error = (np.linalg.norm((orig - recon)) / np.linalg.norm(orig))**2
    return error

def error_map(orig, recon, scale=1):
    orig = to_numpy(orig).astype(np.float32)
    recon = to_numpy(recon).astype(np.float32)
    error = orig - recon    
    error_map = ((error * scale + 127).clip(0,255).astype(np.uint8))
    return error_map

def to_numpy(img):
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, Image.Image):
        return np.array(img)
    elif isinstance(img, torch.Tensor):
        return img.cpu().detach().numpy()
    else:
        raise ValueError("Unsupported image type. Expected NumPy array, PIL Image, or Torch Tensor.")
    
def plot_recon_result(orig_noise, orig_image, recon_noise, recon_image, error_scale, pipe):
    fig1, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(8,8))

    orig_noise_img = ((pipe.decode_image(orig_noise)/2+0.5).clamp(0,1)*255)[0].permute(1,2,0).to(torch.uint8).cpu().numpy()
    recon_noise_img = ((pipe.decode_image(recon_noise)/2+0.5).clamp(0,1)*255)[0].permute(1,2,0).to(torch.uint8).cpu().numpy()
    nmse = error_nmse(orig_noise, recon_noise)
    map = error_map(orig_noise_img, recon_noise_img, scale=error_scale)

    ax0.imshow(orig_noise_img)
    ax0.set_title("Original")
    ax1.imshow(orig_noise_img)
    ax1.set_title("Recon")
    ax2.imshow(map)
    ax2.set_title(f"Error (x{error_scale})")

    ax0.axis('off')
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    fig1.suptitle(f"NMSE: {nmse:.5f}", fontsize=12, ha='center', y=0.33)
    plt.show()

    fig2, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(8,8))

    nmse = error_nmse(orig_image, recon_image)
    map = error_map(orig_image, recon_image, scale=error_scale)

    ax0.imshow(orig_image)
    ax0.set_title("Original")
    ax1.imshow(recon_image)
    ax1.set_title("Recon")
    ax2.imshow(map)
    ax2.set_title(f"Error (x{error_scale})")

    ax0.axis('off')
    ax1.axis('off')
    ax2.axis('off')

    plt.tight_layout()
    fig2.suptitle(f"NMSE: {nmse:.5f}", fontsize=12, ha='center', y=0.33)
    plt.show()