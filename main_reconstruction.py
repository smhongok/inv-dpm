import argparse
import wandb
from tqdm import tqdm
from statistics import mean, stdev

import torch
from torchvision.transforms.functional import to_pil_image, rgb_to_grayscale
import torchvision.transforms as transforms

from src.stable_diffusion.inverse_stable_diffusion import InversableStableDiffusionPipeline 
from diffusers import DPMSolverMultistepScheduler
from src.utils.optim_utils import *
from src.utils.io_utils import *
import torch
import statistics

def evaluate(t1,t2,t3,t4):
    recon_err_T0T = []
    recon_err_0T0 = [] 
    for i in range(len(t1)):
        recon_err_T0T.append( (((t1[i]-t3[i]).norm()/(t1[i].norm())).item())**2 )
        recon_err_0T0.append( (((t2[i]-t4[i]).norm()/(t2[i].norm())).item())**2 )

    if len(t1) == 1:
        print("T0T NMSE:", recon_err_T0T[0])
        print("0T0 NMSE:", recon_err_0T0[0])
        return recon_err_T0T[0], 0, recon_err_0T0[0], 0
    
    data_T0T = recon_err_T0T
    mean_T0T = statistics.mean(data_T0T)
    std_T0T = statistics.stdev(data_T0T)

    print("T0T NMSE")
    print("mean:", mean_T0T)
    print("std:", std_T0T)

    data_0T0 = recon_err_0T0
    mean_0T0 = statistics.mean(data_0T0)
    std_0T0 = statistics.stdev(data_0T0)

    print("0T0 NMSE")
    print("mean:", mean_0T0)
    print("std:", std_0T0)

    return mean_T0T, std_T0T, mean_0T0, std_0T0


def main(args):
    # track with wandb
    table = None
    if args.with_tracking:
        wandb.init(project='reconstruction', name=args.run_name)
        wandb.config.update(args)
        table = wandb.Table(columns=['image','recon_image','n2n_error','i2i_error', 'prompt'])
    
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
        solver_order=args.solver_order,
    )

    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        )
    pipe = pipe.to(device)

    # load dataset
    dataset, prompt_key = get_dataset(args.dataset)

    x_T_first = []
    x_0_second = []
    x_T_third = []
    x_0_fourth = []

    ind = 0
    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        if not args.prompt_unknown:
            text_embeddings_tuple = pipe.encode_prompt(
                current_prompt, 'cuda', 1, args.guidance_scale > 1.0, None)
            text_embeddings = torch.cat([text_embeddings_tuple[1], text_embeddings_tuple[0]])
        else:
            tester_prompt = ''
            text_embeddings = pipe.get_text_embedding(tester_prompt)
        
        ### Generation

        # generate init latent
        set_random_seed(seed)
        init_latents = pipe.get_random_latents()

        x_T_first.append(init_latents.clone())

        # generate image
        outputs, _ = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents,
            )
        orig_image = outputs.images[0]

        x_0_second.append(transforms.ToTensor()(orig_image).to(torch.float32))
        
        ### Inversion

        # image to latent
        img = transform_img(orig_image).unsqueeze(0).to(text_embeddings.dtype).to(device)
        if args.wo_decoder_inv:
            image_latents = pipe.get_image_latents(img, sample=False)
        else:    
            image_latents = pipe.decoder_inv(img)
            
        # forward_diffusion : image to noise
        reversed_latents = pipe.forward_diffusion(
            latents=image_latents, 
            text_embeddings=text_embeddings,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.test_num_inference_steps,
            inverse_opt=(args.inv_order!=0),
            inv_order=args.inv_order
        )

        x_T_third.append(reversed_latents.clone())
            

        ### Reconstrution
        reconstructed_outputs, _ = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=reversed_latents,
        )
        reconstructed_image = reconstructed_outputs.images[0]

        x_0_fourth.append(transforms.ToTensor()(reconstructed_image).to(torch.float32))

        ### Error
        err_T0T = (((x_T_first[-1]-x_T_third[-1]).norm()/x_T_first[-1].norm()).item())**2 # noise to noise error
        err_0T0 = (((x_0_second[-1]-x_0_fourth[-1]).norm()/x_0_second[-1].norm()).item())**2 # image to image error
        print(f"T0T NMSE: {err_T0T}")
        print(f"0T0 NMSE: {err_0T0}")

        if args.with_tracking:
            table.add_data(wandb.Image(orig_image),wandb.Image(reconstructed_image),err_T0T,err_0T0,current_prompt)
        
        ind = ind + 1
        
    
        mean_T0T, std_T0T, mean_0T0, std_0T0 = evaluate(x_T_first,x_0_second,x_T_third,x_0_fourth)

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'mean_T0T' : mean_T0T,'std_T0T' : std_T0T,'mean_0T0' : mean_0T0,'std_0T0' : std_0T0})
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=3.0, type=float, help="classifier-free guidance")
    parser.add_argument('--with_tracking', action='store_true', help="track with wandb")
    
    # experiment
    parser.add_argument('--num_inference_steps', default=50, type=int, help="steps of sampling")
    parser.add_argument("--solver_order", default=1, type=int, help="order of sampling, 1:DDIM, >=2:DPM-solver++") 
    parser.add_argument('--test_num_inference_steps', default=None, type=int, help="steps of inversion")
    parser.add_argument("--inv_order", type=int, default=None, help="order of inversion, 0:Naive DDIM iversion, 1:Alg1, 2:Alg2")
    
    parser.add_argument("--wo_decoder_inv", action="store_true", default=False, help="use encoder instead of exact decoder inversion")
    parser.add_argument("--prompt_unknown", action='store_true', default=False, help="assume during inversion, the original prompt is unknown")

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    if args.inv_order is None:
            args.inv_order = args.solver_order

    main(args)