import gc
import math
import sys

import torch
from torchvision import utils as tv_utils
from torchvision import transforms

from IPython import display
from tqdm.notebook import trange, tqdm

sys.path.append('./v-diffusion-pytorch')

from CLIP import clip
from diffusion import get_model, sampling, utils

from PIL import Image


# Load the models
model = get_model('cc12m_1_cfg')()
_, side_y, side_x = model.shape
model.load_state_dict(torch.load('v-diffusion-pytorch/checkpoints/cc12m_1_cfg.pth', map_location='cpu'))
model = model.half().cuda().eval().requires_grad_(False)
clip_model = clip.load(model.clip_model, jit=False, device='cpu')[0]


def cfg_model_fn(x, t):
    """The CFG wrapper function."""
    n = x.shape[0]
    x_in = x.repeat([2, 1, 1, 1])
    t_in = t.repeat([2])
    clip_embed_repeat = target_embed.repeat([n, 1])
    clip_embed_in = torch.cat([torch.zeros_like(clip_embed_repeat), clip_embed_repeat])
    v_uncond, v_cond = model(x_in, t_in, clip_embed_in).chunk(2, dim=0)
    v = v_uncond + (v_cond - v_uncond) * weight
    return v


def display_callback(info):
    if display_every is None: return
    if info['i'] % display_every != 0: return

    nrow = math.ceil(info['pred'].shape[0]**0.5)
    grid = tv_utils.make_grid(info['pred'], nrow, padding=0)
    tqdm.write(f'Step {info["i"]} of {steps}:')
    display.display(utils.to_pil_image(grid))
    tqdm.write(f'')


def run():
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(current_seed)

    x = torch.randn([1, 3, side_y, side_x], device='cuda')
    t = torch.linspace(1, 0, steps + 1, device='cuda')[:-1]
    step_list = utils.get_spliced_ddpm_cosine_schedule(t)
    # step_list = step_list[step_list > 0.5]
    outs = sampling.sample(cfg_model_fn, x, step_list, eta, {}, callback=display_callback)
    display_image(outs)

    init = transforms.Resize((side_y*2, side_x*2))(outs)
    x = torch.randn([1, 3, side_y*2, side_x*2], device='cuda')
    t = torch.linspace(1, 0, steps + 1, device='cuda')[:-1]
    step_list = utils.get_spliced_ddpm_cosine_schedule(t)
    step_list = step_list[step_list < 0.7]
    alpha, sigma = utils.t_to_alpha_sigma(step_list[0])
    x = init * alpha + x * sigma
    outs = sampling.sample(cfg_model_fn, x, step_list, eta, {}, callback=display_callback)
    display_image(outs)

    tqdm.write('Done!')


def display_image(outs, save=False):
    for i, out in enumerate(outs):
        pil_image = utils.to_pil_image(out)
        display.display(pil_image)
        if save: pil_image.save(filename)


if __name_ == '__main__':
    for batch in batches:
        prompt = batch.get('prompt') or defaults['prompt']
        weight = batch.get('weight') or defaults['weight']
        n_images = batch.get('n_images') or defaults['n_images']
        eta = batch.get('eta') or defaults['eta']
        seed = batch.get('seed') or defaults['seed']
        steps = batch.get('steps') or defaults['steps']
        side_x = batch.get('side_x') or defaults['side_x']
        side_y = batch.get('side_y') or defaults['side_y']
        display_every = batch.get('display_every') or defaults['display_every']

        target_embed = clip_model.encode_text(clip.tokenize(prompt)).float().cuda()
        for i in range(n_images):
            n = '' if i == 0 else f'_{i}'
            filename = f'{batch["name"]}{n}.png'
            current_seed = seed + i
            print(f'Generating {filename}')
            run()
