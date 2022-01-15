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


# Load the model
model = get_model('cc12m_1_cfg')()
model.load_state_dict(torch.load('v-diffusion-pytorch/checkpoints/cc12m_1_cfg.pth', map_location='cpu'))
model = model.half().cuda().eval().requires_grad_(False)
clip_model = clip.load(model.clip_model, jit=False, device='cpu')[0]


class DiffusionModel:
    def __init__(self, params, init=None):
        self.params = params
        self.init = init

    def cfg_model_fn(self, x, t):
        """The CFG wrapper function."""
        n = x.shape[0]
        x_in = x.repeat([2, 1, 1, 1])
        t_in = t.repeat([2])
        clip_embed_repeat = self.target_embed.repeat([n, 1])
        clip_embed_in = torch.cat([torch.zeros_like(clip_embed_repeat), clip_embed_repeat])
        v_uncond, v_cond = model(x_in, t_in, clip_embed_in).chunk(2, dim=0)
        v = v_uncond + (v_cond - v_uncond) * self.params['weight']
        return v

    def display_callback(self, info):
        if self.params['display_every'] is None: return
        if info['i'] % self.params['display_every'] != 0: return

        nrow = math.ceil(info['pred'].shape[0]**0.5)
        grid = tv_utils.make_grid(info['pred'], nrow, padding=0)
        tqdm.write(f'Step {info["i"]} of {len(self.step_list)}:')
        display.display(utils.to_pil_image(grid))

    def display(self, image, save_file=False):
        for i, out in enumerate(image):
            pil_image = utils.to_pil_image(out)
            display.display(pil_image)
            if save_file:
                file = f'{save_file}.png'
                pil_image.save(file)

    def resize_and_center_crop(self, image):
        fac = max(self.params['size'][0] / image.size[0], self.params['size'][1] / image.size[1])
        image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
        return transforms.functional.center_crop(image, image.size[::-1])

    def run(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(self.params['seed'])

        range = (0, 1) if self.params['reverse'] else (1, 0)
        t = torch.linspace(*range, self.params['steps'] + 1, device='cuda')
        step_list = utils.get_spliced_ddpm_cosine_schedule(t)
        self.step_list = step_list[step_list <= self.params['max_timestep']]

        if self.init is not None:
            x = self.init
        elif self.params['load_file'] is not None:
            x = Image.open(self.params['load_file'] + '.png').convert('RGB')
            x = self.resize_and_center_crop(x)
            x = utils.from_pil_image(x).to('cuda')[None]
        else:
            x = torch.randn((1, 3) + self.params['size'], device='cuda')

        if self.params['scale']:
            size = [s * 2 for s in self.params['size']]
            x = transforms.Resize(size)(x)

        if self.params['reverse']:
            zero_embed = torch.zeros([1, clip_model.visual.output_dim], device='cuda')
            output = sampling.reverse_sample(model, x, self.step_list, {'clip_embed': zero_embed}, callback=self.display_callback)
        else:
            self.target_embed = clip_model.encode_text(clip.tokenize(self.params['prompt'])).float().cuda()
            output = sampling.sample(self.cfg_model_fn, x, self.step_list, self.params['eta'], {}, callback=self.display_callback)

        self.display(output, self.params["save_file"])
        return output
