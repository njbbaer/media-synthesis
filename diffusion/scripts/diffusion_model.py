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
        self.set_step_list()

    def set_step_list(self):
        steps = int(self.params['steps'] * self.params['step_multiplier'])
        t = torch.linspace(self.params['max_timestep'], 0, steps + 1, device='cuda')[:-1]
        self.step_list = utils.get_spliced_ddpm_cosine_schedule(t)

    # def cfg_model_fn(self, x, t):
    #     weights = torch.tensor([1 - sum([self.params['weight']]), *[self.params['weight']]], device='cuda')
    #     target_embeds = [clip_model.encode_text(clip.tokenize(self.params['prompt'])).float().cuda()]
    #     n = x.shape[0]
    #     n_conds = len(target_embeds)
    #     x_in = x.repeat([n_conds, 1, 1, 1])
    #     t_in = t.repeat([n_conds])
    #     clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
    #     vs = model(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
    #     v = vs.mul(weights[:, None, None, None, None]).sum(0)
    #     return v

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

    def display_image(self, outs, save=False):
        for i, out in enumerate(outs):
            pil_image = utils.to_pil_image(out)
            display.display(pil_image)
            if save:
                filename = f'{self.params["key"]}_{self.params["seed"]}.png'
                pil_image.save(filename)

    def set_step_list(self):
        if self.params['reverse']:
            t = torch.linspace(0, 1, self.params['steps'] + 1, device='cuda')
        else:
            t = torch.linspace(1, 0, self.params['steps'] + 1, device='cuda')
        step_list = utils.get_spliced_ddpm_cosine_schedule(t)
        self.step_list = step_list[step_list <= self.params['max_timestep']]

    def run(self):
        print(f'Performing {len(self.step_list)} steps at {self.params["size"]}')

        self.set_step_list()

        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(self.params['seed'])

        if self.params['use_init']:
            x = self.init
        else:
            x = torch.randn((1, 3) + self.params['size'], device='cuda')

        if self.params['scale']:
            size = [s * 2 for s in self.params['size']]
            x = transforms.Resize(size)(x)
            self.display_image(x)

        if self.params['reverse']:
            zero_embed = torch.zeros([1, clip_model.visual.output_dim], device='cuda')
            output = sampling.reverse_sample(model, x, self.step_list, {'clip_embed': zero_embed}, callback=self.display_callback)
        else:
            self.target_embed = clip_model.encode_text(clip.tokenize(self.params['prompt'])).float().cuda()
            output = sampling.sample(self.cfg_model_fn, x, self.step_list, self.params['eta'], {}, callback=self.display_callback)

        self.display_image(output, self.params['save'])
        return output
