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
    def __init__(self, params, init_image=None):
        self.params = params
        self.target_embed = clip_model.encode_text(clip.tokenize(self.params['prompt'])).float().cuda()
        self.set_init_image(init_image)
        self.set_step_list()

    def set_step_list(self):
        steps = int(self.params['steps'] * self.params['step_multiplier'])
        t = torch.linspace(self.params['starting_timestep'], 0, steps + 1, device='cuda')[:-1]
        self.step_list = utils.get_spliced_ddpm_cosine_schedule(t)
        # self.step_list = step_list[step_list <= self.params['starting_timestep']]

    def set_init_image(self, init_image):
        if init_image is None:
            self.init_image = None
        else:
            self.init_image = transforms.Resize((self.params['side_y'], self.params['side_x']))(init_image)

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
        tqdm.write(f'')

    def display_image(self, outs, save=False):
        for i, out in enumerate(outs):
            pil_image = utils.to_pil_image(out)
            display.display(pil_image)
            if save:
                filename = f'{self.params["key"]}_{self.params["seed"]}.png'
                pil_image.save(filename)

    def run(self):
        print(f'Performing {len(self.step_list)} steps at {self.params["side_y"]}x{self.params["side_x"]}')

        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(self.params['seed'])

        x = torch.randn([1, 3, self.params['side_y'], self.params['side_x']], device='cuda')
        if self.init_image is not None:
            alpha, sigma = utils.t_to_alpha_sigma(self.step_list[0])
            x = self.init_image * alpha + x * sigma
        self.output = sampling.sample(self.cfg_model_fn, x, self.step_list, self.params['eta'], {}, callback=self.display_callback)

        self.display_image(self.output, self.params['save'])
        return self.output
