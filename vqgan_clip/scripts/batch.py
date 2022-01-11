import os
import sys
import subprocess
import io

COMMAND = 'conda run -n vqgan python generate.py'
SIZE = 720
PROMPTS = [
    ['1910s_futurism', '1910s futurism'],
    ['1920s_futurism', '1920s futurism'],
    ['1930s_futurism', '1930s futurism'],
    ['1940s_futurism', '1940s futurism'],
    ['1950s_futurism', '1950s futurism'],
    ['1960s_futurism', '1960s futurism'],
    ['1970s_futurism', '1970s futurism'],
    ['1980s_futurism', '1980s futurism'],
    ['1990s_futurism', '1990s futurism'],
    ['2000s_futurism', '2000s futurism'],
]

for prompt in PROMPTS:
    print(f'Generating {prompt[0]}...')
    cmd = f'{COMMAND} -p "{prompt[1]}" -s {SIZE} {SIZE} -o output/{prompt[0]}.png'
    os.system(cmd)
