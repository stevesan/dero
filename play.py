
"""
Given a WAD, it'll detect if it's for DOOM1 or 2 (based on map names) and
run GZDoom with the right iwad arg
"""

import argparse
import os
import wad
import subprocess
import dero_config

parser = argparse.ArgumentParser()
parser.add_argument('pwad', type=str, help='Path to the PWAD to play')
args = parser.parse_args()

content = wad.load(args.pwad)
iwad = dero_config.DOOM2_WAD_PATH
firstname = content.maps[0].name
if firstname[0] == 'E' and firstname[2] == 'M':
    iwad = dero_config.DOOM1_WAD_PATH

print(f'First map name: {content.maps[0].name}. Assumed IWAD: {os.path.basename(iwad)}')

subprocess.check_call([
    '/Applications/GZDoom.app/Contents/MacOS/gzdoom',
    args.pwad,
    '-iwad', iwad,
    ])

