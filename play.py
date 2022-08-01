
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

iwad = dero_config.DOOM2_WAD_PATH

for name in wad.enum_map_names(args.pwad):
  print(f'First map name: {name}')
  if name[0] == 'E' and name[2] == 'M':
      print('..looks like DOOM 1')
      iwad = dero_config.DOOM1_WAD_PATH
  else:
      print('..looks like DOOM 2')
  break

print(f'Assumed IWAD: {os.path.basename(iwad)}')

subprocess.check_call([
    '/Applications/GZDoom.app/Contents/MacOS/gzdoom',
    args.pwad,
    '-iwad', iwad,
    ])

