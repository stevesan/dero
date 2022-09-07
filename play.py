#!/opt/homebrew/bin/python3

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
parser.add_argument('--pwad', type=str, default=None, help='Path to the PWAD to play')
parser.add_argument('--nomons', action='store_true', default=False)
parser.add_argument('--voxels', action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

def main():
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
  pwad_dir = os.path.dirname(args.pwad)

  # Find all WADs and PK3's in this folder and load them.
  wadpaths = []
  for file in os.listdir(pwad_dir):
    if file.lower().endswith('.wad') or file.lower().endswith('.pk3'):
      wadpaths.append(os.path.join(pwad_dir, file))

  if args.voxels:
    wadpaths.append('/Users/stevenan/dooming/wads/cheello_voxels.zip')

  gzdoom_path = '/Applications/GZDoom.app/Contents/MacOS/gzdoom'
  call_args = [gzdoom_path] + wadpaths + [
      '-iwad', iwad,
      '-savedir', pwad_dir,
      '-shotdir', pwad_dir,
      ]

  if args.nomons: call_args += ['-nomonsters']

  print('final args:', call_args)
  subprocess.check_call(call_args)

if __name__ == '__main__':
  main()
