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
args = parser.parse_args()

def main():
  if args.pwad is None:
    wadfiles = []
    for root, dirs, files in os.walk('.'):
      for file in files:
        if file.upper().endswith('.WAD'):
          wadfiles.append(os.path.join(root, file))
    for i in range(len(wadfiles)):
      print(f'{i}. {wadfiles[i]}')
    choice = int(input('choose wad file:'))
    args.pwad = wadfiles[choice]

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

  call_args = [
      '/Applications/GZDoom.app/Contents/MacOS/gzdoom',
      args.pwad,
      '-iwad', iwad,
      ]

  if args.nomons: call_args += ['-nomonsters']

  subprocess.check_call(call_args)

if __name__ == '__main__':
  main()
