
"""
Creates a json file listing all found wads and information about them.
"""

import wad
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--wadsdir', type=str, help='Path that will be recursively scanned for WADS.')
parser.add_argument('--outf', type=str, help='Path to write JSON index to.')
args = parser.parse_args()

def get_wads():
  wadfiles = []
  for root, dirs, files in os.walk(args.wadsdir):
    for file in files:
      if file.lower().endswith('.wad'):
        wadfiles.append(os.path.join(root, file))
  wadfiles.sort()
  return wadfiles

def main():
  index = {}
  for pwad in get_wads():
    index[pwad] = {}
    maps = list(wad.enum_map_names(pwad))
    if len(maps) > 0:
      index[pwad]['maps'] = maps
  with open(args.outf, 'w') as f:
    json.dump(index, f, indent=2)

if __name__ == '__main__':
  main()
