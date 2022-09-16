
"""
Creates a json file listing all found wads and information about them.
"""

import wad
import os
import argparse
import json
import zipfile

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

def get_saves():
  rv = []
  for root, dirs, files in os.walk(args.wadsdir):
    for file in files:
      if file.lower().endswith('.zds') and file.lower().startswith('save'):
        rv.append(os.path.join(root, file))
  rv.sort()
  return rv

def main():
  wads_index = {}
  for pwad in get_wads():
    wads_index[pwad] = {}
    maps = list(wad.enum_map_names(pwad))
    if len(maps) > 0:
      wads_index[pwad]['maps'] = maps

  saves_index = {}
  for zdsp in get_saves():
    with zipfile.ZipFile(zdsp, mode='r') as zf:
      print(f'reading {zdsp}')
      def load_json(zp):
        jsonstring = zf.read(zp).decode('utf-8')
        return json.loads(jsonstring)
      g = load_json('globals.json')
      played_levels = []
      if 'statistics' in g and 'levels' in g['statistics']:
        played_levels += [lev['levelname'].upper() for lev in g['statistics']['levels']]

      info = load_json('info.json')
      if 'Current Map' in info:
        played_levels += [info['Current Map'].upper()]

      saves_index[zdsp] = {}
      saves_index[zdsp]['played_levels'] = played_levels

  index = {
    'wads': wads_index,
    'saves': saves_index,
    }

  with open(args.outf, 'w') as f:
    json.dump(index, f, indent=2)

if __name__ == '__main__':
  main()
