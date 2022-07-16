import argparse
import os
import wad

parser = argparse.ArgumentParser()
parser.add_argument('wad', type=str, help='Path to the WAD to dump')
parser.add_argument('outdir', type=str, help='Directory to dump map PNGs to')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
content = wad.load(args.wad)
for mapp in content.maps:
    wad.save_map_png(mapp, os.path.join(args.outdir, mapp.name + '.png'))
