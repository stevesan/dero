import wad
import dero_config

def dump_all_maps(wadp):
    content = wad.load(wadp)
    for m in content.maps:
        wad.save_map_png(m, m.name + '.png')


dump_all_maps(dero_config.DOOM1_WAD_PATH)
dump_all_maps(dero_config.DOOM2_WAD_PATH)
