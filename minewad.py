
import dero_config
import wad

""" 'datamines' wads... nothing really fancy right now """

refwad = wad.load(dero_config.DOOM1_WAD_PATH)
floortexs = [sec.floor_pic for m in refwad.maps for sec in m.sectors if len(sec.floor_pic) > 2]
ceiltexs = [sec.ceil_pic for m in refwad.maps for sec in m.sectors if len(sec.ceil_pic) > 2]
midtexs = [sd.midtex for m in refwad.maps for sd in m.sidedefs if len(sd.midtex) > 2]
uppertexs = [sd.uppertex for m in refwad.maps for sd in m.sidedefs if len(sd.uppertex) > 2]
lowertexs = [sd.lowertex for m in refwad.maps for sd in m.sidedefs if len(sd.lowertex) > 2]

def write_texnames(path, names):
    with open(path, 'w') as f:
        for name in names: f.write(name + '\n')

write_texnames('floortexs.txt', floortexs)
write_texnames('ceiltexs.txt', ceiltexs)
write_texnames('midtexs.txt', midtexs)
write_texnames('uppertexs.txt', uppertexs)
write_texnames('lowertexs.txt', lowertexs)
