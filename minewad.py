
import dero_config
import wad

""" 'datamines' wads... nothing really fancy right now """

def is_valid_texname(name):
    return len(name) > 1 and 'STINKY' not in name

class TextureSet:
    def __init__(s): pass

def read_texsets(path):
    sets = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            
            ts = TextureSet()
            parts = line.strip().split(' ')
            ts.floor = parts[0]
            ts.ceil = parts[1]

            line = f.readline()
            if line.strip() == '':
                continue

            ts.sidetexs = line.strip().split(' ')
            sets += [ts]

    return sets

if __name__ == '__main__':
    refwad = wad.load(dero_config.DOOM1_WAD_PATH)

    with open('texsets.txt', 'w') as f:
        for mapp in refwad.maps:
            for sec in mapp.sectors:
                if not is_valid_texname(sec.floor_pic) or not is_valid_texname(sec.ceil_pic):
                    continue

                f.write('%s %s\n' % (sec.floor_pic, sec.ceil_pic))

                for sd in mapp.sidedefs:
                    if mapp.sectors[sd.sector] != sec:
                        continue
                    if is_valid_texname(sd.lowertex):
                        f.write(sd.lowertex + ' ')
                    if is_valid_texname(sd.midtex):
                        f.write(sd.midtex + ' ')
                    if is_valid_texname(sd.uppertex):
                        f.write(sd.uppertex + ' ')
                f.write('\n')

    print 'dumping full lists'
            
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
