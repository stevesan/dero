import sys
import struct
import pylab
from Queue import * 
import random

import dero_config

"""
    lumps:
        read, write, 

    wadio take lump array, or handlers
        """

class SimpleStruct:

    def read(s,io):
        for field in s.get_fields():
            if field[1] == 'short':
                setattr(s, field[0], io.read_short())
            elif field[1] == 'int':
                setattr(s, field[0], io.read_long())
            elif field[1] == 'string8':
                setattr(s, field[0], io.read_string8())

    def write(s,io):
        for field in s.get_fields():
            if field[1] == 'short':
                io.write_short( getattr(s, field[0]) )
            elif field[1] == 'int':
                io.write_long( getattr(s, field[0]) )
            elif field[1] == 'string8':
                io.write_string8( getattr(s, field[0]) )

    def get_size(s):
        size = 0
        for field in s.get_fields():
            if field[1] == 'short':
                size += 2
            elif field[1] == 'int':
                size += 4
            elif field[1] == 'string8':
                size += 8
        return size

    def clear(s):
        for field in s.get_fields():
            if field[1] == 'short':
                setattr(s, field[0], 0)
            elif field[1] == 'int':
                setattr(s, field[0], 0)
            elif field[1] == 'string8':
                setattr(s, field[0], '')

class LumpInfo(SimpleStruct):

    FIELDS = [
    ('filepos', 'int'),
    ('size', 'int'),
    ('name', 'string8'),
    ]

    def get_fields(s): return LumpInfo.FIELDS

class Thing(SimpleStruct):

    FIELDS = [
    ('x', 'short'),
    ('y', 'short'),
    ('angle', 'short'),
    ('type', 'short'),
    ('options', 'short'),
    ]

    def get_fields(s): return Thing.FIELDS

class Vertex(SimpleStruct):
    FIELDS = [
    ('x', 'short'),
    ('y', 'short'),
    ]
    def get_fields(s): return Vertex.FIELDS

class Sector(SimpleStruct):

    FIELDS = [
    ('floor_height'   , 'short')   , 
    ('ceil_height'    , 'short')   , 
    ('floor_pic'      , 'string8') , 
    ('ceil_pic'       , 'string8') , 
    ('light_level'    , 'short')   , 
    ('special_sector' , 'short')   , 
    ('tag'            , 'short')   , 
    ]

    def get_fields(s): return Sector.FIELDS

class LineDef(SimpleStruct):

    FLAGBIT = {
       "Impassible" : 0,
       "Block Monsters" : 1,
       "Two-sided" : 2,
       "Upper Unpegged" : 3,
       "Lower Unpegged" : 4,
       "Secret" : 5,
       "Block Sound" : 6,
       "Not on Map" : 7,
       "Already on Map" : 8 }

    FIELDS = [
    ('vert0', 'short'),
    ('vert1', 'short'),
    ('flags', 'short'),
    ('function', 'short'),
    ('tag', 'short'),
    ('sd_left', 'short'),
    ('sd_right', 'short'),
    ]

    def get_fields(s): return LineDef.FIELDS

class SideDef(SimpleStruct):

    FIELDS = [
    ('xofs', 'short'),
    ('yofs', 'short'),
    ('uppertex', 'string8'),
    ('lowertex', 'string8'),
    ('midtex', 'string8'),
    ('sector', 'short'),
    ]

    def get_fields(s): return SideDef.FIELDS

class DummyLump():
    """ Used for directory markers, like levels """

    def __init__(s, name):
        s.name = name

    def get_name(s):
        return s.name

    def get_size(s):
        return 0

    def write(s, io): pass

def get_color_for_thing(thing_type):
    if thing_type == 1:
        return 'g'
    else:
        return 'y'

class ArrayLump:

    def __init__(s, name, array):
        s.name = name
        s.array = array

    def write(s, io):
        io.write_array_lump(s.array)

    def get_size(s):
        if len(s.array) == 0:
            return 0
        else:
            return len(s.array) * s.array[0].get_size()

    def get_name(s):
        return s.name

class Map:

    def __init__(s, name):
        s.name = name
        s.things = []
        s.verts = []
        s.linedefs = []
        s.sidedefs = []
        s.sectors = []

    def plot(s):

        for t in s.things:
            pylab.plot([t.x], [t.y], '.'+get_color_for_thing(t.type))
        for ld in s.linedefs:
            p0 = s.verts[ld.vert0]
            p1 = s.verts[ld.vert1]
            pylab.plot( [p0.x, p1.x], [p0.y, p1.y], 'k-' )

        # make it square
        xx = [v.x for v in s.verts]
        yy = [v.y for v in s.verts]
        dx = max(xx) - min(xx)
        dy = max(yy) - min(yy)
        len = max(dx, dy) * 1.1
        cx = (max(xx)+min(xx))/2.0
        cy = (max(yy)+min(yy))/2.0
        aspect_ratio = 4.0/5.0
        left = cx - len/2.0/aspect_ratio
        right = cx + len/2.0/aspect_ratio
        top = cy + len/2.0
        bot = cy - len/2.0
        pylab.xlim([ left, right ])
        pylab.ylim([ bot, top ])

    def unique_textures(s):
        uniqs = set()
        for sd in s.sidedefs:
            uniqs.add(sd.uppertex)
            uniqs.add(sd.lowertex)
            uniqs.add(sd.midtex)
        return uniqs

    def handle_lump(s, io, lump, lumpend):
        name = lump.name
        if name == 'THINGS':
            s.things += io.read_array_lump(lumpend, Thing)

        elif name == 'VERTEXES':
            s.verts += io.read_array_lump(lumpend, Vertex)

        elif name == 'LINEDEFS':
            s.linedefs += io.read_array_lump(lumpend, LineDef)
        
        elif name == 'SIDEDEFS':
            s.sidedefs += io.read_array_lump(lumpend, SideDef)

        elif name == 'SECTORS':
            s.sectors += io.read_array_lump(lumpend, Sector)
        else:
            return False

        return True

    def append_lumps(s, lumps):
        lumps += [
        DummyLump(s.name),
        ArrayLump('THINGS', s.things),
        ArrayLump('VERTEXES', s.verts),
        ArrayLump('LINEDEFS', s.linedefs),
        ArrayLump('SIDEDEFS', s.sidedefs),
        ArrayLump('SECTORS', s.sectors),
        ]

def write_wad(fout, header, lumps):
    io = WADFile(fout)
    fout.write('PWAD')
    io.write_long( len(lumps) )

    # dir offset
    total_lump_size = sum([ lump.get_size() for lump in lumps])
    dir_offset = 4 + 4 + 4 + total_lump_size
    io.write_long( dir_offset )

    # write lumps while bookeeping
    lumpstart = 4 + 4 + 4
    directory = []

    print 'dir off set = %d' % dir_offset

    for lump in lumps:
        print 'start = %d, tell = %d' % (lumpstart, fout.tell())
        assert lumpstart == fout.tell()
        lump.write(io)

        # create dir entry
        entry = LumpInfo()
        entry.clear()
        entry.name = lump.get_name()
        entry.size = lump.get_size()
        entry.filepos = lumpstart

        directory += [entry]
        print '%d += %d' % (lumpstart, entry.size)
        lumpstart += entry.size

    assert lumpstart == dir_offset
    io.write_array_lump(directory)

class WADContent:
    """ Should contain all essential contents of a WAD """

    def __init__(s):
        s.maps = []
        s.other_lumps = []
        s.end_msg = None

class WADFile:
    """ Low-level operations for read/writing lumps, for reading and writing wads """

    def __init__(s, f):
        s.f = f
        s.verbose = False
        s.state = 'inited'
        s.maps = []

    def read(s):
        assert s.state == 'inited'
        s.state = 'header'

        header = s.f.read(4)
        assert header == 'IWAD' or header == 'PWAD'

        s.num_lumps = s.read_long()
        if s.verbose: print 'num lumps: ' + str(s.num_lumps)
        s.dir_offset = s.read_long()
        if s.verbose: print 'directory offset: ' + str(s.dir_offset)
        s.lumps_offset = s.f.tell()
        if s.verbose: print 'lumps offset: ' + str(s.lumps_offset)
        s.f.seek(s.dir_offset)
        s.read_directory()
        s.f.seek(s.lumps_offset)
        rv = WADContent()
        s.read_lumps(rv)
        return rv

    def read_long(s):
        return struct.unpack('i', s.f.read(4))[0]

    def write_long(s, val):
        return s.f.write( struct.pack('i', val) )

    def read_short(s):
        return struct.unpack('h', s.f.read(2))[0]

    def write_short(s, val):
        return s.f.write( struct.pack('h', val) )

    def read_string8(s):
        raw = s.f.read(8)
        term = raw.find('\0')
        if term == -1:
            return raw
        else:
            return raw[0:term]

    def write_string8(s, val):
        final = val
        while len(final) < 8: final += '\0'
        s.f.write(final)

    def read_directory(s):
        assert s.state == 'header'
        s.state = 'directory'
        infosize = LumpInfo().get_size()
        end = s.f.tell() + s.num_lumps * infosize
        s.directory = s.read_array_lump(end, LumpInfo)

    def read_array_lump(s, lumpend, clazz):
        size = clazz().get_size()
        assert (lumpend - s.f.tell()) % size == 0
        rv = []
        while s.f.tell() < lumpend:
            block = clazz()
            block.read(s)
            rv += [block]
        return rv

    def write_array_lump(s, array):
        for block in array:
            block.write(s)

    def is_map_start_lump(s, name):
        return name.startswith('MAP') or (name[0] == 'E' and name[2] == 'M')

    def read_lumps(s, content):

        total_lump_size = sum([entry.size for entry in s.directory])
        print 'total lump size: %d' % total_lump_size

        _map = None
        
        for entry in s.directory:
            s.f.seek(entry.filepos)
            lumpend = s.f.tell() + entry.size
            name = entry.name

            if s.is_map_start_lump(name):
                assert entry.size == 0
                print 'reading map ' + entry.name
                _map = Map(entry.name)
                content.maps += [_map]

            elif _map and _map.handle_lump(s, entry, lumpend):
                # no need to do anything - it handled it
                pass
                    
            elif name == 'ENDOOM':
                # sanity check
                assert entry.size == 4000
                content.end_msg = s.f.read(4000)

            else:
                # ignore this lump
                pass

# print 'ENDOOM:\n' + msg

        print 'finished reading lumps'
        print 'curr pos: %d, dir start pos: %d' % (s.f.tell(), s.dir_offset)
        print 'read %d maps' % len(content.maps)

def save_map_png(_map, fname):
    pylab.figure()
    print 'plotting ...'
    _map.plot()
    pylab.grid(True)
    pylab.savefig(fname)
    print 'done plotting'

def test_doom1_wad():
    path = dero_config.DOOM1_WAD_PATH
    _map = None
    with open(path, 'rb') as f:
        wad = WADFile(f)
        wad.verbose = False
        content = wad.read()

        assert len(content.maps) == 36
        assert content.end_msg
# _map = random.choice(content.maps)
        _map = content.maps[0]

    # filter out all things except player start
    _map.things = [t for t in _map.things if t.type == 1]

    # write the map back
    lumps = []
    _map.append_lumps(lumps)
    with open('%s.wad' % _map.name, 'wb') as f:
        write_wad(f, 'PWAD', lumps)

    # run bsp on it
    dero_config.build_wad( '%s.wad' % _map.name, 'test.wad' )

    # read it back
    _map2 = None
    with open('%s.wad' % _map.name, 'rb') as f:
        wad = WADFile(f)
        wad.verbose = True
        content = wad.read()
        assert len(content.maps) == 1
        assert content.end_msg == None

        _map2 = content.maps[0]
        assert _map2.name == _map.name
        assert len(_map2.verts) == len(_map.verts)

    # draw maps for comparison
    save_map_png( _map, 'expected.png')
    save_map_png( _map2, 'actual.png')

if __name__ == "__main__":
    test_doom1_wad()
