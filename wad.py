import sys
import struct
import pylab
from Queue import * 

class LumpStruct:

    def read(s,wad):
        wad.read_basic_lump(s, s.get_fields())

    def write(s,wad):
        wad.write_basic_lump(s, s.get_fields())

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

class LumpInfo(LumpStruct):

    FIELDS = [
    ('filepos', 'int'),
    ('size', 'int'),
    ('name', 'string8'),
    ]

    def get_fields(s): return LumpInfo.FIELDS

class WadThing(LumpStruct):

    FIELDS = [
    ('x', 'short'),
    ('y', 'short'),
    ('angle', 'short'),
    ('type', 'short'),
    ('options', 'short'),
    ]

    def get_fields(s): return WadThing.FIELDS

class Sector(LumpStruct):

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

class LineDef(LumpStruct):

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

class SideDef(LumpStruct):

    FIELDS = [
    ('xofs', 'short'),
    ('yofs', 'short'),
    ('uppertex', 'string8'),
    ('lowertex', 'string8'),
    ('midtex', 'string8'),
    ('sector', 'short'),
    ]

    def get_fields(s): return SideDef.FIELDS

class DummyLump(LumpStruct):
    """ Used for directory markers, like levels """
    FIELDS = []
    def get_fields(s): return DummyLump.FIELDS

def get_color_for_thing(thing_type):
    if thing_type == 1:
        return 'g'
    else:
        return 'y'

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
            pylab.plot( [p0[0], p1[0]], [p0[1], p1[1]], 'k-' )

        # make it square
        xx = [v[0] for v in s.verts]
        yy = [v[1] for v in s.verts]
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
            s.things = io.read_array_lump(lumpend, WadThing)
            for thing in s.things:
                if thing.type == 1:
                    print 'player start at %d %d' % (thing.x,thing.y)

        elif name == 'VERTEXES':
            while io.f.tell() < lumpend:
                s.verts += [(io.read_short(), io.read_short())]

        elif name == 'LINEDEFS':
            s.linedefs = io.read_array_lump(lumpend, LineDef)
        
        elif name == 'SIDEDEFS':
            s.sidedefs = io.read_array_lump(lumpend, SideDef)

        elif name == 'SECTORS':
            s.sectors = io.read_array_lump(lumpend, Sector)
        else:
            return False

        return True

def read(s, wad, lumpend, clazz):

    def write(s, wad):
        for data in s.array:
            data.write(wad)

class WADWriter:
    """ Handles specifics of writing WADs """

    def __init__(s):
        s.lumps = {}
        s.header = 'PWAD'

    def add_lump(s, name, data):
        if name in s.lumps:
            raise Error('%s already in WAD lumps!' % name)
        s.lumps[name] = data

    def write(s, fout):
        total_lump_size = sum([ lump.get_size() for lump in s.lumps])
        dir_offset = 4 + 8 + 8 + total_lump_size
        directory = []

# for lump in s.lumps:
# directory[lump] = 

        print 'total lump size = %d' % total_lump_size

        io = WADIO(fout)
        fout.write('PWAD')
        io.write_long( len(s.lumps) )
        io.write_long( dir_offset )

        lumpstart = 4 + 8 + 8

        for lump in lumps:
            data = lumps[lump]
            if type(data) == list:
                io.write_array_lump(data)
            else:
                data.write(io)
            # create dir entry
            entry = LumpInfo()
            entry.clear()
            entry.name = lump
            entry.size = data.get_size()
            entry.filepos = lumpstart
            directory += [entry]
            lumpstart += entry.size

        assert lumpstart == dir_offset
        io.write_array_lump(directory)

class WADContent:
    """ Should contain all essential contents of a WAD """

    def __init__(s):
        s.maps = []
        s.other_lumps = []

class WADIO:
    """ Low-level operations for read/writing lumps """

    def __init__(s, f):
        s.f = f
        s.verbose = False
        s.state = 'inited'
        s.maps = []

    def read_basic_lump(s, lump, fields):
        for field in fields:
            if field[1] == 'short':
                setattr(lump, field[0], s.read_short())
            elif field[1] == 'int':
                setattr(lump, field[0], s.read_long())
            elif field[1] == 'string8':
                setattr(lump, field[0], s.read_string8())

    def write_basic_lump(s, lump, fields):
        for field in fields:
            if field[1] == 'short':
                s.write_short( getattr(lump, field[0]) )
            elif field[1] == 'int':
                s.write_long( getattr(lump, field[0]) )
            elif field[1] == 'string8':
                s.write_string8( getattr(lump, field[0]) )

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
        print 'after reading dir: ' + str(s.f.tell())

    def read_array_lump(s, lumpend, clazz):
        size = clazz().get_size()
        assert (lumpend - s.f.tell()) % size == 0
        rv = []
        while s.f.tell() < lumpend:
            block = clazz()
            block.read(wad)
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

        state = 'lumps'
        map_entry = None
        _map = None

        uniq_texs = set()
        
        for entry in s.directory:
            s.f.seek(entry.filepos)
            lumpend = s.f.tell() + entry.size
            name = entry.name

            if s.is_map_start_lump(name):
                if _map:
                    # finish off current map
                    pylab.figure()
                    print 'plotting ...'
                    _map.plot()
                    pylab.grid(True)
                    pylab.savefig(map_entry.name+'.png')
                    print 'done plotting'
                    content.maps += [_map]

                assert entry.size == 0
                print 'reading map ' + entry.name
                state = 'map'
                map_entry = entry
                _map = Map(entry.name)

            elif _map and _map.handle_lump(s, entry, lumpend):
                # no need to do anything - it handled it
                pass
                    
            elif name == 'ENDOOM':
                # sanity check
                assert entry.size == 4000

            else:
                # ignore this lump
                pass

# print 'ENDOOM:\n' + msg

        print 'finished reading lumps'
        print 'curr pos: %d, dir start pos: %d' % (s.f.tell(), s.dir_offset)
        print 'read %d maps' % len(content.maps)

if __name__ == "__main__":
    with open(sys.argv[1], 'rb') as f:
        wad = WADIO(f)
        wad.verbose = True
        wad.read()
