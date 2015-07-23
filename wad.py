import sys
import struct
import pylab

class WAD:

    def __init__(s, f):
        s.f = f
        s.verbose = False
        s.state = 'inited'

    def read(s):
        assert s.state == 'inited'
        s.state = 'header'

        tag = s.f.read(4)
        assert tag == 'IWAD' or tag == 'PWAD'

        s.num_lumps = s.read_long()
        if s.verbose: print 'num lumps: ' + str(s.num_lumps)
        s.dir_offset = s.read_long()
        if s.verbose: print 'directory offset: ' + str(s.dir_offset)
        s.lumps_offset = s.f.tell()
        if s.verbose: print 'lumps offset: ' + str(s.lumps_offset)
        s.f.seek(s.dir_offset)
        s.read_directory()
        s.f.seek(s.lumps_offset)
        s.read_lumps()

    def read_long(s):
        return struct.unpack('i', s.f.read(4))[0]

    def read_short(s):
        return struct.unpack('h', s.f.read(2))[0]

    def read_string8(s):
        raw = s.f.read(8)
        term = raw.find('\0')
        if term == -1:
            return raw
        else:
            return raw[0:term]
# TODO: strip off trailing NULLs

    def read_directory(s):
        assert s.state == 'header'
        s.state = 'directory'
        s.directory = []
        for i in range(s.num_lumps):
            entry = {
                'seekpos' : s.read_long(),
                'size' : s.read_long(),
                'name' : s.read_string8() }
            s.directory += [entry]
        print 'after reading dir: ' + str(s.f.tell())

    def read_lumps(s):

        total_lump_size = sum([entry['size'] for entry in s.directory])
        print 'total lump size: %d' % total_lump_size

        state = 'lumps'
        map_entry = None
        num_maps = 0
        for entry in s.directory:
            s.f.seek(entry['seekpos'])
            lumpend = s.f.tell() + entry['size']
            name = entry['name']
# print lumpend, name
            if name.startswith('MAP') or (name[0] == 'E' and name[2] == 'M'):
                assert entry['size'] == 0
                print 'reading map ' + entry['name']
                state = 'map'
                map_entry = entry
                num_maps += 1
            elif name == 'THINGS':
                assert state == 'map'
                assert map_entry
                size = entry['size']
                assert (size % (5*2)) == 0
                print 'THINGS size = %d, num things = %d' % (size, size/10)
                xx = []
                yy = []
                pstart_found = False
                while s.f.tell() < lumpend:
                    x = s.read_short()
                    y = s.read_short()
                    angle = s.read_short()
                    ttype = s.read_short()
                    options = s.read_short()
                    if ttype == 1:
                        assert not pstart_found
                        pstart_found = True
                        print 'player start at %d %d' % (x,y)
            elif name == 'ENDOOM':
                assert entry['size'] == 4000
                msg = ''
                for i in range(2000):
                    msg += s.f.read(2)[1]

# print 'ENDOOM:\n' + msg
            # always make sure we end up at end of lump
            s.f.seek( lumpend )

        print 'finished reading lumps'
        print 'curr pos: %d, dir start pos: %d' % (s.f.tell(), s.dir_offset)
        print 'read %d maps' % num_maps

def summarize(wadpath):
    with open(wadpath, 'rb') as f:
        wad = WAD(f)
        wad.verbose = True
        wad.read()

if __name__ == "__main__":
    summarize(sys.argv[1])
