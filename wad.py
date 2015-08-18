import sys
import struct
import pylab
from Queue import * 
import random
import utils
import dero_config

"""
This is totally derived from the Unofficial DOOM spec taken from:
http://www.gamers.org/dhs/helpdocs/dmsp1666.html
on July 20th, 2015
"""
THING_TABLE = {
  -1 : "ffff   ---- -        (nothing)",
   0 : "0000   ---- -        (nothing)",
   1 : "0001   PLAY +        Player 1 start (Player 1 start needed on ALL levels)",
   2 : "0002   PLAY +        Player 2 start (Player starts 2-4 are needed in)",
   3 : "0003   PLAY +        Player 3 start (cooperative mode multiplayer games)",
   4 : "0004   PLAY +        Player 4 start",
  11 : "000b   ---- -        Deathmatch start positions. Should have >= 4/level",
  14 : "000e   ---- -        Teleport landing. Where players/monsters land when they teleport to the SECTOR containing this thing",
3004 : "0bbc   POSS +      # FORMER HUMAN: regular pistol-shooting zombieman",
  84 : "0054 2 SSWV +      # WOLFENSTEIN SS: guest appearance by Wolf3D blue guy",
   9 : "0009   SPOS +      # FORMER HUMAN SERGEANT: black armor, shotgunners",
  65 : "0041 2 CPOS +      # HEAVY WEAPON DUDE: red armor, chaingunners",
3001 : "0bb9   TROO +      # IMP: brown, hurl fireballs",
3002 : "0bba   SARG +      # DEMON: pink, muscular bull-like chewers",
  58 : "003a   SARG +      # SPECTRE: invisible version of the DEMON",
3006 : "0bbe r SKUL +     ^# LOST SOUL: flying flaming skulls, they really bite",
3005 : "0bbd r HEAD +     ^# CACODEMON: red one-eyed floating heads. Behold...",
  69 : "0045 2 BOS2 +      # HELL KNIGHT: grey-not-pink BARON, weaker",
3003 : "0bbb   BOSS +      # BARON OF HELL: cloven hooved minotaur boss",
  68 : "0044 2 BSPI +      # ARACHNOTRON: baby SPIDER, shoots green plasma",
  71 : "0047 2 PAIN +     ^# PAIN ELEMENTAL: shoots LOST SOULS, deserves its name",
  66 : "0042 2 SKEL +      # REVENANT: Fast skeletal dude shoots homing missles",
  67 : "0043 2 FATT +      # MANCUBUS: Big, slow brown guy shoots barrage of fire",
  64 : "0040 2 VILE +      # ARCH-VILE: Super-fire attack, ressurects the dead!",
   7 : "0007 r SPID +      # SPIDER MASTERMIND: giant walking brain boss",
  16 : "0010 r CYBR +      # CYBER-DEMON: robo-boss, rocket launcher",
  88 : "0058 2 BBRN +      # BOSS BRAIN: Horrifying visage of the ultimate demon",
  89 : "0059 2 -    -        Boss Shooter: Shoots spinning skull-blocks",
  87 : "0057 2 -    -        Spawn Spot: Where Todd McFarlane's guys appear",
2005 : "07d5   CSAW a      $ Chainsaw",
2001 : "07d1   SHOT a      $ Shotgun",
  82 : "0052 2 SGN2 a      $ Double-barreled shotgun",
2002 : "07d2   MGUN a      $ Chaingun, gatling gun, mini-gun, whatever",
2003 : "07d3   LAUN a      $ Rocket launcher",
2004 : "07d4 r PLAS a      $ Plasma gun",
2006 : "07d6 r BFUG a      $ Bfg9000",
2007 : "07d7   CLIP a      $ Ammo clip",
2008 : "07d8   SHEL a      $ Shotgun shells",
2010 : "07da   ROCK a      $ A rocket",
2047 : "07ff r CELL a      $ Cell charge",
2048 : "0800   AMMO a      $ Box of Ammo",
2049 : "0801   SBOX a      $ Box of Shells",
2046 : "07fe   BROK a      $ Box of Rockets",
  17 : "0011 r CELP a      $ Cell charge pack",
   8 : "0008   BPAK a      $ Backpack: doubles maximum ammo capacities",
2011 : "07db   STIM a      $ Stimpak",
2012 : "07dc   MEDI a      $ Medikit",
2014 : "07de   BON1 abcdcb ! Health Potion +1% health",
2015 : "07df   BON2 abcdcb ! Spirit Armor +1% armor",
2018 : "07e2   ARM1 ab     $ Green armor 100%",
2019 : "07e3   ARM2 ab     $ Blue armor 200%",
  83 : "0053 2 MEGA abcd   ! Megasphere: 200% health, 200% armor",
2013 : "07dd   SOUL abcdcb ! Soulsphere, Supercharge, +100% health",
2022 : "07e6 r PINV abcd   ! Invulnerability",
2023 : "07e7 r PSTR a      ! Berserk Strength and 100% health",
2024 : "07e8   PINS abcd   ! Invisibility",
2025 : "07e9   SUIT a     (!)Radiation suit - see notes on ! above",
2026 : "07ea   PMAP abcdcb ! Computer map",
2045 : "07fd   PVIS ab     ! Lite Amplification goggles",
   5 : "0005   BKEY ab     $ Blue keycard",
  40 : "0028 r BSKU ab     $ Blue skullkey",
  13 : "000d   RKEY ab     $ Red keycard",
  38 : "0026 r RSKU ab     $ Red skullkey",
   6 : "0006   YKEY ab     $ Yellow keycard",
  39 : "0027 r YSKU ab     $ Yellow skullkey",
2035 : "07f3   BAR1 ab+    # Barrel; not an obstacle after blown up (BEXP sprite)",
  72 : "0048 2 KEEN a+     # A guest appearance by Billy",
  48 : "0030   ELEC a      # Tall, techno pillar",
  30 : "001e r COL1 a      # Tall green pillar",
  32 : "0020 r COL3 a      # Tall red pillar",
  31 : "001f r COL2 a      # Short green pillar",
  36 : "0024 r COL5 ab     # Short green pillar with beating heart",
  33 : "0021 r COL4 a      # Short red pillar",
  37 : "0025 r COL6 a      # Short red pillar with skull",
  47 : "002f r SMIT a      # Stalagmite: small brown pointy stump",
  43 : "002b r TRE1 a      # Burnt tree: gray tree",
  54 : "0036 r TRE2 a      # Large brown tree",
2028 : "07ec   COLU a      # Floor lamp",
  85 : "0055 2 TLMP abcd   # Tall techno floor lamp",
  86 : "0056 2 TLP2 abcd   # Short techno floor lamp",
  34 : "0022   CAND a        Candle",
  35 : "0023   CBRA a      # Candelabra",
  44 : "002c r TBLU abcd   # Tall blue firestick",
  45 : "002d r TGRE abcd   # Tall green firestick",
  46 : "002e   TRED abcd   # Tall red firestick",
  55 : "0037 r SMBT abcd   # Short blue firestick",
  56 : "0038 r SMGT abcd   # Short green firestick",
  57 : "0039 r SMRT abcd   # Short red firestick",
  70 : "0046 2 FCAN abc    # Burning barrel",
  41 : "0029 r CEYE abcb   # Evil Eye: floating eye in symbol, over candle",
  42 : "002a r FSKU abc    # Floating Skull: flaming skull-rock",
  49 : "0031 r GOR1 abcb  ^# Hanging victim, twitching",
  63 : "003f r GOR1 abcb  ^  Hanging victim, twitching",
  50 : "0032 r GOR2 a     ^# Hanging victim, arms out",
  59 : "003b r GOR2 a     ^  Hanging victim, arms out",
  52 : "0034 r GOR4 a     ^# Hanging pair of legs",
  60 : "003c r GOR4 a     ^  Hanging pair of legs",
  51 : "0033 r GOR3 a     ^# Hanging victim, 1-legged",
  61 : "003d r GOR3 a     ^  Hanging victim, 1-legged",
  53 : "0035 r GOR5 a     ^# Hanging leg",
  62 : "003e r GOR5 a     ^  Hanging leg",
  73 : "0049 2 HDB1 a     ^# Hanging victim, guts removed",
  74 : "004a 2 HDB2 a     ^# Hanging victim, guts and brain removed",
  75 : "004b 2 HDB3 a     ^# Hanging torso, looking down",
  76 : "004c 2 HDB4 a     ^# Hanging torso, open skull",
  77 : "004d 2 HDB5 a     ^# Hanging torso, looking up",
  78 : "004e 2 HDB6 a     ^# Hanging torso, brain removed",
  25 : "0019 r POL1 a      # Impaled human",
  26 : "001a r POL6 ab     # Twitching impaled human",
  27 : "001b r POL4 a      # Skull on a pole",
  28 : "001c r POL2 a      # 5 skulls shish kebob",
  29 : "001d r POL3 ab     # Pile of skulls and candles",
  10 : "000a   PLAY w        Bloody mess (an exploded player)",
  12 : "000c   PLAY w        Bloody mess, this thing is exactly the same as 10",
  24 : "0018   POL5 a        Pool of blood and flesh",
  79 : "004f 2 POB1 a        Pool of blood",
  80 : "0050 2 POB2 a        Pool of blood",
  81 : "0051 2 BRS1 a        Pool of brains",
  15 : "000f   PLAY n        Dead player",
  18 : "0012   POSS l        Dead former human",
  19 : "0013   SPOS l        Dead former sergeant",
  20 : "0014   TROO m        Dead imp",
  21 : "0015   SARG n        Dead demon",
  22 : "0016 r HEAD l        Dead cacodemon",
  23 : "0017 r SKUL k        Dead lost soul, invisible (they blow up when killed)",
}

class WADFile:
    """ Low-level operations for read/writing lumps, for reading and writing wads """

    def __init__(s, f):
        s.f = f

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

    def fill(s, values):
        assert len(values) == len(s.get_fields())
        for i in range(len(values)):
            field = s.get_fields()[i]
            val = values[i]
            if field[1] == 'short':
                assert type(val) == int
                setattr(s, field[0], val)
            elif field[1] == 'int':
                assert type(val) == int
                setattr(s, field[0], val)
            elif field[1] == 'string8':
                assert type(val) == str
                setattr(s, field[0], val)
        return s

    def copy(s, src):
        for field in s.get_fields():
            setattr(s, field[0], getattr(src, field[0]))

    def __repr__(s):
        rv = ''
        for field in s.get_fields():
            rv += field[0] + ':' + str(getattr(s,field[0])) + ','
        return rv

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

    def has_all_textures(s):
        return len(s.floor_pic) > 1 and len(s.ceil_pic) > 1

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
    ('sd_right', 'short'),
    ('sd_left', 'short'),
    ]

    def get_fields(s):
        return LineDef.FIELDS

    def set_flag(s, flag):
        assert flag in LineDef.FLAGBIT
        n = LineDef.FLAGBIT[flag]
        s.flags |= 1 << n
        return s

    def get_flag(s, flag):
        assert flag in LineDef.FLAGBIT
        n = LineDef.FLAGBIT[flag]
        return (s.flags & 1 << n) > 0

    def clear_flag(s, flag):
        assert flag in LineDef.FLAGBIT
        n = LineDef.FLAGBIT[flag]
        s.flags &= ~(1 << n)
        return s

    def flip_orientation(s):

        t = s.vert0
        s.vert0 = s.vert1
        s.vert1 = t

        t = s.sd_left
        s.sd_left = s.sd_right
        s.sd_right = t

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

    def has_all_textures(s):
        return len(s.uppertex) > 1 and len(s.midtex) > 1 and len(s.lowertex)> 1

    def set_clear_textures(s):
        s.midtex = '-'
        s.uppertex = '-'
        s.lowertex = '-'

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
        assert type(s.array) == list

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
        s.clear()
        s.name = name

    def __str__(s):
        return '%s: %d verts, %d sectors, %d sides, %d lines' % (s.name, len(s.verts), len(s.sectors), len(s.sidedefs), len(s.linedefs))

    def clear(s):
        s.name = None
        s.things = []
        s.verts = []
        s.linedefs = []
        s.sidedefs = []
        s.sectors = []

    def get_size(s):
        xx = [v.x for v in s.verts]
        yy = [v.y for v in s.verts]
        dx = max(xx) - min(xx)
        dy = max(yy) - min(yy)
        return (dx, dy)

    def plot(s):

        print 'plotting %d things, %d lines' % (len(s.things), len(s.linedefs))
        for t in s.things:
            pylab.plot([t.x], [t.y], '.'+get_color_for_thing(t.type))
        for ld in s.linedefs:
            p0 = s.verts[ld.vert0]
            p1 = s.verts[ld.vert1]
            color = 'k'
            if ld.sd_right >= 0 and ld.sd_left >= 0:
                assert ld.get_flag('Two-sided')
                color = '0.8'
            pylab.plot( [p0.x, p1.x], [p0.y, p1.y], '-', color=color )

        # make it square
        xx = [v.x for v in s.verts]
        yy = [v.y for v in s.verts]
        dx = max(xx) - min(xx)
        dy = max(yy) - min(yy)
        L = max(dx, dy) * 1.1
        cx = (max(xx)+min(xx))/2.0
        cy = (max(yy)+min(yy))/2.0
        aspect_ratio = 4.0/5.0
        left = cx - L/2.0/aspect_ratio
        right = cx + L/2.0/aspect_ratio
        top = cy + L/2.0
        bot = cy - L/2.0
        pylab.xlim([ left, right ])
        pylab.ylim([ bot, top ])

        print 'done'

    def unique_textures(s):
        uniqs = set()
        for sd in s.sidedefs:
            uniqs.add(sd.uppertex)
            uniqs.add(sd.lowertex)
            uniqs.add(sd.midtex)
        return uniqs

    LUMP_TO_ELEMENT_CLASS = {
        'THINGS' : Thing,
        'VERTEXES' : Vertex,
        'LINEDEFS' : LineDef,
        'SIDEDEFS' : SideDef,
        'SECTORS' : Sector,
    }

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

    def append_lumps_to(s, lumps):
        lumps += [
        DummyLump(s.name),
        ArrayLump('THINGS', s.things),
        ArrayLump('VERTEXES', s.verts),
        ArrayLump('LINEDEFS', s.linedefs),
        ArrayLump('SIDEDEFS', s.sidedefs),
        ArrayLump('SECTORS', s.sectors),
        ]

    def add_player_start(s, x, y, angle):
        t = Thing().fill([x, y, angle, 1, 0])
        s.things += [t]

    def sanity_asserts(s):
        print 'checking %d verts for dupes' % len(s.verts)
        uniqverts = set()
        for v in s.verts:
            v2 = utils.Int2(v.x, v.y)
            uniqverts.add(v2)
        assert( len(uniqverts) == len(s.verts) )
        print 'done'

# check linedefs
        for ld in s.linedefs:
            assert ld.sd_right != None and ld.sd_right >= 0
            assert ld.sd_right != ld.sd_left

        for sd in s.sidedefs:
            assert sd.sector != None and sd.sector >= 0

class WADContent:
    """ Should contain all essential contents of a WAD """

    def __init__(s):
        s.maps = []
        s.other_lumps = []
        s.end_msg = None

    def read_lumps( s, directory, wad ):
        mapp = None
        
        for entry in directory:
            wad.f.seek(entry.filepos)
            lumpend = wad.f.tell() + entry.size
            name = entry.name

            if wad.is_map_start_lump(name):
                assert entry.size == 0
                print 'reading map ' + entry.name
                mapp = Map(entry.name)
                s.maps += [mapp]

            elif mapp and mapp.handle_lump(wad, entry, lumpend):
                # no need to do anything - it handled it
                pass
                    
            elif name == 'ENDOOM':
                # sanity check
                assert entry.size == 4000
                s.end_msg = wad.f.read(4000)

            else:
                # ignore this lump
                pass

def load(path):
    """ This will yield (lumpinfo, wadfile) tuples for each lump """
    with open(path, 'rb') as f:
        wad = WADFile(f)

        header = f.read(4)
        num_lumps = wad.read_long()
        dir_offset = wad.read_long()

        assert header == 'IWAD' or header == 'PWAD'

        # read directory
        f.seek(dir_offset)
        infosize = LumpInfo().get_size()
        end = f.tell() + num_lumps * infosize
        directory = wad.read_array_lump(end, LumpInfo)

        # lumps
        rv = WADContent()
        rv.read_lumps( directory, wad )
        return rv

def save(path, header, lumps):
    with open(path, 'wb') as fout:
        """ Writes an array of lumps to a *single* WAD file, handling proper directory setup, etc. """
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
            # print 'start = %d, tell = %d' % (lumpstart, fout.tell())
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


def save_map_png(_map, fname):
    pylab.figure()
    _map.plot()
    pylab.savefig(fname)
    print 'done plotting to %s' % fname
    pylab.close()

def dump_all_maps(wadp):
    content = load(wadp)
    for m in content.maps:
        save_map_png(m, m.name + '.png')

def test_doom1_wad():
    path = dero_config.DOOM1_WAD_PATH
    content = load(path)

    assert len(content.maps) == 36
    assert content.end_msg
    _map = content.maps[0]

    # create square map
    m3 = create_square_map(_map)
    lumps = []
    m3.append_lumps_to(lumps)
    save('square.wad', 'PWAD', lumps)
    dero_config.build_wad( 'square.wad', 'square-built.wad' )

    # filter out all things except player start
    _map.things = [t for t in _map.things if t.type == 1]

    # print out all unique LD functions
    funcs = set([ ld.function for ld in _map.linedefs] )
    print 'unique functions: ' + str(funcs)

    # write the map back
    lumps = []
    _map.append_lumps_to(lumps)
    save('%s.wad' % _map.name, 'PWAD', lumps)

    # run bsp on it
    dero_config.build_wad( '%s.wad' % _map.name, 'test.wad' )

    # read it back
    cont2 = load('%s.wad' % _map.name)
    assert len(cont2.maps) == 1
    assert cont2.end_msg == None
    _map2 = cont2.maps[0]
    assert _map2.name == _map.name
    assert len(_map2.verts) == len(_map.verts)
    assert len(_map2.linedefs) == len(_map.linedefs)
    assert len(_map2.things) == 1

    # draw maps for comparison
    save_map_png( _map, 'expected.png')
    save_map_png( _map2, 'actual.png')

def create_square_map(ref):

    L = 200
    rv = Map('E1M1')
    rv.add_player_start(L/2,L/2,0)
    rv.verts = [
        Vertex().fill([0,0]),
        Vertex().fill([0,L]),
        Vertex().fill([L,L]),
        Vertex().fill([L,0]),
        Vertex().fill([0,2*L]),
        Vertex().fill([L,2*L]),
        ]

    random.seed(42)

    valid_sec_ids = [sid for sid in range(len(ref.sectors)) if ref.sectors[sid].has_all_textures()]
    ref_sec_id = random.choice(valid_sec_ids)
    ref_sec_id2 = random.choice(valid_sec_ids)
    refsec = ref.sectors[ref_sec_id]
    refsec2 = ref.sectors[ref_sec_id2]

    rv.sectors = [
        Sector().fill([0, 100,  refsec.floor_pic, refsec.ceil_pic, 200, 0, 0]),
        Sector().fill([16, 16,  refsec2.floor_pic, refsec2.ceil_pic, 200, 0, 0]),
    ]

    exit_lds = [ld for ld in ref.linedefs if ld.function == 11]
    exit_sd = ref.sidedefs[ exit_lds[0].sd_right ]
    print 'exit ld = ', exit_lds[0]
    print 'exit sd = ', exit_sd

    refsd = random.choice([sd for sd in ref.sidedefs if sd.has_all_textures()])
    refsd2 = random.choice([sd for sd in ref.sidedefs if sd.has_all_textures()])
    print 'refsd', refsd
    print 'refsd2', refsd2

    rv.sidedefs = [
        SideDef().fill([0, 0,   refsd.uppertex, refsd.lowertex, refsd.midtex, 0]), # 0
        SideDef().fill([0, 0,   refsd.uppertex, refsd.lowertex, refsd.midtex, 0]), # 1
        SideDef().fill([0, 0,   refsd.uppertex, refsd.lowertex, refsd.midtex, 0]), # 2
        SideDef().fill([0, 0,   refsd.uppertex, refsd.lowertex, '-', 0]), # 3
        SideDef().fill([0, 0,   refsd2.uppertex, refsd2.lowertex, '-', 1]), # 4
        SideDef().fill([0, 0,   refsd2.uppertex, refsd2.lowertex, refsd2.midtex, 1]), # 5
        SideDef().fill([0, 0,   refsd2.uppertex, refsd2.lowertex, refsd2.midtex, 1]), # 6
        SideDef().fill([0, 0,   refsd2.uppertex, refsd2.lowertex, refsd2.midtex, 1]), # 7
        ]

    """
    4   5

    1   2

    0   3
    """

    rv.linedefs = [
        LineDef().fill([2, 3,   0, 0, 0,    0, -1]).set_flag('Impassible'),
        LineDef().fill([3, 0,   0, 0, 0,    1, -1]).set_flag('Impassible'),
        LineDef().fill([0, 1,   0, 0, 0,    2, -1]).set_flag('Impassible'),
        LineDef().fill([1, 2,   0, 31, 0,    3, 4]).set_flag('Impassible').clear_flag('Impassible').set_flag('Two-sided'),
        LineDef().fill([1, 4,   0, 0, 0,    5, -1]).set_flag('Impassible').set_flag('Lower Unpegged'),
        LineDef().fill([4, 5,   0, 0, 0,    6, -1]).set_flag('Impassible').set_flag('Lower Unpegged'),
        LineDef().fill([5, 2,   0, 0, 0,    7, -1]).set_flag('Impassible').set_flag('Lower Unpegged'),
        ]

    print 'FOO'
    print str(exit_sd)


    return rv

if __name__ == "__main__":
    test_doom1_wad()
