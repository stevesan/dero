
import os

DOOM1_WAD_PATH = '/Users/Steve/Documents/Zandronum/DOOM.WAD'
DOOM2_WAD_PATH = '/Users/Steve/Documents/Zandronum/DOOM2.WAD'

def build_wad( srcwad, destwad ):
    os.system( '/Users/Steve/bsp-5.2/bsp %s -o %s' % (srcwad, destwad) )
