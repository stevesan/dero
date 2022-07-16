
import os
import subprocess as sp

DOOM1_WAD_PATH = '/Users/stevenan/dooming/wads/doom1/DOOM.WAD'
DOOM2_WAD_PATH = '/Users/stevenan/dooming/wads/doom2/DOOM2.WAD'

def build_wad( srcwad, destwad ):
    """
    Compile bsp5.2 from source: http://games.moria.org.uk/games/doom/bsp/
    TLDR: ./configure && make
    then bsp binary is in the bsp5.2 folder.
    """
    sp.check_call(['/Users/stevenan/Downloads/bsp-5.2/bsp',
        srcwad,
        '-o',  destwad])
    print(f'OK built wad into {destwad}')
