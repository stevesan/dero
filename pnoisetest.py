
from utils import *
import pylab as pl
import noise
import math

L = 400
G = Grid2(L, L, 0)
S = 10.0/L

minval = 999999.0
maxval = -999999
for (u,_) in G.piter():
    x = u.x * S
    y = u.y * S
    val = noise.pnoise2(x, y)

# val = min(val, -0.2)
# val = noise.pnoise1(x)
    G.pset(u, val)

    minval = min(minval, val)
    maxval = max(maxval, val)

print(minval, maxval)

G.show_image_scalar(minval, maxval)
pl.show()

