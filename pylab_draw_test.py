
import pylab as pl
import math

for r in [0.5, 0.6, 0.7]:
    N = 20
    xx = []
    yy = []
    for i in range(N):
        t = math.pi*2.0/N * i
        x = r*math.cos(t)
        y = r*math.sin(t)
        xx += [x]
        yy += [y]
    xx += [xx[0]]
    yy += [yy[0]]
    pl.plot(xx, yy, 'r.-')

pl.grid(True)
pl.xlim([-1,1])
pl.ylim([-1,1])
pl.show()

