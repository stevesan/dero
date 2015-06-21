import random
import numpy
import networkx as nx
import pylab
import math
import sys
from planar import Vec2


nbor4dx = [1, 0, -1, 0]
nbor4dy = [0, 1, 0, -1]

nbor8dx = [1, 1, 0, -1, -1, -1, 0, 1]
nbor8dy = [0, 1, 1, 1, 0, -1, -1, -1]

class Int2:
    x = 0
    y = 0

    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def yieldtest(self):
        yield 1
        yield 2

    def yield_4nbors(self):
        """ in ccw order """
        p = self
        for i in range(4):
            yield Int2(p.x+nbor4dx[i], p.y+nbor4dy[i])

    def yield_8nbors(self):
        """ in ccw order """
        p = self
        for i in range(8):
            yield Int2(p.x+nbor8dx[i], p.y+nbor8dy[i])

    def yield_8nbors_rand(self):
        """ in random order """
        p = self
        for i in numpy.random.permutation(8):
            yield Int2(p.x+nbor8dx[i], p.y+nbor8dy[i])

    def astuple(self): return (self.x, self.y)

class Grid:
    W = 1
    H = 1
    grid = None

    def __init__(self,_W, _H, default):
        self.W = _W
        self.H = _H
        self.grid = range(self.W*self.H)
        for i in range(self.W*self.H):
            self.grid[i] = default

    def check(self,p):
        return p.x >= 0 and p.x < self.W and p.y >= 0 and p.y < self.H

    def get(self,x,y):
        return self.grid[self.W*y + x]

    def pget(self,p):
        return self.grid[self.W*p.y + p.x]

    def set(self,x,y,value):
        self.grid[self.W*y+x] = value

    def pset(self,p,value):
        self.grid[self.W*p.y+p.x] = value

    def printgrid(self):
        for y in range(self.H):
            for x in range(self.W):
                print self.get(x,y),
            print ''

    def iter(self):
        for y in range(self.W):
            for x in range(self.H):
                yield (x,y)

def testBasic():
    g = Grid(80,80,' ')

    for (x,y) in g.iter():
        if random.random() < 0.2:
            g.set(x,y,'X')

    g.printgrid()


def gaussian(frac, stdev):
    return math.exp(-1 * (frac*frac)/(stdev*stdev))

class InterestCurve:
    firstBump = 0.5
    lastBump = 1.0
    def eval(self, frac):
        rv = 0
        rv += self.firstBump * gaussian(frac-0.1, 0.10)
        for center in [ 0.3, 0.5, 0.7]:
            rv += (center*0.8) * 1.0 * gaussian(frac-center, 0.075)
        rv += 1.0 * gaussian(frac-0.95, 0.1)
        return rv

curve = InterestCurve()
ff = range(200)
ii = range(200)
for i in range(len(ff)):
    ff[i] = i*1.0/(len(ff)-1)
# ii[i] = gaussian(ff[i], 0.2)
    ii[i] = curve.eval(ff[i])

# pylab.plot(ff,ii,'.-')
# pylab.xlim([0, 1])
# pylab.ylim([0, 1])
# show()

def tunnel(g, p, c, numdigs, maxdigs):
    # only dig if one nbor only is 
    foundOne = False
    for nbor in p.yield_8nbors():
        if not g.check(nbor): continue
        if g.pget(nbor) == c:
            if foundOne:
                return
            else:
                foundOne = True
    # ok, dig it
    g.pset(p, c)

    numdigs += 1
    if numdigs >= maxdigs:
        return

    # recurse
    for nbor in p.yield_8nbors_rand():
        if not g.check(nbor): continue
        if g.pget(nbor) != c:
            tunnel(g, nbor, c, numdigs, maxdigs)

# g = Grid(40,40,' ')
# tunnel(g, Int2(0, 0), 'X', 0, 20)
# g.printgrid()

G = nx.Graph()
NV = 20
numGroups = 3
groupSize = NV/numGroups

nodePos = {}
for i in range(NV):
    G.add_node(i)
    nodePos[i] = Vec2(random.random(), random.random())

# compute distances
D = Grid(NV,NV, 0.0)
for u in range(NV):
    for v in range(NV):
        pu = nodePos[u]
        pv = nodePos[v]
        D.set(u,v, (pu-pv).length)

nodeGroup = {}

def key_func(v):
    if v in nodeGroup:
        return sys.float_info.max
    else:
        return D.get

for u in range(NV):
    if u in nodeGroup: continue
    nbors = range(NV)
    nbors.sort(key=lambda v: D.get(u,v) if v not in nodeGroup else sys.float_info.max)
    for i in range(0,groupSize+1):
        v = nbors[i]
        if v in nodeGroup: continue
        nodeGroup[v] = u
        print u,v
        G.add_edge(u,v)

drawPos = {}
for node in nodePos:
    vec = nodePos[node]
    drawPos[node] = (vec.x, vec.y)
nx.draw(G, drawPos)
pylab.show()
