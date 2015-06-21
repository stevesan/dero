
import numpy
import random
from planar import Vec2

vonNeumannNhoodDX = [1, 0, -1, 0]
vonNeumannNhoodDY = [0, 1, 0, -1]

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

    def __repr__(self):
        return '%d,%d' % (self.x, self.y)

    def yieldtest(self):
        yield 1
        yield 2

    def yield_4nbors(self):
        """ in ccw order """
        p = self
        for i in range(4):
            yield Int2(p.x+vonNeumannNhoodDX[i], p.y+vonNeumannNhoodDY[i])

    def yield_4nbors_rand(self):
        """ in random order """
        p = self
        for i in numpy.random.permutation(4):
            yield Int2(p.x+vonNeumannNhoodDX[i], p.y+vonNeumannNhoodDY[i])

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

    def write(self):
        for y in range(self.H):
            for x in range(self.W):
                print self.get(x,y),
            print ''

    def iter(self):
        for y in range(self.W):
            for x in range(self.H):
                yield (x,y)

    def piter(self):
        for y in range(self.W):
            for x in range(self.H):
                yield (Int2(x,y), self.get(x,y))

    def nbors4(self, p):
        for nbor in p.yield_4nbors():
            if self.check(nbor):
                yield (nbor, self.pget(nbor))

    def nbors4_rand(self, p):
        for nbor in p.yield_4nbors_rand():
            if self.check(nbor):
                yield (nbor, self.pget(nbor))

    def cells_adjacent_to(self, freeval, valueset):
        rv = []
        for (p,a) in self.piter():
            if a != freeval: continue
            if a in valueset: continue
            for (q,b) in self.nbors4(p):
                if b in valueset:
                    rv += [ (p,q) ]
        return rv

    def set_border(self, val):
        H = self.H
        W = self.W
        G = self
        for y in range(H):
            G.set(0, y, val)
            G.set(W-1, y, val)
        for x in range(W):
            G.set(x, 0, val)
            G.set(x, H-1, val)

    def cells_with_values(self, valueset):
        rv = []
        for (p,x) in self.piter():
            if x in valueset:
                rv += [p]
        return rv

def vec2_dist(a,b): return (a-b).length
                    
def assign_nearest_center(points, centers):
    assignment = range(len(points))

    for i in range(len(points)):
        p = points[i]
        bestj = -1
        bestdist = 0.0
        for j in range(len(centers)):
            dist = vec2_dist(p,centers[j])
            if bestj == -1 or dist < bestdist:
                bestj = j
                bestdist = dist
        assignment[i] = bestj
    return assignment

def slow_poisson_sampler(mindist, numpoints):
    points = []
    while len(points) < numpoints:
        while True:
            p = Vec2(random.random(), random.random())
            bad = False
            for q in points:
                if vec2_dist(p,q) < mindist:
                    bad = True
                    break
            if not bad:
                points += [p]
                break
    return points

def distance_grid(poss):
    N = len(poss)
    D = Grid(N,N, 0.0)
    for u in range(N):
        for v in range(N):
            pu = poss[u]
            pv = poss[v]
            D.set(u,v, (pu-pv).length)
    return D

def gaussian(x, a, b, c):
    return a * math.exp(-1 * (x-b)*(x-b)/(2*c*c))

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

def pick_random(l):
    return l[ random.randint(0, len(l)-1) ]

def seed_spread(seedvals, sews, G, freecell, maxspreads):
    seedvalset = set()
    for v in seedvals: seedvalset.add(v)

    freespots = []
    for (x,y) in G.iter():
        if G.get(x,y) == freecell:
            freespots += [Int2(x,y)]

    # initial seedings
    for sew in range(sews):
        for i in range(len(seedvals)):
            u = pick_random(freespots)
            freespots.remove(u)
            G.pset(u, seedvals[i])

    # spread iteration
    spreads = 0
    while len(freespots) > 0 and spreads < maxspreads:
        if spreads % 10 == 0:
            print '%d/%d' % (spreads, maxspreads)
        # compute front
        front = []
        for p in freespots:
            nextToSpread = False
            for (q, val) in G.nbors4(p):
                if val in seedvalset:
                    nextToSpread = True
                    break
            if nextToSpread:
                front += [p]

        if len(front) <= 0:
            break

        # spread
        p = pick_random(front)
        # which seed would spread here?
        for (q, val) in G.nbors4_rand(p):
            if val in seedvalset:
                G.pset(p, val)
                spreads += 1
                break
        freespots.remove(p)

def value_adjacency(G):
    rv = {}

# TODO randomize iteration order, so we don't bias access points towards origin
    for (p,a) in G.piter():
        for (q,b) in G.nbors4(p):
            if a == b: continue
            if not (a,b) in rv:
                rv[(a,b)] = (p,q)

    return rv

