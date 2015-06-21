
import random
from planar import Vec2

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

