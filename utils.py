
import numpy
import pylab
import random
import noise
from planar import Vec2
from Queue import *
import sys

vonNeumannNhoodDX = [1, 0, -1, 0]
vonNeumannNhoodDY = [0, 1, 0, -1]

nbor8dx = [1, 1, 0, -1, -1, -1, 0, 1]
nbor8dy = [0, 1, 1, 1, 0, -1, -1, -1]

def char_times(c, x):
    rv = ''
    for i in range(x):
        rv += c
    return rv

def add2(a,b):
    return (a[0]+b[0], a[1]+b[1])

class Int2:
    x = 0
    y = 0

    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x,self.y))

    def __repr__(self):
        return '%d,%d' % (self.x, self.y)

    def __add__(u,v):
        return Int2(u.x+v.x, u.y+v.y)

    def __sub__(u,v):
        return Int2(u.x-v.x, u.y-v.y)

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

class Grid2:
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
# first compute widest string
        maxlen = 0
        for (u,x) in self.piter():
            maxlen = max( len(str(x)), maxlen )

        yy = range(self.H)
        yy.reverse()
        for y in yy:
            rowstr = ''
            for x in range(self.W):
                val = self.get(x,y)
                rowstr += char_times(' ', maxlen-len(str(val))+1)
                rowstr += str(val)
            print rowstr

    def unique_values(self):
        vals = set()
        for (u,x) in self.piter():
            if not x in vals:
                vals.add(x)
        return vals

    def show_image_scalar(self, minval, maxval):
        matrix = numpy.ndarray([self.W,self.H])
        for (u,val) in self.piter():
            matrix[self.H-u.y-1, u.x] = (val-minval)/(maxval-minval)

        pylab.imshow(matrix, interpolation='nearest')
# pylab.imshow(matrix)

    def show_image(self):
        matrix = numpy.ndarray([self.W,self.H])
        scalarVal = 0.0
        val2scalar = {}
        uniques = self.unique_values()
        for val in uniques:
            val2scalar[val] = scalarVal
            scalarVal += 1.0/len(uniques)

        for (u,x) in self.piter():
            matrix[self.H-u.y-1, u.x] = val2scalar[x]

        pylab.imshow(matrix, interpolation='nearest')
# pylab.imshow(matrix)

    def iter(self):
        for y in range(self.W):
            for x in range(self.H):
                yield (x,y)

    def piter(self):
        for y in range(self.W):
            for x in range(self.H):
                yield (Int2(x,y), self.get(x,y))

    def piter_rand(self):
        for y in numpy.random.permutation(self.W):
            for x in numpy.random.permutation(self.H):
                yield (Int2(x,y), self.get(x,y))

    def nbors4(self, p):
        for nbor in p.yield_4nbors():
            if self.check(nbor):
                yield (nbor, self.pget(nbor))

    def nbors4_rand(self, p):
        for nbor in p.yield_4nbors_rand():
            if self.check(nbor):
                yield (nbor, self.pget(nbor))

    def free_cells_adjacent_to(self, freeval, valueset):
        rv = []
        for (p,a) in self.piter():
            if a != freeval: continue
            for (q,b) in self.nbors4(p):
                if b in valueset:
                    rv += [p]
                    break
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

    def cells_with_value(self, value):
        for (p,x) in self.piter():
            if x == value:
                yield p

    def tally(self):
        table = {}
        for (p,x) in self.piter():
            if not x in table:
                table[x] = 0
            table[x] += 1
        return table

    def duplicate(self):
        rv = Grid2(self.W, self.H, None)
        for (p,x) in self.piter():
            rv.pset(p,x)
        return rv

    def connected_components_grid(self, valueFilter):
        C = Grid2(self.W, self.H, -1)
        def helper(u, cid, value):
            count = 0
            if C.pget(u) == -1 and self.pget(u) == value:
                C.pset(u, cid)
                count += 1
                for (v,_) in self.nbors4(u):
                    count += helper(v, cid, value)
            return count

        compid = 10
        compsizes = {}
        for (u,value) in self.piter():
            if valueFilter and value != valueFilter:
                continue
            size = helper(u, compid, value)
            if size > 0:
                compsizes[compid] = size
                compid += 1

        return (C, compsizes)

    def replace(self,q,new):
        for (u, value) in self.piter():
            if value == q:
                self.pset(u, new)

    def piter_outside_radius(self, r):
        cx = self.W/2.0
        cy = self.H/2.0
        for (u, value) in self.piter():
            dx = u.x+0.5 - cx
            dy = u.y+0.5 - cy
            if (dx*dx + dy*dy) > r*r:
                yield (u, value)

    def value_adjacency(G):
        rv = {}
        # randomize order so we don't bias adjacency locations
        for (p,a) in G.piter_rand():
            for (q,b) in G.nbors4_rand(p):
                if a == b: continue
                # strict order
                if a > b:
                    if not (b,a) in rv:
                        rv[(b,a)] = (q,p)
                else:
                    if not (a,b) in rv:
                        rv[(a,b)] = (p,q)
        return rv

    def compute_centroids(G):
        value2sum = {}
        value2count = {}
        for (u,x) in G.piter():
            if not x in value2sum:
                value2sum[x] = Int2(0,0)
            value2sum[x] = value2sum[x] + u

            if not x in value2count:
                value2count[x] = 0
            value2count[x] = value2count[x] + 1

        value2cent = {}
        for x in value2sum:
            s = value2sum[x]
            t = value2count[x]
            value2cent[x] = (s.x*1.0/t, s.y*1.0/t)

        return value2cent


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
    D = Grid2(N,N, 0.0)
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

def pick_random_from_set(s):
    return pick_random([x for x in s])

class FrontManager:
    grid = None
    freevalue = None
    frontcells = set()
    invalues = None

    def __init__(self, grid, freevalue, invalues):
        self.grid = grid
        self.freevalue = freevalue
        self.invalues = invalues

    def recompute(self):
        self.frontcells = set()
        for u in self.grid.cells_with_value(self.freevalue):
            for (v,y) in self.grid.nbors4(u):
                if y in self.invalues:
                    self.frontcells.add(u)
                    break

    def on_fill(self, u):
        if self.grid.pget(u) == self.freevalue:
            print 'ERROR'
        if not self.grid.pget(u) not in self.freevalue:
            print 'ERROR'

        if u in self.frontcells:
            self.frontcells.remove(u)
        for (v,val) in self.grid.nbors4(u):
            if val == self.freevalue and v not in self.frontcells:
                self.frontcells.add(v)

    def sample(self):
        return random.sample(self.frontcells, 1)[0]

    def size(self):
        return len(self.frontcells)

    def check(self):
        for u in self.frontcells:
            found = False
            for (v,y) in self.grid.nbors4(u):
                if y != self.freevalue:
                    found = True
                    break
            if not found:
                print 'ERROR: '+u

def seed_spread(seedvals, sews, G, freevalue, maxspreads):
    seedvalset = set(seedvals)
    front = FrontManager(G, freevalue, seedvalset)
    front.recompute()
    front.check()

    # initial seedings
    freespots = [x for x in G.cells_with_value(freevalue)]
    random.shuffle(freespots)
    freeid = 0
    for sew in range(sews):
        for val in seedvals:
            u = freespots[freeid]
            freeid += 1
            G.pset(u, val)
            front.on_fill(u)
    front.check()

    # spread iteration
    spreads = 0
    while front.size() > 0 and spreads < maxspreads:
        if spreads % 100 == 0:
            print '\r %d' %spreads,
        # spread
        u = front.sample()
        # choose a random region, which nbors this front cell, to spread to it
        filled = False
        for (v, val) in G.nbors4_rand(u):
            if val in seedvalset:
                G.pset(u, val)
                front.on_fill(u)
                spreads += 1
                filled = True
                break

        if not filled:
            print 'ERROR!'
            print u
            for (v,val) in G.nbors4(u):
                print v, val
            print front.array
            G.write()
            sys.exit(1)

    return spreads

def eval_subtree_sizes(G, root):
    sizes = {}
    def recurse(u):
        count = 1 # count node itself
        for e in G.edges([u]):
            recurse(e[1])
            count += sizes[e[1]]
        sizes[u] = count

    recurse(root)
    return sizes

def find_ancestor_family(G, leaf, maxsize, sizes):
    u = leaf
    while True:
        parent = get_parent(G, u)
        if parent == None:
            return u
        else:
            if sizes[parent] > maxsize:
                return u
            else:
                u = parent


def yield_dfs(G, root, stopset):
    if stopset and root in stopset:
        return
    yield root
    for e in G.out_edges(root):
        v = e[1]
        for u in yield_dfs(G,v, stopset):
            yield u

def copy_graph_without_subtree(G, root, subroot):
    keeps = set()
    u = root
    for node in yield_dfs(G, root, set([subroot])):
        keeps.add(node)
    return G.subgraph(keeps)

def get_parent(T, node):
    for u in T.in_edges(node):
        return u[0]
    return None

