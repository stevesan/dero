
import numpy
import math
import pylab
import random
import noise
from planar import Vec2
from Queue import *
import sys
from euclid import *
import wad

def char_times(c, x):
    rv = ''
    for i in range(x):
        rv += c
    return rv

def add2(a,b):
    return (a[0]+b[0], a[1]+b[1])

def unordered_equal(t0, t1):
    return (t0[0] == t1[0] and t0[1] == t1[1]) or (t0[0] == t1[1] and t0[1] == t1[0])

class IntMatrix2:
    def __init__(self, elems):
        """ elems should be row-major list of elements, ie. (e_00, e_01, e_10, e_11) """
        self.elems = elems

    def transform(m, u):
        return Int2(
                m.elems[0] * u.x + m.elems[1] * u.y,
                m.elems[2] * u.x + m.elems[3] * u.y)

    @staticmethod
    def new_rotation(rads):
        elems = [
            int(math.cos(rads)), int(-1*math.sin(rads)),
            int(math.sin(rads)), int(math.cos(rads))
        ]
        return IntMatrix2(elems)

    @staticmethod
    def new_scale(s):
        elems = [
            s, 0.0,
            0.0, s
        ]
        return IntMatrix2(elems)

INT2_CCW_QUARTER_ROT_MATRICES = [ IntMatrix2.new_rotation(math.pi/2*i) for i in range(4) ]
VECTOR2_CCW_QUARTER_ROT_MATRICES = [ Matrix3.new_rotate(math.pi/2*i) for i in range(4) ]

class Int2:
    x = 0
    y = 0

    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self == other

    def __cmp__(self, other):
        if self.x < other.x: return -1
        elif self.x > other.x: return 1
        elif self.y < other.y: return -1
        elif self.y > other.y: return 1
        else: return 0

    def __hash__(self):
        return hash((self.x,self.y))

    def __repr__(self):
        return 'Int2(%d,%d)' % (self.x, self.y)

    def __add__(u,v):
        return Int2(u.x+v.x, u.y+v.y)

    def __sub__(u,v):
        return Int2(u.x-v.x, u.y-v.y)

    def __div__(u, s):
        return Int2(u.x/s, u.y/s)

    def __mul__(u, s):
        return Int2(u.x*s, u.y*s)

    def scale(u, v):
        return Int2(u.x * v.x, u.y * v.y)

    def with_y(u, y):
        return Int2(u.x, y)

    def with_x(u, x):
        return Int2(x, u.y)

    def abs(u):
        return Int2(abs(u.x), abs(u.y))

    def dot(u,v):
        return u.x*v.x + u.y*v.y

    def __getitem__(u, index):
        if index >= 2:
            raise ValueError('Only index values of 0 and 1 are supported by Int2. Index given: %d' % index)
        if index == 0:
            return u.x
        else:
            return u.y

    def yieldtest(self):
        yield 1
        yield 2

    def yield_4nbors(self):
        """ in ccw order """
        p = self
        for i in range(4):
            yield p + EDGE_TO_NORM[i]

    def yield_4nbors_rand(self):
        """ in random order """
        p = self
        for i in numpy.random.permutation(4):
            yield p + EDGE_TO_NORM[i]

    def yield_8nbors(self):
        """ in ccw order """
        p = self
        for i in range(8):
            yield Int2(p.x+nbor8dx[i], p.y+nbor8dy[i])

    def yield_9square(self):
        """ in ccw order """
        for nbor in self.yield_8nbors():
            yield nbor
        yield self

    def yield_8nbors_rand(self):
        """ in random order """
        p = self
        for i in numpy.random.permutation(8):
            yield Int2(p.x+nbor8dx[i], p.y+nbor8dy[i])

    def astuple(self):
        return (self.x, self.y)

    def turn(self, turns):
        """ CCW 90-degree multiple turn """
        return INT2_CCW_QUARTER_ROT_MATRICES[turns].transform(self)

    def avg_dist(u, vs):
        d = 0.0
        for v in vs:
            d += Int2.euclidian_dist(u, v)
        d /= len(vs)
        return d

    @staticmethod
    def floor(v2):
        return Int2(
                int(math.floor(v2.x)),
                int(math.floor(v2.y)) )

    @staticmethod
    def manhattan_dist(u, v):
        return abs(v.x - u.x) + abs(v.y - u.y)

    @staticmethod
    def euclidian_dist(u, v):
        dx = v.x - u.x
        dy = v.y - u.y
        return math.sqrt(dx*dx + dy*dy)

    @staticmethod
    def centroid(coords):
        summ = Int2(0,0)
        for c in coords:
            summ = summ + c
        return summ / len(coords)

    @staticmethod
    def incrange(a, b):
        """ inclusive range """
        for x in range(a.x, b.x+1):
            for y in range(a.y, b.y+1):
                yield Int2(x,y)

class Grid2:
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

    def printself(self):
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
        for idx in numpy.random.permutation(self.W * self.H):
            y = idx / self.W
            x = idx % self.W
            yield (Int2(x,y), self.get(x,y))

    def nbors4(self, u):
        for nbor in u.yield_4nbors():
            if self.check(nbor):
                yield (nbor, self.pget(nbor))

    def nbors8(self, u):
        for nbor in u.yield_8nbors():
            if self.check(nbor):
                yield (nbor, self.pget(nbor))

    def nbors4_rand(self, u):
        for nbor in u.yield_4nbors_rand():
            if self.check(nbor):
                yield (nbor, self.pget(nbor))

    def touches4(self, u, val):
        for (v, q) in self.nbors4(u):
            if q == val:
                return True
        return False

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
        for (p,x) in self.piter():
            if x in valueset:
                yield p

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
        """ computes connected components, returning (C,S), where C is a grid of cell->component id, and S[componet id] -> num cells in the component """
        C = Grid2(self.W, self.H, -1)
        def helper(u, cid, value):
            count = 0
            if C.pget(u) == -1 and self.pget(u) == value:
                C.pset(u, cid)
                count += 1
                for (v,_) in self.nbors4(u):
                    count += helper(v, cid, value)
            return count

        compid = 0
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

    def separate(G, separator_value, exclude_cell):
        H = Grid2(G.W, G.H, None)
        for (u, p) in G.piter_rand():
            H.pset(u, p)
            if p == separator_value:
                continue
            if exclude_cell and exclude_cell(u):
                continue
            for (v, q) in G.nbors8(u):
                if q == separator_value:
                    continue
                elif p == q:
                    continue
                else:
                    H.pset(u, separator_value)
                    break
        return H

    def integer_supersample(G, factor):
        S = factor
        H = Grid2(G.W*S, G.H*S, None)

        for (u, p) in H.piter():
            H.pset(u, G.pget(u/S))

        return H

    def select(G, use_value):
        for (u, p) in G.piter():
            if use_value(p):
                yield (u, p)

    def nbor8_values(G, u):
        touch_vals = set()
        for (v,q) in G.nbors8(u):
            touch_vals.add(q)
        return touch_vals

    def nbor4_values(G, u):
        touch_vals = set()
        for (v,q) in G.nbors4(u):
            touch_vals.add(q)
        return touch_vals

    def size(s):
        return Int2(s.W, s.H)

    def floodfill(s, u, freeval, fillval):
        assert freeval != fillval
        queue = Queue()
        queue.put(u)
        while not queue.empty():
            u = queue.get()
            if s.pget(u) != freeval:
                # already visited
                continue
            s.pset(u, fillval)
            for (v, q) in s.nbors4(u):
                if q == freeval:
                    queue.put(v)

    @staticmethod
    def new_same_size(other, default_val):
        g = Grid2(other.W, other.H, default_val)
        return g

EDGE_TO_NORM = [
    Int2(1, 0),
    Int2(0, 1),
    Int2(-1, 0),
    Int2(0, -1)
]

NORM_TO_EDGE = { EDGE_TO_NORM[edge] : edge for edge in range(4) }

nbor8dx = [1, 1, 0, -1, -1, -1, 0, 1]
nbor8dy = [0, 1, 1, 1, 0, -1, -1, -1]

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
    def __init__(self, grid, freevalue):
        self.grid = grid
        self.freevalue = freevalue
        self.frontcells = None

    def recompute(self, invalues):
        self.frontcells = set()
        for u in self.grid.cells_with_value(self.freevalue):
            for (v,y) in self.grid.nbors4(u):
                if y in invalues:
                    self.frontcells.add(u)
                    break

    def on_fill(self, u):
        assert self.grid.pget(u) != self.freevalue
        if u in self.frontcells:
            self.frontcells.remove(u)
        for (v,val) in self.grid.nbors4(u):
            if val == self.freevalue and v not in self.frontcells:
                self.frontcells.add(v)

    def on_off_limits(self, u):
        """ Like on_fill, but will not expand the front """
        if u in self.frontcells:
            self.frontcells.remove(u)

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
            assert found

def seed_spread(seedvals, sews, G, freevalue, max_spreads):
    seedvalset = set(seedvals)
    front = FrontManager(G, freevalue)
    front.recompute(seedvalset)
    front.check()

    def only_touches_values(u, values):
        for (v,q) in G.nbors8(u):
            if q not in values:
                return False
        return True

    # initial seedings
    freespots = [x for x in G.cells_with_value(freevalue)]
    random.shuffle(freespots)
    for sew in range(sews):
        if len(freespots) == 0:
            break
        for val in seedvals:
            if len(freespots) == 0:
                print 'WARNING: Ran out of free spots before seed sewing'
                break
            u = freespots.pop()
            # make sure this keeps regions separate
            while not only_touches_values(u, (freevalue,)):
                u = freespots.pop()
            G.pset(u, val)
            front.on_fill(u)

    front.check()

    sepvalue = '/'
    assert sepvalue not in seedvalset

    # spread iteration
    spreads = 0
    while front.size() > 0 and spreads < max_spreads:
        spreads += 1
        if spreads % 100 == 0:
            print '%d/%d' % (spreads, max_spreads)
        # spread
        u = front.sample()

        # do NOT spread to this if it is separating
        touched_regions = set()
        for (v, q) in G.nbors8(u):
            if q in seedvalset:
                touched_regions.add(q)

        if len(touched_regions) > 1:
            # 'fill' this with a free value, and mark it as filled so no one uses this
            G.pset(u, sepvalue)
            front.on_off_limits(u)
            continue

        if len(touched_regions) == 0:
            # must be bordering existing border or separator
            continue

        # ok, but we can only spread to a 4-nbor
        for (v, q) in G.nbors4(u):
            if q in seedvalset:
                G.pset(u, q)
                front.on_fill(u)
                break

    G.replace(sepvalue, freevalue)

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


def yield_ancestors(G, start):
    u = start
    while u:
        yield u
        u = get_parent(G, u)

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

def pick_max(items, score_func):
    best = None
    best_score = 0
    for item in items:
        score = score_func(item)
        if best == None or score > best_score:
            best = item
            best_score = score
    return best
        
def pick_min(items, score_func):
    best = None
    best_score = 0
    for item in items:
        score = score_func(item)
        if best == None or score < best_score:
            best = item
            best_score = score
    return best
        
def asc(a,b):
    if b < a:
        return (b, a)
    else:
        return (a, b)
        
def asc2(t):
    a = t[0]
    b = t[1]
    if b < a:
        return (b, a)
    else:
        return (a, b)

TEST_POLYS = [
    [
        (0,0),
        (0.5, 1),
        (1,0),
        (0,0,)
    ],
    [
        (1,0),
        (1.5, 1),
        (2,0),
        (1,0,)
    ],
    ]

def turned_center_offset(u, offset, turns):
    center = Vector2(u.x+0.5, u.y+0.5)
    return center + VECTOR2_CCW_QUARTER_ROT_MATRICES[turns] * offset

def left_vert(u, edge):
    return turned_center_offset(u, Vector2(0.5, 0.5), edge)

def right_vert(u, edge):
    return turned_center_offset(u, Vector2(0.5, -0.5), edge)

def polygonate(G, is_in_val, oob_is_inside, on_edge):
    polys = []
    edge_done_grid = Grid2(G.W, G.H, 0)

    def is_in(u):
        if not G.check(u):
            return oob_is_inside
        else:
            return is_in_val(G.pget(u))

    def trace_polygon(u, edge, poly, poly_id):
        while True:
            dones = edge_done_grid.pget(u)
            if (dones & (1 << edge)) > 0:
                print 'done'
                break

            # the last one in the chain will take of our left-vert, so don't worry about it
            poly += [ right_vert(u, edge) ]
            if on_edge: on_edge(u, u+EDGE_TO_NORM[edge], poly_id, len(poly)-2)
# print u, edge
# print poly
            new_done = dones | (1<<edge)
            edge_done_grid.pset(u, new_done)

            v_right = u + Int2(0,-1).turn(edge)
            if not is_in(v_right):
                edge = (edge-1)%4
            else:
                v_right_fwd = u + Int2(1,-1).turn(edge)
                if is_in(v_right_fwd):
                    u = v_right_fwd
                    edge = (edge+1)%4
                else:
                    u = v_right
                    edge = edge

    for (u,val) in G.piter():
        if not is_in(u):
            continue
        for edge in range(4):
            dones = edge_done_grid.pget(u)
            if (dones & (1 << edge)) > 0:
                continue
            v = u + EDGE_TO_NORM[edge]
            if is_in(v):
                continue
            poly = []
            polys += [poly]
            trace_polygon(u, edge, poly, len(polys)-1)

    return polys
# return TEST_POLYS

def plot_poly(verts, style):
    xx = [v.x for v in verts] + [verts[0].x]
    yy = [v.y for v in verts] + [verts[0].y]
    pylab.plot( xx, yy, style )

def get_wrap(array, i):
    return array[ i % len(array) ]

def linear_simplify_poly(poly):
    """ A pretty efficient, linear simplification. Is NOT optimal by any means, but probably takes care of 90% of simplifications that should be done """
    if len(poly) < 4:
        return poly

    q = Queue()
    for v in poly:
        q.put(v)

    new_poly = []
    a = q.get()
    b = q.get()
    while True:
        if q.empty():
            new_poly += [a,b]
            break
        c = q.get()
        e1 = (b-a).normalized()
        e2 = (c-b).normalized()
        if abs(1.0 - e1.dot(e2)) < 1e-2:
            # colinear. skip b.
            a = a
            b = c
        else:
            # a,b needed.
            new_poly += [a]
            a = b
            b = c
    return new_poly

class GridEdges2:

    def __init__(s, W, H, default):
        s.leftlines = Grid2(W+1, H+1, default)
        s.botlines = Grid2(W+1, H+1, default)

    def get_grid_cell(s, u, edge ):
        if edge == 0:
            return (s.leftlines, u+Int2(1,0))
        elif edge == 1:
            return (s.botlines, u+Int2(0,1))
        elif edge == 2:
            return (s.leftlines, u)
        else:
            return (s.botlines, u)

    def get(s, u, edge):
        (grid, ut) = s.get_grid_cell(u, edge)
        return grid.pget(ut)

    def set(s, u, edge, value):
        (grid, ut) = s.get_grid_cell(u, edge)
        grid.pset( ut, value )

    def get_between(s, u, v):
        norm = v - u
        edge = NORM_TO_EDGE[norm]
        return s.get(u, edge)

class GridVerts2:
    def __init__(s, W, H, default):
        s.verts = Grid2(W+1, H+1, default)

    EDGE_RIGHT_OFFSET = [
        Int2(1,0),
        Int2(1,1),
        Int2(0,1),
        Int2(0,0)
    ]

    def get_right(s, u, edge):
        return s.verts.pget(u + GridVerts2.EDGE_RIGHT_OFFSET[edge])

    def set_right(s, u, edge, value):
        s.verts.pset( u + GridVerts2.EDGE_RIGHT_OFFSET[edge], value )

    def get_left(s, u, edge):
        return s.get_right(u, (edge + 1) % 4)

    def set_left(s, u, edge, value):
        return s.set_right(u, (edge + 1) % 4, value)

def id_iter(listt):
    for i in range(len(listt)):
        yield (i, listt[i])

def flip(a,b):
    return (b,a)

def flip2(v):
    return ( v[1], v[0] )

class figure_to_png:
    def __init__(s, pngf):
        s.pngf = pngf

    def __enter__(s):
        pylab.figure()

    def __exit__(s, type, value, traceback):
        pylab.grid(True)
        pylab.savefig(s.pngf)
        pylab.close()

def argmin(xx):
    b = None
    for i in range(len(xx)):
        if not b or xx[i] < xx[b]:
            b = i
    return b

def argmax(xx):
    b = None
    for i in range(len(xx)):
        if not b or xx[i] > xx[b]:
            b = i
    return b

def save_grid_png(G, path):
    pylab.figure()
    G.show_image()
    pylab.savefig(path)
    pylab.close()

def ring_pairs(v):
    """ calling on [A, B, C] would yield AB, BC, and CA """
    for i in range(len(v)):
        a = v[i]
        b = v[ (i+1) % len(v) ]
        yield (a,b)


# Shamelessly copied from https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain#Python
def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1] + upper[:-1]


def line_points(start, end):
    """
    Bresenham's Line Algorithm
    Produces a list of tuples from start and end
    Taken from: http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm#Python
    """
    # Setup initial conditions
    x1, y1 = start.astuple()
    x2, y2 = end.astuple()
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = Int2(y, x) if is_steep else Int2(x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

import timeit

class profileit:
    LEVEL = 0
    def __init__(s, label):
        s.label = label

    def __enter__(s):
        profileit.LEVEL += 1
        s.t0 = timeit.default_timer()
        print '%sBEGIN %s' % (' '*profileit.LEVEL, s.label)

    def __exit__(s, type, value, traceback):
        t1 = timeit.default_timer()
        print '%sEND %s - took %f s' % (' '*profileit.LEVEL, s.label, t1-s.t0)
        profileit.LEVEL -= 1

def compute_convex_mask(G, fillval):
    mask = Grid2.new_same_size(G, False)

    pts = []
    for (u,p) in G.piter():
        if p == fillval:
            pts.append(u)

    hull = convex_hull(pts)
    for (u,v) in ring_pairs(hull):
        for p in line_points(u,v):
            mask.pset(p, True)

    mask.floodfill( Int2.centroid(hull), False, True )
    return mask

if __name__ == '__main__':
    with profileit("convex hull"):
        # Example: convex hull of a 10-by-10 grid.
        gridpts = [Int2(i//10, i%10) for i in range(100)]
        assert convex_hull(gridpts) == [Int2(0, 0), Int2(9, 0), Int2(9, 9), Int2(0, 9)]

    with profileit("line_points"):
        points1 = line_points(Int2(0, 0), Int2(3, 4))
        points2 = line_points(Int2(3, 4), Int2(0, 0))
        assert(set(points1) == set(points2))
        assert points1 == [Int2(0, 0), Int2(1, 1), Int2(1, 2), Int2(2, 3), Int2(3, 4)]
        assert points2 == [Int2(3, 4), Int2(2, 3), Int2(1, 2), Int2(1, 1), Int2(0, 0)]
