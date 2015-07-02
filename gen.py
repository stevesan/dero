import random
import numpy
import networkx as nx
import pylab
import math
import scipy
from scipy.cluster.vq import kmeans
import sys
from planar import Vec2
from utils import *


def testBasic():
    g = Grid2(80,80,' ')

    for (x,y) in g.iter():
        if random.random() < 0.2:
            g.set(x,y,'X')

    g.write()


"""
curve = InterestCurve()
ff = range(200)
ii = range(200)
for i in range(len(ff)):
    ff[i] = i*1.0/(len(ff)-1)
    ii[i] = curve.eval(ff[i])
    """

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

def tunnel_test():
    g = Grid2(40,40,' ')
    tunnel(g, Int2(0, 0), 'X', 0, 20)
    g.write()

def cluster_verts_test():
    G = nx.Graph()
    NV = 35
    numGroups = 4
    groupSize = NV/numGroups

    nodePos = slow_poisson_sampler(0.1, NV)

    for u in range(len(nodePos)):
        G.add_node(u)

    """
    for i in range(NV):
        G.add_node(i)
        nodePosDict[i] = Vec2(random.random(), random.random())
        """

# compute distances

    nodeGroup = [-1 for u in range(NV)]

    """
    for u in range(NV):
        if nodeGroup[u] != -1: continue
        nbors = range(NV)
        nbors.sort(key=lambda v: D.get(u,v) if nodeGroup[v] == -1 else sys.float_info.max)
        for i in range(0,groupSize+1):
            v = nbors[i]
            if nodeGroup[v] != -1: continue
            nodeGroup[v] = u
            print u,v
            G.add_edge(u,v)
    """

    tupleArray = [(p.x, p.y) for p in nodePos]
    posMatrix = numpy.matrix(tupleArray)
    print posMatrix
# TODO call whiten here??
    (centroids, distortion) = kmeans(posMatrix, numGroups)

    centersVecArray = [Vec2(row[0], row[1]) for row in centroids]

    nodeGroup = assign_nearest_center(nodePos, centersVecArray)

    nx.draw(G, nodePos, node_color=nodeGroup, cmap=pylab.get_cmap('jet'))
    pylab.show()

def spread_test_2():
    L = 30
    G = Grid2(L,L, ' ')
    G.set_border('b')
    seed_spread(['b'], 0, G, ' ', L*4*1)

# spawn initial region
    seed_spread(['0'], 1, G, ' ', L*L/6)

    groups = ['0', '1', '2', '3']
    doors = []
    keys = []

    for ig in range(1, len(groups)):
# place the key in the previous group
        prevgroup = groups[ig-1]
        keypos = pick_random( G.cells_with_values(set([prevgroup])) )
        keys += [keypos]
        # pick the first door to enter this group
        group = groups[ig]
        prevgroups = groups[0:ig]

        (regions, regionsizes) = G.connected_components_grid(' ')
        front = G.free_cells_adjacent_to(' ', set(prevgroups))

        regions.write()

# bias by region count
        bag = []
        for u in front:
            region = regions.pget(u)
            size = regionsizes[region]
            for i in range(size):
                bag += [u]

        door = pick_random(bag)

# TODO bias towards larger components

# G.write()
# DEBUG
        T = G.duplicate()
        for p in G.free_cells_adjacent_to(' ', set(prevgroups)):
            T.pset(p, '=')
        T.write()

        doors += [door]
        # seed and spread
        G.pset(door, group)
        seed_spread([group], 0, G, ' ', L*L/6)

    # pick exit point
    exit = pick_random([x for x in G.cells_with_value(groups[-1])])
    G.pset(exit, 'X')

    G.write()
    print '----'
    print G.tally()

    for door in doors:
        G.pset(door, '=')
    for key in keys:
        G.pset(key, 'K')

    G.write()

    image = numpy.ndarray(L,L)

def spread_test(L, numRegions):
    if not numRegions:
        numRegions = L*L/100
    G = Grid2(L, L, ' ')

    # first spread the border a bit, so the level doesn't look squareish
    # avoid square shape
    for (u,_) in G.piter_outside_radius(L/2-1):
        G.pset(u, 'b')
    seed_spread(['b'], 0, G, ' ', L*L/6 )

    seedvals = [str(i) for i in range(numRegions)]
    colors = {}
    for val in seedvals:
        colors[val] = 'w'

    seed_spread(seedvals, 1, G, ' ', L*L)
    G.replace('b', ' ')


    adj = G.value_adjacency()

    # create graph rep
    C = nx.Graph()
    regions = set()
    # print 'region adjacencies'
    for (a,b) in adj:
        if a == ' ':
            continue
        C.add_edge(a,b)
        regions.add(a)
        regions.add(b)
        # print (a,b), '-->', adj[(a,b)]

    labels = {}
    for node in C.nodes():
        labels[node] = str(node)

    MSTundir = nx.minimum_spanning_tree(C)
    spawn_node = seedvals[0]
    MST = nx.dfs_tree(MSTundir, spawn_node)
    nodepos = G.compute_centroids()

    # pick a leaf for exit

    colors[spawn_node] = 'b'
    labels[spawn_node] += ' START'

    def draw_labels(graph):
        for node in graph.nodes():
            pylab.annotate(labels[node], xy=add2(nodepos[node],(-2, 3)))

# DEFINITION: a gate node means, to get TO IT, requires a key.

    gates = []
    keys = []

    remaining = MST.copy()

    def on_key(k):
        keys.append(k)
        colors[k] = 'y'
        labels[k] += ' K%d' % len(keys)

    def on_gate(g):
        gates.append(g)
        colors[g] = 'r'
        labels[g] += ' G%d' % len(gates)

        for u in yield_dfs(MST, g, set()):
            colors[u] = 'r'

    def write_state_png():
        pylab.figure()
        nx.draw(MST, nodepos, node_color=[colors[v] for v in MST.nodes()])
        pylab.xlim([0, L])
        pylab.ylim([0, L])
        draw_labels(MST)
        pylab.savefig('gates%d.png' % len(gates))

    idealsize = numRegions/3

    write_state_png()

    # KEY/LOCK ALGO
    while True:

        # find subtree of appropriate size for this zone
        sizes = eval_subtree_sizes(remaining, spawn_node)

        best = None
        bestsize = 0
        for node in sizes:
            size = sizes[node]
            if best == None or abs(size-idealsize) < abs(bestsize-idealsize):
                best = node
                bestsize = size

        # place gate at root of subtree
        gate = best
        remaining = copy_graph_without_subtree(remaining, spawn_node, gate)

        if len(remaining.nodes()) <= 1:
            break

        key = random.choice(remaining.nodes())

# TODO should choose key pos furthest from previous gate
# actually, we should do key placement after we do the cycle-restoring.
# also, we don't need to place the key in this zone, we can place it in any remaining spot
        on_key(key)
        on_gate(gate)
        write_state_png()

        # stop if we only have 1 or 0 nodes left (probably just the spawn node)
# if len(remaining.nodes()) <= 1:
# break

    pylab.figure()
    G.show_image()
    pylab.savefig('grid.png')

spread_test(int(sys.argv[1]), int(sys.argv[2]))
