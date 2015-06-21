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
    g = Grid(80,80,' ')

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
    g = Grid(40,40,' ')
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

def spread_test():
    L = 20
    G = Grid(L, L, ' ')

    # first spread the border a bit, so the level doesn't look squareish
    border = '.'
    for y in range(L):
        G.set(0, y, border)
        G.set(L-1, y, border)
    for x in range(L):
        G.set(x, 0, border)
        G.set(x, L-1, border)

    seed_spread([border], 0, G, ' ', L*4*2 )

    seedvals = ['+','-','=']
    seed_spread(seedvals, 4, G, ' ', L*L)
    G.write()

    adj = value_adjacency(G)
    for (a,b) in adj:
        print (a,b), '-->', adj[(a,b)]

    labels = {}
    for (p,x) in G.piter():
        if x not in labels:
            labels[x] = str(x)

    # create graph rep
    H = nx.Graph()
    for (a,b) in adj:
        H.add_edge(a,b)

    pos = nx.spring_layout(H)
    nx.draw(H, pos)
    nx.draw_networkx_labels(H, pos, labels, font_size=16)
    pylab.show()


def spread_test_2():
    L = 20
    G = Grid(L,L, ' ')
    G.set_border('b')
    seed_spread(['b'], 0, G, ' ', L*4*1)

# spawn initial region
    seed_spread(['.'], 1, G, ' ', L*L/5)

    groups = ['.', '1', '2', '3']

    for ig in range(1, len(groups)):
        # pick the first door to enter this group
        group = groups[ig]
        prevgroups = groups[0:ig]
        door = pick_random( G.cells_adjacent_to(' ', set(prevgroups)) )[0]
        # seed and spread
        G.pset(door, group)
        seed_spread([group], 0, G, ' ', L*L/5)

    G.write()


spread_test_2()
