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
import dero_config
import wad

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

def squidi_keylock_algo(tree, spawn_node, ideal_zone_size):
    remaining = tree
    while len(remaining.nodes()) > 1:
        # find subtree of appropriate size for this zone
        sizes = eval_subtree_sizes(remaining, spawn_node)

        best = None
        bestsize = 0
        for node in sizes:
            size = sizes[node]
            if node == spawn_node: continue
            if best == None or abs(size-ideal_zone_size) < abs(bestsize-ideal_zone_size):
                best = node
                bestsize = size

        # place lock at root of subtree
        lock = best
        remaining = copy_graph_without_subtree(remaining, spawn_node, lock)

# TODO should choose key pos furthest from previous lock
# actually, we should do key placement after we do the cycle-restoring.
# also, we don't need to place the key in this zone, we can place it in any remaining spot
        key = random.choice(remaining.nodes())

        yield (key, lock)

def needy_squidi_keylock_algo(tree, spawn_node, exit_node, ideal_zone_size):

    needed_nodes = set()

    def mark_needed(u):
        """ By definition, if a node is needed, all its ancestors are as well """
        for u in yield_ancestors(tree, u):
            needed_nodes.add(u)

    mark_needed(exit_node)
    remaining = tree
    while len(remaining.nodes()) > ideal_zone_size:
        # stop if only the spawn node is left

        # find subtree of appropriate size for this zone
        sizes = eval_subtree_sizes(remaining, spawn_node)

        eligible = [u for u in remaining.nodes() if u in needed_nodes and u != spawn_node]
        score_func = lambda u : abs(sizes[u]-ideal_zone_size)
        lock = pick_min( eligible, score_func )
        if not lock:
            break
# mark_needed(lock) # this is unnecessary

        # update and pick key location
        remaining = copy_graph_without_subtree(remaining, spawn_node, lock)

# choose a location that is furthest from a needed node
# this should approximately put keys far away from locks

        def eval_dist_to_needed(node):
            count = 0
            u = node
            while u and not u in needed_nodes:
                count += 1
                u = get_parent(remaining, u)
            return count

        score_func = lambda u : eval_dist_to_needed(u)
        key = pick_max( remaining.nodes(), score_func )
        mark_needed(key)

        yield (key, lock)

def method2(L, numRegions):
    if not numRegions:
        numRegions = L*L/100
    G = Grid2(L, L, ' ')

    # first spread the border a bit, so the level doesn't look squareish
    # avoid square shape
    # TODO should modulate with some perlin noise. with high-res, the spreading just ends up looking
    # noisey but still very circular.
    for (u,_) in G.piter_outside_radius(L/2-1):
        G.pset(u, 'b')
    print 'spreading border'
    seed_spread(['b'], 0, G, ' ', L*L/6 )

    space_vals = [str(i) for i in range(numRegions)]

    colors = {}
    for val in space_vals:
        colors[val] = 'w'

    print 'spreading space seeds'
    seed_spread(space_vals, 1, G, ' ', L*L)
    G.replace('b', ' ')

    print 'computing doors'
    door_dict = G.value_adjacency()

    # create graph rep
    adj_graph = nx.Graph()
    for (a,b) in door_dict:
        if a == ' ':
            continue
        adj_graph.add_edge(a,b)

    labels = {}
    for node in adj_graph.nodes():
        labels[node] = ''

    print 'computing space tree'
    und_space_tree = nx.minimum_spanning_tree(adj_graph)
    # tree with the spawn node as the root
    spawn_node = space_vals[0]
    space_tree = nx.dfs_tree(und_space_tree, spawn_node)
    nodepos = G.compute_centroids()

    colors[spawn_node] = 'b'
    labels[spawn_node] += 'SP'

    # choose a random leaf to be the exit
    sizes = eval_subtree_sizes(space_tree, spawn_node)
    exit_node = random.choice( [u for u in sizes if sizes[u] == 1] )
    print 'exit node = ', exit_node
    colors[exit_node] = 'g'
    labels[exit_node] += 'EX'

    def draw_labels(graph):
        for node in graph.nodes():
            pylab.annotate(labels[node], xy=add2(nodepos[node],(-2, 3)))

# DEFINITION: a lock node means, to get TO IT, requires a key.

    locks = []
    keys = []

    def on_key(k):
        keys.append(k)
        colors[k] = 'y'
        labels[k] += ' K%d' % len(keys)

    def on_gate(g):
        locks.append(g)
        colors[g] = 'r'
        labels[g] += ' G%d' % len(locks)

        for u in yield_dfs(space_tree, g, set()):
            colors[u] = 'r'
        colors[g] = 'k'

    def write_state_png():
        pylab.figure()
        nx.draw(space_tree, nodepos, node_color=[colors[v] for v in space_tree.nodes()])
        pylab.xlim([0, L])
        pylab.ylim([0, L])
        draw_labels(space_tree)
        pylab.savefig('locks%d.png' % len(locks))

    write_state_png()

    print 'start needy squidi..'
    for (key, lock) in needy_squidi_keylock_algo(space_tree, spawn_node, exit_node, numRegions/3):
        on_key(key)
        on_gate(lock)
        write_state_png()

    # only keep the doors in the tree
    # we can re-add some of these later too, if they don't break puzzle structure
    reduced_doors = {}
    for (a,b) in space_tree.edges():
        reduced_doors[(a,b)] = door_dict[asc(a,b)]

    return (G, locks, keys, space_tree, reduced_doors)

def v_case():
    T = nx.DiGraph()
    T.add_edge(1,2)
    T.add_edge(1,3)

    for (key, lock) in needy_squidi_keylock_algo(T, 1, 1):
        print key, lock

# v_case()


def test_polygonate():
    G = Grid2(3,3,0)
    G.set(1,1,1)
    polys = polygonate(G, lambda x : x == 0, False, None)
    colors = 'rgbky'
    ci = 0
    for poly in polys:
        c = colors[ci % len(colors)]
        plot_poly(poly, c+'.-')
        ci += 1
    pylab.show()

def draw_polys(polys):
    colors = 'rgbky'
    ci = 0
    for poly in polys:
        c = colors[ci % len(colors)]
        plot_poly(poly, c+'.-')
        ci += 1

def test_polygonate_2():
    G = Grid2(10,10,0)
    G.set(1,1,1)
    G.set(1,2,1)
    G.set(2,2,1)
    G.set(2,3,1)
    G.set(3,2,1)
    G.set(3,3,1)
    G.set(4,3,1)
    polys = polygonate(G, lambda x : x == 1, False, None)
    draw_polys(polys)
    pylab.xlim([-1, G.W+1])
    pylab.ylim([-1, G.H+1])
    pylab.grid(True)
    pylab.show()

def test_polygonate_perlin():
    L = 400
    G = Grid2(L, L, 0)
    S = 10.0/L

    minval = 999999.0
    maxval = -999999
    for (u,_) in G.piter():
        x = u.x * S
        y = u.y * S
        val = noise.pnoise2(x, y)
        G.pset(u, val)

    polys = polygonate(G, lambda x : x > -0.1 and x < 0.2, False, None)

    for i in range(len(polys)):
        polys[i] = linear_simplify_poly(polys[i])

    draw_polys(polys)
    marx = G.W*0.1
    mary = G.H*0.1
    pylab.xlim([-marx, G.W+marx])
    pylab.ylim([-mary, G.H+mary])
    pylab.grid(True)
    pylab.show()


def test_quat_turns():
    print Int2(1,0).turn(0)
    print Int2(1,0).turn(1)
    print Int2(1,0).turn(2)
    print Int2(1,0).turn(3)
    print Int2(1,0).turn(1).turn(1).turn(1).turn(1)

def test_left_vert():
    poly = [left_vert(Int2(0,2), edge) for edge in range(4)]
    plot_poly( poly, '.-' )
    pylab.show()

def synth_map(G, doors):
# refwad = wad.read_wad(dero_config.DOOM1_WAD_PATH)

    spaces = set([a for (a,b) in doors] + [b for (a,b) in doors])

    for space in spaces:
        def on_edge(u, v, polyid, edgeid):
            su = G.pget(u)
            sv = G.pget(v)
            door = asc(su, sv)
            if door in doors:
                (du,dv) = doors[door]
                if (du == u and dv == v) or (du == v and dv == u):
                    print 'door', str(door) + ' at edge %d' % (edgeid), du, dv

        print '--- space ' + space
        polys = polygonate(G, lambda x : x == space, True, on_edge)
        for i in range(len(polys)):
            polys[i] = linear_simplify_poly(polys[i])
        pylab.figure()
        draw_polys(polys)
        pylab.xlim([-1, G.W+1])
        pylab.ylim([-1, G.H+1])
        pylab.grid(True)
        pylab.savefig('space-%s-poly.png' % space)

# test_polygonate_2()
# test_polygonate_perlin()
if __name__ == '__main__':
    L = int(sys.argv[1])
    (G, locks, keys, space_tree, doors) = method2(L, int(sys.argv[2]))

    print 'draw grid'
    pylab.figure()
    G.show_image()
    # write adjacency positions too, ie. the doors from one space to the next
    for (a,b) in doors:
        (u,v) = doors[(a,b)]
        pylab.annotate( '%s-%s' % (a,b), xy=(u.x, L-u.y-1))
    pylab.savefig('grid.png')

    # synth playable wad
    synth_map(G, doors)

