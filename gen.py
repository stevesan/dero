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
import noise

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
            while u and u not in needed_nodes:
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

    return (G, locks, keys, space_tree, reduced_doors, spawn_node)

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

class MapGeoBuilder:
    """ Converts a flat grid to a valid WAD with two-sided lines between all spaces """

    def __init__(s, mapp):
        s.mapp = mapp

    def reset(s, G):
        s.val2sectorid = {}
        s.vid2uses = {}
        s.vertids = GridVerts2(G.W, G.H, None)
        s.lineids = GridEdges2(G.W, G.H, None)

    def get_linedef_id(s, u, v):
        return s.lineids.get_between(u, v)

    def synth_grid(s, G, scale, is_unreachable):
        s.reset(G)

        mapp = s.mapp
        val2sectorid = s.val2sectorid
        vid2uses = s.vid2uses

        def add_linedef(u, edge, ld):
            lid = len(mapp.linedefs)
            mapp.linedefs.append(ld)
            s.lineids.set(u, edge, lid)

        def new_right_vert(u, edge):
            c = Int2.floor(right_vert(u, edge) * scale)
            return wad.Vertex().fill([c.x, c.y])

        def get_or_set_right_vert(u, edge):
            vid = s.vertids.get_right(u, edge)
            if vid == None:
                vert = new_right_vert(u, edge)
                vid = len(mapp.verts)
                mapp.verts.append(vert)
                s.vertids.set_right( u, edge, vid )
            return vid

        def get_or_set_left_vert(u, edge): return get_or_set_right_vert(u, (edge+1)%4)

        for (u,p) in G.piter():
            if is_unreachable(p):
                continue

            if p in val2sectorid:
                sid = val2sectorid[p]
            else:
                print 'creating sector for grid value %s' % p
                sid = len(mapp.sectors)
                sector = wad.Sector().fill([0, 100,      '-', '-',       128, 0, 0])
                mapp.sectors.append(sector)
                val2sectorid[p] = sid

            for edge in range(4):
                v = u + EDGE_TO_NORM[edge]
                if not G.check(v):
                    continue
                q = G.pget(v)
                if p != q:
                    # check if linedef here already
                    lid = s.lineids.get(u, edge)
                    if lid == None:
                        vid_left = get_or_set_left_vert(u, edge)
                        vid_right = get_or_set_right_vert(u, edge)
                        ld = wad.LineDef().fill([vid_left, vid_right, 0, 0, 0,   -1, -1])
                        lid = add_linedef( u, edge, ld)

                        if vid_left not in vid2uses: vid2uses[vid_left] = 0
                        vid2uses[vid_left] += 1
                        if vid_right not in vid2uses: vid2uses[vid_right] = 0
                        vid2uses[vid_right] += 1
                    else:
                        ld = mapp.linedefs[lid]

                    # create our side def
                    sd = wad.SideDef().fill([0, 0,      '-', '-', '-', sid])
                    sdid = len(mapp.sidedefs)
                    mapp.sidedefs.append(sd)

                    if get_or_set_right_vert(u,edge) == ld.vert1:
                        assert ld.sd_right == -1
                        ld.sd_right = sdid
                    else:
                        assert ld.sd_left == -1
                        ld.sd_left = sdid

    def make_border(s, u, v):
        """ Borders are just linedefs that demarcate height and/or texture, as oppposed to being walls """
        ld = s.mapp.linedefs[ s.get_linedef_id(u, v) ]
        ld.clear_flag('Impassible').set_flag('Two-sided')
        sd_right = s.mapp.sidedefs[ld.sd_right]
        sd_left = s.mapp.sidedefs[ld.sd_left]
        sd_right.midtex = '-'
        sd_left.midtex = '-'

# test_polygonate_2()
# test_polygonate_perlin()

def test_grid2map():
    G = Grid2(3, 3, 0)
    G.set(1, 1, 1)
    scale = 100.0
    m = wad.Map('E1M1')
    builder = MapGeoBuilder(m)
    builder.synth_grid(G, scale, lambda x : x == 0)
    assert len(m.verts) == 4
    assert len(m.linedefs) == 4
    assert len(m.sidedefs) == 4
    assert len(m.sectors) == 1

    # make sure there are no dupe verts
    m.sanity_asserts()
        
    for v in m.verts:
        v.x += int(0.2*scale*(random.random()*2-1))
        v.y += int(0.2*scale*(random.random()*2-1))
    wad.save_map_png(m, 'mapgeobuilder-square-test.png')

def read_texnames(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f]

def assign_debug_textures(mapp):

    for (sid, sec) in id_iter(mapp.sectors):
        sec.floor_pic = 'FLOOR0_1'
        sec.ceil_pic = 'CEIL1_1'

    for sd in mapp.sidedefs:
        sd.midtex = 'METAL'
        sd.uppertex = 'PIPE2'
        sd.lowertex = 'BROWN1'

    for ld in mapp.linedefs:
        if ld.sd_right != -1 and ld.sd_left != -1:
            ld.set_flag('Two-sided')
            mapp.sidedefs[ld.sd_right].midtex = '-'
            mapp.sidedefs[ld.sd_left].midtex = '-'

class CellData:
    def __init__(s):
        s.space = None
        s.floorht = int(0)
        s.ceilht = int(100)

    def __str__(s):
        return 'CellData, space=%s, [%d,%d]' % (str(s.space), s.floorht, s.ceilht)

    def __eq__(s, other):
        return s.space == other.space and s.floorht == other.floorht and s.ceilht == other.ceilht

    def __ne__(s, other):
        return not s.__eq__(other)

    def __hash__(s):
        return hash((s.space, s.floorht, s.ceilht))

    @staticmethod
    def test():
        d1 = CellData()
        d2 = CellData()
        d3 = CellData()

        d1.space = 'a'
        d2.space = 'a'
        d3.space = ' '

        assert d1 == d1
        assert d1 == d2
        assert d2 == d1
        assert d3 != d1
        assert d3 != d2


if __name__ == '__main__':

    CellData.test()

    test_grid2map()

    L = int(sys.argv[1])
    (G, locks, keys, space_tree, doors, spawn_node) = method2(L, int(sys.argv[2]))

    print 'draw space grid'
    pylab.figure()
    G.show_image()
    # write adjacency positions too, ie. the doors from one space to the next
    for (a,b) in doors:
        (u,v) = doors[(a,b)]
        pylab.annotate( '%s-%s' % (a,b), xy=(u.x, L-u.y-1))
    pylab.savefig('grid.png')

    # create enhanced grid
    G2 = Grid2(G.W, G.H, None)
    for (u, p) in G.piter():
        data = CellData()
        data.space = p
        G2.pset(u, data)

    # synth playable wad
    scale = 6000/L

    spawnAreaCells = [c for c in G.cells_with_value(spawn_node)]

    # poke some random holes in the spawn area
    for _ in range(25):
        G2.pget( random.choice(spawnAreaCells) ).space = ' '

    startcell = random.choice(spawnAreaCells)
    # raise start pos a bit
    startdata = G2.pget(startcell)
    startdata.floorht = 40

    mapp = wad.Map('E1M1')
    builder = MapGeoBuilder(mapp)
    builder.synth_grid(G2, scale, lambda data : data.space == ' ')
    assign_debug_textures(mapp)

    # transfer floor/ceil hts
    for data in builder.val2sectorid:
        sid = builder.val2sectorid[data]
        sector = mapp.sectors[sid]
        sector.floor_height = data.floorht
        sector.ceil_height = data.ceilht

    # add player start
    mapp.add_player_start( int((startcell.x+0.5)*scale), int((startcell.y+0.5)*scale), 0 )

# draw it
    print '%d linedefs' % len(mapp.linedefs)
# jitter all verts a bit, to reveal dupes
    for v in mapp.verts:
        v.x += int(0.2*scale*(random.random()*2-1))
        v.y += int(0.2*scale*(random.random()*2-1))
    wad.save_map_png(mapp, 'grid2mapgeo-test.png')

    mapp.sanity_asserts()

    lumps = []
    mapp.append_lumps_to(lumps)
    wad.save('source.wad', 'PWAD', lumps)
    dero_config.build_wad( 'source.wad', 'built-playable.wad' )

    # readback
    actwad = wad.load('source.wad')
# wad.save_map_png( actwad.maps[0], 'actual.png')


