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

def add_poly_to_map(m, poly, uniform_scale, translation):
    assert type(translation) == Vector2
    nverts = len(poly)
    first_vid = len(m.verts)
    first_lid = len(m.linedefs)
    first_sid = len(m.sidedefs)
    secid = len(m.sectors)

    for vert in poly:
        vert = Int2.floor( uniform_scale*vert + translation )
        m.verts.append( wad.Vertex().fill([vert.x, vert.y]) )

    sector = wad.Sector().fill([0, 100, '-', '-', 128, 0, 0])
    m.sectors.append( sector )

    for vertnum in range(len(poly)):
        sd = wad.SideDef().fill([0, 0, '-', '-', '-', secid])
        m.sidedefs.append(sd)

        v0 = first_vid + (vertnum % nverts)
        v1 = first_vid + ((vertnum+1) % nverts)
        assert v0 < len(m.verts)
        assert v1 < len(m.verts)
        sd_right = first_sid + vertnum
        assert sd_right == len(m.sidedefs)-1
        ld = wad.LineDef().fill([v0, v1,    0, 0, 0,    sd_right, -1]).set_flag('Impassible')
        lineid = vertnum
        m.linedefs.append(ld)

    return sector

def synth_map(G, doors, mapname, scale, refwad):
    refsec = random.choice([s for s in refwad.maps[0].sectors if s.floor_pic and s.ceil_pic])

    floortexs = set([sec.floor_pic for m in refwad.maps for sec in m.sectors if sec.floor_pic])
    ceiltexs = set([sec.ceil_pic for m in refwad.maps for sec in m.sectors if sec.ceil_pic])
    walltexs = set([sd.midtex for m in refwad.maps for sd in m.sidedefs if sd.midtex])

    print '----------------------------------------'
    print floortexs
    print ceiltexs

    spaces = set([a for (a,b) in doors] + [b for (a,b) in doors])
    rv = wad.Map(mapname)

    for space in spaces:
        print '--- space ' + space

        door_edges = set()

        def on_edge(u, v, polyid, edgeid):
            assert polyid == 0
            door = asc(G.pget(u), G.pget(v))
            if door in doors:
                if unordered_equal(doors[door], (u,v)):
                    print 'door', str(door) + ' at edge %d' % (edgeid), u, v
                    door_edges.add(edgeid)

        polys = polygonate(G, lambda x : x == space, True, on_edge)
        assert len(polys) == 1
        poly = polys[0]
# poly = linear_simplify_poly(poly)
        """
        pylab.figure()
        draw_polys([poly])
        pylab.xlim([-1, G.W+1])
        pylab.ylim([-1, G.H+1])
        pylab.grid(True)
        pylab.savefig('space-%s-poly.png' % space)
        """

        lineid_base = len(rv.linedefs)
        sdid_base = len(rv.sidedefs)
        sector = add_poly_to_map( rv, poly, scale, Vector2(0,0) )
        sector.floor_pic = random.sample(floortexs,1)[0]
        sector.ceil_pic = random.sample(ceiltexs,1)[0]

        walltex = random.sample(walltexs,1)[0]
        for sdid in range(sdid_base, len(rv.sidedefs)):
            rv.sidedefs[sdid].midtex = walltex

        # make doors where they should be
        doortex = random.choice([tex for tex in walltexs if 'DOOR' in tex])
        print 'door tex = %s' % doortex
        for did in door_edges:
            ld = rv.linedefs[lineid_base + did]
            ld.clear_flag('Impassible')
            sd = rv.sidedefs[ld.sd_right]
            sd.midtex = doortex
            v0 = rv.verts[ld.vert0]
            v1 = rv.verts[ld.vert1]

            print 'made door at %d,%d -> %d,%d' % (v0.x*1.0/scale, v0.y*1.0/scale,
                    v1.x*1.0/scale, v1.y*1.0/scale)

        assert len(rv.linedefs) == lineid_base + len(poly)
        assert len(rv.sidedefs) == sdid_base + len(poly)

    return rv

def grid2map(G, refwad, scale):
    """ Converts a flat grid to a valid WAD with two-sided lines between all spaces """
    val2sector = {}

    # Grids holding other primitive info
    verts = Grid2(G.W+2, G.H+2, None)
    leftlines = Grid2(G.W+2, G.H+2, None)
    botlines = Grid2(G.W+2, G.H+2, None)
    rv = wad.Map('')
    offset = Int2(1,1)

    def get_line_gridcell( u, edge ):
        if edge == 0:
            return (leftlines, u+offset+Int2(1,0))
        elif edge == 1:
            return (botlines, u+offset+Int2(0,1))
        elif edge == 2:
            return (leftlines, u+offset)
        else:
            return (botlines, u+offset)

    def get_linedef(u,edge):
        (grid, ut) = get_line_gridcell(u, edge)
        return grid.pget(ut)

    def set_linedef(u, edge, ld):
        lid = len(rv.linedefs)
        rv.linedefs.append(ld)
        (grid, ut) = get_line_gridcell(u, edge)
        grid.pset( ut, lid )

    def new_right_vert(u, edge):
        c = Int2.floor(right_vert(u, edge) * scale)
        print 'created right vert ' + str(c)
        return wad.Vertex().fill([c.x, c.y])

    def get_or_set_right_vert(u, edge):
        if edge == 0:
            ut = u + offset + Int2(1,0)
        elif edge == 1:
            ut = u + offset + Int2(1,1)
        elif edge == 2:
            ut = u + offset + Int2(0,1)
        else:
            ut = u + offset + Int2(0,0)

        vid = verts.pget(ut)
        if vid == None:
            vert = new_right_vert(u, edge)
            vid = len(rv.verts)
            rv.verts.append(vert)
            verts.pset( ut, vid )
            
        return vid

    def get_or_set_left_vert(u, edge): return get_or_set_right_vert(u, (edge+1)%4)

    for (u,p) in G.piter():
        sector = None
        if p in val2sector:
            sector = val2sector[p]
        else:
            sector = wad.Sector().fill([0, 128,      '-', '-',       128, 0, 0])
            val2sector[p] = sector
            sid = len(rv.sectors)
            rv.sectors.append(sector)

        for edge in range(4):
            print u, edge
            v = u + EDGE_TO_NORM[edge]
            if not G.check(v):
                continue
            q = G.pget(v)
            if p != q:
                # check if linedef here already
                lid = get_linedef(u, edge)
                if lid == None:
                    # verts there?
                    vid_left = get_or_set_left_vert(u, edge)
                    vid_right = get_or_set_right_vert(u, edge)
                    ld = wad.LineDef().fill([vid_left, vid_right, 0, 0, 0,   -1, -1])
                    lid = set_linedef( u, edge, ld)
                else:
                    print 'reuse linedef %d' % lid
                    ld = rv.linedefs[lid]

    return rv

# test_polygonate_2()
# test_polygonate_perlin()

def test_grid2map():
    refwad = wad.load(dero_config.DOOM1_WAD_PATH)
    G = Grid2(3, 3, 0)
    G.set(1, 1, 1)
    scale = 100.0
    m = grid2map(G, refwad, scale)
    print '%d verts, %d linedefs, %d sidedefs, %d sectors' % (len(m.verts), len(m.linedefs), len(m.sidedefs), len(m.sectors))
    for v in m.verts:
        v.x += int(0.2*scale*(random.random()*2-1))
        v.y += int(0.2*scale*(random.random()*2-1))
    wad.save_map_png(m, 'grid2map-square-test.png')

if __name__ == '__main__':

    test_grid2map()

    L = int(sys.argv[1])
    (G, locks, keys, space_tree, doors, spawn_node) = method2(L, int(sys.argv[2]))

    print 'draw grid'
    pylab.figure()
    G.show_image()
    # write adjacency positions too, ie. the doors from one space to the next
    for (a,b) in doors:
        (u,v) = doors[(a,b)]
        pylab.annotate( '%s-%s' % (a,b), xy=(u.x, L-u.y-1))
    pylab.savefig('grid.png')

    # synth playable wad
    refwad = wad.load(dero_config.DOOM1_WAD_PATH)
    scale = 4096/L

# m = synth_map(G, doors, 'E1M1', scale, refwad)
    m = grid2map(G, refwad, scale)
    m.name = 'E1M1'

# draw it
    print '%d linedefs' % len(m.linedefs)
# jitter all verts a bit, to reveal dupes
    for v in m.verts:
        v.x += int(0.2*scale*(random.random()*2-1))
        v.y += int(0.2*scale*(random.random()*2-1))
    wad.save_map_png(m, 'grid2map-test.png')

    # add player start
    startpos = random.choice([c for c in G.cells_with_value(spawn_node)])
    m.add_player_start( int((startpos.x+0.5)*scale), int((startpos.y+0.5)*scale), 0 )
    lumps = []
    m.append_lumps_to(lumps)
    wad.save('source.wad', 'PWAD', lumps)
# wad.save_map_png( m, 'expected.png' )
    dero_config.build_wad( 'source.wad', 'built-playable.wad' )

    # readback
    actwad = wad.load('source.wad')
# wad.save_map_png( actwad.maps[0], 'actual.png')


