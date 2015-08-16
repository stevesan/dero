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
import noise
import wad
import minewad

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

def remove_smallest_regions(G, fill_val, values, remove_ratio):
    valSizes = []
    for val in values:
        valSize = (val, len([c for c in G.cells_with_value(val)]))
        valSizes += [valSize]

    valSizes.sort(key = lambda vs : vs[1])

    numRemove = int(len(valSizes) * remove_ratio)

    for i in range(numRemove):
        val = valSizes[i][0]
        G.replace( val, fill_val )
        values.remove(val)

    return values

def save_grid_png(G, path):
    pylab.figure()
    G.show_image()
    pylab.savefig(path)
    pylab.close()

def setup_doors_presynth(geo_grid, doors):
    for ((a,b), (u,v)) in doors.iteritems():
        direction = v - u
        this_door_cells = []
        this_door_cells.append(u*3 + Int2(1,1) + direction)
        this_door_cells.append(v*3 + Int2(1,1) - direction)

        floorhtsum = 0
        zone = 'door(%s,%s)' % (a,b)
        firstzone = None # use the same zone for both 'sides' of the door
        for w in this_door_cells:
            c = geo_grid.pget(w)
            floorhtsum += c.floorht

        floorht = int(floorhtsum*1.0/len(this_door_cells))
        for w in this_door_cells:
            c = geo_grid.pget(w)
            c.zone = zone
            c.floorht = floorht
            c.ceilht = floorht + 128

def setup_doors(mapp, doors):
    # script up doors
    door_secids = set()
    for cell in builder.val2sectorid:
        if 'door' in cell.zone:
            door_secids.add( builder.val2sectorid[cell] )

    doortexs = []
    with open('midtexs.txt') as f:
        for line in f:
            if 'DOOR' in line:
                doortexs.append(line.strip())
    assert len(doortexs) > 10

    print door_secids
    secid2doortex = {}
    for secid in door_secids:
        sec = mapp.sectors[secid]
        sec.ceil_height = sec.floor_height
        secid2doortex[secid] = random.choice(doortexs)

    for ld in mapp.linedefs:
        def proc_sd(sdid, other_sdid):
            if sdid < 0 or sdid >= len(mapp.sidedefs):
                return
            sd = mapp.sidedefs[sdid]
            if sd.sector not in door_secids:
                return
            ld.set_flag('Lower Unpegged')
            tex = secid2doortex[sd.sector]
            if ld.get_flag('Two-sided'):
                ld.function = 31
                assert other_sdid > 0 and other_sdid < len(mapp.sidedefs)
                othersd = mapp.sidedefs[other_sdid]
                othersd.midtex = '-'
                othersd.uppertex = tex
                othersd.lowertex = tex
            else:
                sd.midtex = tex
                sd.uppertex = tex
                sd.lowertex = tex
        proc_sd(ld.sd_left, ld.sd_right)
        proc_sd(ld.sd_right, ld.sd_left)

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

    save_grid_png(G, 'grid-pre-remove-spaces.png')

    space_vals = remove_smallest_regions(G, ' ', space_vals, 0.5)

    save_grid_png(G, 'grid-post-remove-spaces.png')

    # save grid of spaces only
    pylab.figure()
    G.show_image()
    pylab.savefig('grid-spaces.png')
    pylab.close()

    print 'computing space adjacency'
    edge2cells = G.value_adjacency()

    # create graph rep
    adj_graph = nx.Graph()
    for (a,b) in edge2cells:
        if a == ' ':
            continue
        adj_graph.add_edge(a,b)

    labels = {}
    for node in adj_graph.nodes():
        labels[node] = ''

    print 'computing space tree'
    und_space_tree = nx.minimum_spanning_tree(adj_graph)
    # tree with the spawn node as the root
    spawn_space = space_vals[0]
    space_tree = nx.dfs_tree(und_space_tree, spawn_space)
    nodepos = G.compute_centroids()

    colors[spawn_space] = 'b'
    labels[spawn_space] += 'SP'

    # choose a random leaf to be the exit
    sizes = eval_subtree_sizes(space_tree, spawn_space)
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
    space2zone = {}

    def on_key_node(k):
        keys.append(k)
        colors[k] = 'y'
        labels[k] += ' K%d' % len(keys)

    def on_lock_node(g):
        zoneid = len(locks)
        locks.append(g)
        colors[g] = 'r'
        labels[g] += ' G%d' % len(locks)

        for u in yield_dfs(space_tree, g, set()):
            colors[u] = 'r'

            if u not in space2zone:
                space2zone[u] = zoneid
        colors[g] = 'k'

    def write_state_png():
        pylab.figure()
        nx.draw(space_tree, nodepos, node_color=[colors[v] for v in space_tree.nodes()])
        pylab.xlim([0, L])
        pylab.ylim([0, L])
        draw_labels(space_tree)
        pylab.savefig('locks%d.png' % len(locks))
        pylab.close()

    write_state_png()

    print 'start needy squidi..'
    for (key, lock) in needy_squidi_keylock_algo(space_tree, spawn_space, exit_node, len(space_vals)/4):
        on_key_node(key)
        on_lock_node(lock)
        write_state_png()

    # mark un-marked nodes as the last zone
    for u in space_tree.nodes():
        if u not in space2zone:
            space2zone[u] = len(locks)
    numzones = len(locks) + 1

    # draw the tree, labeling each node by its zone
    pylab.figure()
    nx.draw(space_tree, nodepos)
    pylab.xlim([0, L])
    pylab.ylim([0, L])
    for node in space2zone:
        zone = space2zone[node]
        pylab.annotate( str(zone), xy=add2(nodepos[node], (-2, 3)) )
    pylab.savefig('zoned-space-tree.png' )
    pylab.close()

    # we can re-add some of these later too, if they don't break puzzle structure

    # make grid only reflective of the zones now
    if False:
        for val in space2zone:
            G.replace(val, space2zone[val])

        zone_adj2edge = {}
        for (a,b) in space_tree.edges():
            zoneA = space2zone[a]
            zoneB = space2zone[b]
            if zoneA != zoneB:
                zone_adj2edge[asc(zoneA,zoneB)] = edge2cells[asc(a,b)]

        return (G, [space2zone[lock] for lock in locks], [space2zone[key] for key in keys], zone_adj2edge, space2zone[spawn_space])
    else:
        # only include edges actually in the tree
        tree_adjs = {edge:edge2cells[asc(edge[0], edge[1])] for edge in space_tree.edges()}
        return (G, locks, keys, tree_adjs, spawn_space)

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
                sector = wad.Sector().fill([0, 100,      '-', '-',       160, 0, 0])
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

    def relax_verts(s):
        vert2nborIds = {}
        for v in s.mapp.verts:
            vert2nborIds[v] = []

        for ld in s.mapp.linedefs:
            v0 = s.mapp.verts[ld.vert0]
            v1 = s.mapp.verts[ld.vert1]
            vert2nborIds[v0] += [ld.vert1]
            vert2nborIds[v1] += [ld.vert0]

        for v in s.mapp.verts:
            nborIds = vert2nborIds[v]
            assert len(nborIds) >= 2

            avgx = pylab.mean([s.mapp.verts[nid].x for nid in nborIds] + [v.x])
            avgy = pylab.mean([s.mapp.verts[nid].y for nid in nborIds] + [v.y])
            v.x = avgx
            v.y = avgy

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

def get_wrap(listt, idx):
    return listt[ idx % len(listt) ]

def assign_textures(mapp, builder):

    zones = set([val.zone for val in builder.val2sectorid])

    sec2zone = {}
    for (cell, secid) in builder.val2sectorid.iteritems():
        if secid in sec2zone:
            assert sec2zone[secid] == cell.zone
        else:
            sec2zone[secid] = cell.zone

    # choose a texture set for each zone
    sets = minewad.read_texsets('texsets.txt')
    zonetexs = { zone : random.choice(sets) for zone in zones }

    for (sid, sec) in id_iter(mapp.sectors):
        zone = sec2zone[sid]
        ts = zonetexs[zone]
        sec.floor_pic = ts.floor
        sec.ceil_pic = ts.ceil

    for sd in mapp.sidedefs:
        sid = sd.sector
        sec = mapp.sectors[sid]
        zone = sec2zone[sid]
        ts = zonetexs[zone]
        sd.midtex = get_wrap(ts.sidetexs, 0)
        sd.uppertex = get_wrap(ts.sidetexs, 1)
        sd.lowertex = get_wrap(ts.sidetexs, 2)

    for ld in mapp.linedefs:
        if ld.sd_right != -1 and ld.sd_left != -1:
            ld.set_flag('Two-sided')
            mapp.sidedefs[ld.sd_right].midtex = '-'
            mapp.sidedefs[ld.sd_left].midtex = '-'

class SynthCell:
    def __init__(s):
        s.zone = None
        s.floorht = int(0)
        s.ceilht = int(100)

    def __str__(s):
        return 'SynthCell, zone=%s, [%d,%d]' % (str(s.zone), s.floorht, s.ceilht)

    def __eq__(s, t):
        return s.zone == t.zone \
            and s.floorht == t.floorht \
            and s.ceilht == t.ceilht

    def __ne__(s, t):
        return not s.__eq__(t)

    def __hash__(s):
        return hash((s.zone, s.floorht, s.ceilht))

    @staticmethod
    def test():
        d1 = SynthCell()
        d2 = SynthCell()
        d3 = SynthCell()

        d1.zone = 'a'
        d2.zone = 'a'
        d3.zone = ' '

        assert d1 == d1
        assert d1 == d2
        assert d2 == d1
        assert d3 != d1
        assert d3 != d2


if __name__ == '__main__':

    SynthCell.test()

    test_grid2map()

    L = int(sys.argv[1])
    (zone_grid, locks, keys, doors, spawn_zone) = method2(L, int(sys.argv[2]))

    """
    print 'draw zone grid'
    pylab.figure()
    zone_grid.show_image()
    # write adjacency positions too, ie. the doors from one zone to the next
    for ((a,b), (u,v)) in doors.iteritems():
        pylab.annotate( '%s-%s' % (a,b), xy=(u.x, L-u.y-1))
    pylab.savefig('grid-zones.png')
    pylab.close()
        """

    fine_grid = zone_grid.integer_supersample(3)

    # separate zones with filler, but make sure to leave door cells alone.
    door_cells = set()
    """
    for ((a,b), (u,v)) in doors.iteritems():
        door_cells.add(u)
        door_cells.add(v)
        """
    fine_grid = fine_grid.separate(' ', lambda u : u/3 in door_cells)

    pylab.figure()
    fine_grid.show_image()
    pylab.savefig('grid-separated-zones.png')
    pylab.close()

    # create enhanced grid
    geo_grid = Grid2(fine_grid.W, fine_grid.H, None)
    for (u, p) in fine_grid.piter():
        data = SynthCell()
        data.zone = p
        geo_grid.pset(u, data)

# perlin noise heights
        if True:
            noiseval = noise.pnoise2( u.x*5.0/L, u.y*5.0/L )
            ht = int((noiseval*0.5 + 0.5) * 5) * 16
        else:
            ht = 0
        data.floorht = ht
        data.ceilht = ht + 256

    # mark door cells as separate sectors
    setup_doors_presynth(geo_grid, doors)

    spawnAreaCells = [c for c in fine_grid.cells_with_value(spawn_zone)]

    # poke some random holes in the spawn area
    """
    for _ in range(25):
        geo_grid.pget( random.choice(spawnAreaCells) ).zone = ' '
        """

    # raise start pos a bit
    startcell = random.choice(spawnAreaCells)
    startdata = geo_grid.pget(startcell)
    startdata.floorht = 40

    mapp = wad.Map('E1M1')
    builder = MapGeoBuilder(mapp)
# scale = 4096/
    scale = 96
    builder.synth_grid(geo_grid, scale, lambda data : data.zone == ' ')
    builder.relax_verts()
    assign_textures(mapp, builder)

    # transfer floor/ceil hts
    for data in builder.val2sectorid:
        sid = builder.val2sectorid[data]
        sector = mapp.sectors[sid]
        sector.floor_height = data.floorht
        sector.ceil_height = data.ceilht

    # add player start
    mapp.add_player_start( int((startcell.x+0.5)*scale), int((startcell.y+0.5)*scale), 0 )

    setup_doors(mapp, doors)

# draw it
    print '%d linedefs' % len(mapp.linedefs)
    """
# jitter all verts a bit, to reveal dupes
    for v in mapp.verts:
        v.x += int(0.2*scale*(random.random()*2-1))
        v.y += int(0.2*scale*(random.random()*2-1))
    """

# wad.save_map_png(mapp, 'grid2mapgeo-test.png')

    mapp.sanity_asserts()

    lumps = []
    mapp.append_lumps_to(lumps)
    wad.save('source.wad', 'PWAD', lumps)
    dero_config.build_wad( 'source.wad', 'built-playable.wad' )

    # readback
    actwad = wad.load('source.wad')
# wad.save_map_png( actwad.maps[0], 'actual.png')


