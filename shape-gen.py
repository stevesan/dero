from utils import *

FREE = '.'
FILL = 'X'

def compute_symmetry( G, axis ):
    center = Int2(G.W/2, G.H/2)
    H = Int2(G.W/2, G.H/2)
    perp = axis.turn(1)
    def reflect(u):
        perpcoord = (u-center).dot(perp)
        return u - perp*(2*perpcoord)

    singles = 0
    fills = 0
    for (u, p) in G.piter():
        if p != FILL:
            continue
        fills += 1
        v = reflect(u)
        if not G.check(v) or G.pget(v) != FILL:
            singles += 1

    return 1.0 - singles * 1.0 / fills

def compute_convexity( G ):
    H = compute_convex_mask(G, FILL)

    hullarea = 0
    cavityarea = 0
    for (u,in_hull) in H.piter():
        if in_hull:
            hullarea += 1
            if G.pget(u) != FILL:
                cavityarea += 1
    convexity = 1.0 - cavityarea * 1.0 / hullarea

    # H.printself()
    # save_grid_png(H, 'hull.png')

    print 'compute_convexity: 1.0 - %d / %d' % (cavityarea, hullarea)

    return convexity

def compute_outside_mask(G):
    """ A cell is 'outside' if it's 1) free and 2) reachable from the border via other free cells """
    M = Grid2.new_same_size(G, False)
    def check_edge(u,v):
        return G.pget(v) == FREE and not M.pget(v)
    for u in G.iterborder():
        if G.pget(u) == FREE:
            for v in G.bfs(u, check_edge, 4):
                M.pset(v, True)

    return M

def enforce_symmetry(G, min_symmetry, axis, force_contiguous):
    """ returns true if G was modified """
    assert 0 <= min_symmetry
    assert min_symmetry <= 1.0

    # having odd length simplifies some things
    assert G.W % 2 == 1
    assert G.H % 2 == 1

    center = Int2(G.W/2, G.H/2)

    perp = axis.turn(1)

    def reflect(u):
        perpcoord = (u-center).dot(perp)
        return u - perp*(2*perpcoord)

    # compute unmatched (single) fills
    totalfills = 0
    singles = []
    for (u,p) in G.piter():
        if p == FILL:
            totalfills += 1
            v = reflect(u)
            if G.pget(v) != FILL:
                singles += [u]

    def curr_symmetry():
        return 1.0 - len(singles)*1.0 / totalfills

    modded = False
    while curr_symmetry() < min_symmetry:
        # find a single to pair up, but make sure to do it contiguously
        eligible = []
        for u in singles:
            v = reflect(u)
            if not force_contiguous or G.touches4(v, FILL):
                eligible += [u]

        if len(eligible) == 0:
            G.printself()
            print 'UH OH could not find any eligible points to increase symmetry'

        lucky = random.choice(eligible)
        singles.remove(lucky)
        other = reflect(lucky)
        G.pset(other, FILL)
        totalfills += 1
        modded = True
    return modded

def enforce_convexity(G, min_convexity):
    """ returns true if G was modified """
    H = compute_convex_mask(G, FILL)
    outside = compute_outside_mask(G)
    hullarea = 0
    cavities = []
    for (u,in_hull) in H.piter():
        if in_hull:
            hullarea += 1
            if G.pget(u) != FILL and outside.pget(u):
                cavities += [u]

    # TEMP TEMP fill all non-outside holes
    # or maybe not so temp..this works quite well
    for (u, p) in G.piter():
        if not outside.pget(u):
            G.pset(u, FILL)

    def curr_convexity():
        return 1.0 - len(cavities) * 1.0 / hullarea

    modded = False
    while curr_convexity() < min_convexity:
        # find a cavity to fill, but only if it's adjacent to an existing fill
        eligible = []
        for u in cavities:
            if G.touches4(u, FILL):
                eligible += [u]

        lucky = random.choice(eligible)
        cavities.remove(lucky)
        G.pset( lucky, FILL )
        modded = True

    return modded

def enforce_min_nbors(G, min_nbors, on_value, off_value):
    modded = False
    freethese = []
    for (u, p) in G.piter():
        if p != on_value: continue
        count = 0
        for (v, q) in G.nbors4(u):
            if q == on_value: count+= 1

        if count < min_nbors:
            freethese.append(u)
            modded = True

    for u in freethese:
        G.pset(u, off_value)
    return modded

def enforce_boxiness(G, min_boxiness):
    bbox = G.bbox(lambda p : p == FILL)

    cavities = []
    numfills = 0
    for (u, p) in G.piter():
        if p == FREE and bbox.contains(u):
            cavities += [u]
        elif p == FILL:
            numfills += 1

    def curr_boxiness():
        return 1.0 - len(cavities)*1.0/numfills
    modded = False
    while curr_boxiness() < min_boxiness:
        eligible = []
        for u in cavities:
            if G.touches4(u, FILL):
                eligible += [u]

        lucky = random.choice(eligible)
        cavities.remove(lucky)
        G.pset( lucky, FILL )
        modded = True

    return modded

def test_symmetry():
    L = 41
    G = Grid2(L,L,FREE)
    cent = Int2(L/2, L/2)
    G.pset(cent, FILL)
    seed_spread([FILL], 0, G, FREE, L*L/3)

    for (u,p) in G.piter():
        if u.y <= L/2:
            G.pset(u, FREE)

    assert compute_symmetry(G, Int2(1,0)) == 0.0

def test_y_symmetry():
    L = 11
    G = Grid2(L, L, FREE)
    for (u, p) in G.piter():
        if random.random() < 0.2:
            G.pset(u, FILL)
    center = Int2(L/2, L/2)
    axis = Int2(0,1)
    G.printself()
    print compute_symmetry(G, axis)

    enforce_symmetry( G, 1.0, axis, False )
    G.printself()
    print compute_symmetry(G, axis)

def fixed_point_iterate(steps, max_steps):
    assert type(steps) == list
    assert type(max_steps) == int
    """ each step should return True if it changed the state """
    modded = True
    num_steps = 0
    while modded and num_steps < max_steps:
        modded = False
        for (label, func) in steps:
            with PROFILE(label):
                thismodded = func()
            modded = modded or thismodded

        num_steps += 1

def clamp01(x):
    return max(0.0, min(1.0, x))

def lerp(a, b, t):
    return a + t*(b-a)

def ceilodd(x):
    if x % 2 == 0:
        return x+1
    else:
        return x

def quadratic_formula(a, b, c):
    dis = b*b - 4*a*c
    if dis < 0:
        return (None, None)
    else:
        return ( (-1*b + math.sqrt(dis)) / (2*a),
                (-1*b - math.sqrt(dis)) / (2*a))

def random_width_height(min_delta, max_delta):
    assert min_delta < max_delta
    delta = int(lerp( min_delta, max_delta, random.random() ))
    # 1600 = x * y
    # 1600 = x * (x+delta)
    # 0 = x^2 + delta*x - 1600
    width = int(math.ceil(max(quadratic_formula(1, delta, -1600))))
    height = width + delta
    return (width, height)

if __name__ == '__main__':
    test_symmetry()
    test_y_symmetry()

    for i in range(20):
        with PROFILE('shape %d' % i):
            # means and stdevs are hand-tuned
            # good params are, xs=0.0, ys=1.0, cv=0.95
            # so these gamma distributions are meant to typically yield values around those
            # min_xsym = clamp01(random.gammavariate(1.0, 2.0)/20.0 * 1.0)
            # min_ysym = clamp01(1.0 - random.gammavariate(1.0, 2.0)/20.0 * 0.2)
            # min_convex = clamp01(1.0 - random.gammavariate(1.0, 2.0)/20.0 * 0.10)
            # min_xsym, min_ysym, min_convex, min_boxiness = (0, 1, 1.0, 0.0)
            # min_xsym, min_ysym, min_convex, min_boxiness = (0.5, 1, 0.90, 0.0)
            # min_xsym, min_ysym, min_convex, min_boxiness = (0.7, 1, 0.98, 0.8)
            # min_xsym, min_ysym, min_convex, min_boxiness = (0.5, 1, 0.95, 0.0)  # best 9/10
            min_xsym, min_ysym, min_convex, min_boxiness = (random.random(), 1.0, 0.8, 0.0)
        
            print 'constraint parameters:', min_xsym, min_ysym, min_convex

            width = 75
            height = 75
            width = ceilodd(width)
            height = ceilodd(height)

            G = Grid2(width,height,FREE)
            cent = Int2(width/2, height/2)
            G.pset(cent, FILL)
            with PROFILE('spread'):
                seed_spread([FILL], 0, G, FREE, width*height/4)

            def print_status():
                print 'xsym %f \t ysym %f \t convex %f' % (
                        compute_symmetry( G, Int2(1,0) ),
                        compute_symmetry( G, Int2(0,1) ),
                        compute_convexity( G ))
                return False

            fixed_point_iterate([
                    ('xsym', lambda : enforce_symmetry(G, min_xsym, Int2(1,0), True)),
                    ('ysym', lambda : enforce_symmetry(G, min_ysym, Int2(0,1), True)),
                    ('conv', lambda : enforce_convexity(G, min_convex)),
                    ('box', lambda : enforce_boxiness(G, min_boxiness)),
                    ('nborsout', lambda : enforce_min_nbors(G, 2, FREE, FILL)),
                    ('nborsin', lambda : enforce_min_nbors(G, 2, FILL, FREE)),
                    ('eval', lambda : print_status()) ],
                    3 )

            save_grid_png(G, 'shape-%d.png' % i)
            G.printself()
