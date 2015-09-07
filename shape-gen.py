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

    return convexity

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
    hullarea = 0
    cavities = []
    for (u,in_hull) in H.piter():
        if in_hull:
            hullarea += 1
            if G.pget(u) != FILL:
                cavities += [u]

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


if __name__ == '__main__':
    test_symmetry()
    test_y_symmetry()

    min_xsym = float(sys.argv[1])
    min_ysym = float(sys.argv[2])
    min_convex = float(sys.argv[3])
    L = 41
    sym_axis = Int2(0,1)



    G = Grid2(L,L,FREE)
    cent = Int2(L/2, L/2)
    G.pset(cent, FILL)
    seed_spread([FILL], 0, G, FREE, L*L/3)

    save_grid_png(G, 'before-enforcements.png')

    modded = True
    while modded:
        print 'x-symmetry', compute_symmetry(G, Int2(1,0))
        print 'y-symmetry', compute_symmetry(G, Int2(0,1))
        print 'convexity', compute_convexity(G)
        modded = False
        modded = modded or enforce_symmetry(G, min_xsym, Int2(1,0), True)
        modded = modded or enforce_symmetry(G, min_ysym, Int2(0,1), True)
        modded = modded or enforce_convexity(G, min_convex)

    save_grid_png(G, 'final.png')
    G.printself()
