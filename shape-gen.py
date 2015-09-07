from utils import *

FREE = '.'
FILL = 'X'

def compute_symmetry( G, axis ):
    C = Int2(G.W/2, G.H/2)
    H = Int2(G.W/2, G.H/2)
    perp = axis.turn(1)
    def reflect(u):
        return u - (u-C).scale(perp)*2

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
    H = compute_convex_mask(G, FREE, FILL)

    hullarea = 0
    cavityarea = 0
    for (u,p) in H.piter():
        if p == FILL:
            hullarea += 1
            if G.pget(u) != FILL:
                cavityarea += 1
    convexity = 1.0 - cavityarea * 1.0 / hullarea

    # H.printself()
    # save_grid_png(H, 'hull.png')

    return convexity

def enforce_symmetry(G, target_symmetry, axis):
    assert 0 <= target_symmetry
    assert target_symmetry <= 1.0

    # having odd length simplifies some things
    assert G.W % 2 == 1
    assert G.H % 2 == 1

    center = Int2(G.W/2, G.H/2)

    perp = axis.turn(1)

    def reflect(u):
        return u - (u-center).scale(perp)*2

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

    while curr_symmetry() < target_symmetry:
        print 'curr sym', curr_symmetry()
        # find a single to make whole, but make sure to do it contiguously
        eligible = []
        for u in singles:
            v = reflect(u)
            if G.touches4(v, FILL):
                eligible += [u]

        lucky = random.choice(eligible)
        singles.remove(lucky)
        other = reflect(lucky)
        G.pset(other, FILL)
        totalfills += 1

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

if __name__ == '__main__':
    test_symmetry()

    sym = float(sys.argv[1])
    convex = float(sys.argv[2])

    L = 41

    G = Grid2(L,L,FREE)
    cent = Int2(L/2, L/2)
    G.pset(cent, FILL)

    seed_spread([FILL], 0, G, FREE, L*L/3)


    save_grid_png(G, 'pre-symmetry-enforce.png')

    enforce_symmetry(G, sym, Int2(1,0))

    save_grid_png(G, 'post-symmetry-enforce.png')

    G.printself()

    print 'convexity', compute_convexity(G)

    print 'symmetry', compute_symmetry(G, Int2(1,0))
