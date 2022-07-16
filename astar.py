from queue import PriorityQueue
import itertools
from heapq import *

class heap(object):

    def __init__(s):
        s.reset()

    def reset(s):
        s.pq = []                         # list of entries arranged in a heap
        s.entry_finder = {}               # mapping of items to entries
        s.REMOVED = '<removed-item>'      # placeholder for a removed item
        s.counter = itertools.count()     # unique sequence count
        s.count = 0

    def add(s, item, priority=0):
        'Add a new item or update the priority of an existing item'
        if item in s.entry_finder:
            s.remove(item)
        count = next(s.counter)
        entry = [priority, count, item]
        s.entry_finder[item] = entry
        heappush(s.pq, entry)
        s.count += 1

    def remove(s, item):
        'Mark an existing item as REMOVED.  Raise KeyError if not found.'
        entry = s.entry_finder.pop(item)
        entry[-1] = s.REMOVED
        s.count -= 1

    def empty(s):
        return s.count == 0

    def pop(s):
        'Remove and return the lowest priority item. Raise KeyError if empty.'
        while s.pq:
            priority, count, item = heappop(s.pq)
            if item is not s.REMOVED:
                del s.entry_finder[item]
                s.count -= 1
                return item
        raise KeyError('pop from an empty priority queue')

def astar(start, target, yield_nbors, edge_cost, est_to_target):

    cost_to = {}
    prev = {}
    processed = set()
    hq = heap()

    u = start
    cost_to[u] = 0
    prev[u] = None

    while u != target:
        for v in yield_nbors(u):
            if v in processed:
                continue
            suv = cost_to[u] + edge_cost(u,v)
            if v not in cost_to or suv < cost_to[v]:
                cost_to[v] = suv
                prev[v] = u
                cost_thru = suv + est_to_target(v)
                hq.add(v, cost_thru)
        processed.add(u)

        if hq.empty():
            raise RuntimeError("No path!")

        u = hq.pop()
        if u == target:
            # done!
            break

    # now yield the best path backwards
    u = target
    while u:
        yield u
        u = prev[u]

from utils import *
from random import *
import noise
def test_grid_path():
    BLOCK = '#'
    L = 60
    G = Grid2(L, L, ' ')
    xofs = random() * 10.0
    yofs = random() * 10.0
    thresh = -0.5 + random()*1.0
    for (u,p) in G.piter():
        if noise.pnoise2(u.x*10.0/L + xofs, u.y*10.0/L + yofs) < thresh:
            G.pset(u, BLOCK)

    def yield_nbors(u):
        for v in G.nbors4(u):
            yield v[0]

    def edge_cost(u,v):
        if G.pget(v) == BLOCK:
            return 9999
        else:
            return 1

    start = Int2(0,0)
    target = Int2(L-1, L-1)

    def est_to_target(u):
        return Int2.euclidian_dist(u, target)

    G.printself()
    print('---')

    breaks = 0
    for u in astar( start, target, yield_nbors, edge_cost, est_to_target ):
        if G.pget(u) == BLOCK:
            G.pset(u, 'X')
            breaks += 1
        else:
            G.pset(u, '.')

    G.printself()
    print(breaks)

def test_heap():
    h = heap()
    h.add( 'b', 2 )
    h.add( 'a', 1 )
    h.add( 'c', 3 )

    assert h.pop() == 'a'
    assert h.pop() == 'b'
    assert h.pop() == 'c'

if __name__ == '__main__':
    test_heap()
    test_grid_path()

