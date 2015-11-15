
from utils import *
from euclid import *

class Node(object):
    def contains(self, u):
        raise NotImplementedError()

    def rasterize(self, grid, value):
        for (u, p) in grid.piter():
            if self.contains(u):
                grid[u] = value

class Union(Node):
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def contains(self, u):
        return self.a.contains(u) or self.b.contains(u)

class Intersection(Node):
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def contains(self, u):
        return self.a.contains(u) and self.b.contains(u)

class Difference(Node):
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def contains(self, u):
        return self.a.contains(u) and not self.b.contains(u)

class Rectangle(Node):
    def __init__(self, bounds):
        self.bounds = bounds

    def contains(self, u):
        return self.bounds.contains(u)

class Circle(Node):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def contains(self, u):
        return Int2.euclidian_dist(self.center, u) <= self.radius

class Reflected(Node):
    def __init__(self, node, reflect_point, reflect_normal):
        self.node = node
        self.reflect_point = reflect_point
        self.reflect_normal = reflect_normal

    def contains(self, u):
        d = u - self.reflect_point
        p = self.reflect_normal * d.dot(self.reflect_normal)
        return self.node.contains(u - p*2)

def test():
    L = 200
    center = Int2(L/2, L/2)

    a = Rectangle(Bounds2.from_center_dims(center+Int2(L/8, 0), Int2(L/2, 2)))
    b = Rectangle(Bounds2.from_center_dims(center+Int2(L/8, 0), Int2(2, L/2)))
    c = Circle(center+Int2(L/8,0), L/8)
    tree = Union(c, Union(a,b))
    tree = Union(tree, Reflected(tree, center, Int2(1,0)))

    G = Grid2(L, L, '.')
    tree.rasterize(G, 'X')
    G.save_png('cgstest.png')

if __name__ == '__main__':
    test()
