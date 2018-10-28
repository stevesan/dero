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
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def contains(self, u):
        return self.a.contains(u) or self.b.contains(u)


class Intersection(Node):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def contains(self, u):
        return self.a.contains(u) and self.b.contains(u)


class Difference(Node):
    def __init__(self, a, b):
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


def random_primitive(center_range, size_range):
    assert type(center_range) == Bounds2
    assert type(center_range) == Bounds2

    center = center_range.random_inside()
    size = size_range.random_inside()

    if random.random() < 0.5:
        return Circle(center, (size.x+size.y)/2/2)
    else:
        return Rectangle(Bounds2.from_center_dims(center, size))


def random_csg_tree(bounds, num_octaves, objs_per_octave):
    size_range = bounds.get_size()/2
    tree = None
    for octave in range(num_octaves):
        denom = 2**(octave+1)
        next_denom = 2**(octave+2)
        size_range = Bounds2(bounds.get_size()/next_denom,
                             bounds.get_size()/denom)
        center_range = bounds.shrink(size_range.maxs/2)
        # print "octave %d, size %s, center %s" % (octave, size_range, center_range)

        for _ in range(objs_per_octave):
            node = random_primitive(center_range, size_range)

            if tree == None:
                tree = node
            else:
                if random.random() < 0.2:
                    tree = Difference(tree, node)
                else:
                    tree = Union(tree, node)

    return tree


def make_vertically_symmetric(tree, center):
    return Union(tree, Reflected(tree, center, Int2(1, 0)))


def test():
    L = 200
    center = Int2(L/2, L/2)

    a = Rectangle(Bounds2.from_center_dims(center+Int2(L/8, 0), Int2(L/2, 2)))
    b = Rectangle(Bounds2.from_center_dims(center+Int2(L/8, 0), Int2(2, L/2)))
    c = Circle(center+Int2(L/8, 0), L/8)
    tree = Union(c, Union(a, b))
    tree = Union(tree, Reflected(tree, center, Int2(1, 0)))

    G = Grid2(L, L, '.')
    tree.rasterize(G, 'X')
    G.save_png('cgstest0.png')


def test1():

    for i in range(30):
        L = 200
        G = Grid2(L, L, '.')
        tree = random_csg_tree(G.get_bounds(), 3, 4)
        tree = make_vertically_symmetric(tree, G.get_center())
        tree.rasterize(G, 'X')
        G.save_png('rand_csg_tree_%d.png' % i)


if __name__ == '__main__':
    test()
    test1()
