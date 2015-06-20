import random
import numpy
from pylab import *

nbordx = [1, 0, -1, 0]
nbordy = [0, -1, 0, 1]

class Point:
	x = 0
	y = 0

	def __init__(self, _x, _y):
		self.x = _x
		self.y = _y

	def yieldtest(self):
		yield 1
		yield 2

	def yield_4nbors(self):
		""" in ccw order """
		p = self
		for i in range(4):
			yield Point(p.x+nbordx[i], p.y+nbordy[i])

	def yield_4nbors_rand(self):
		""" in random order """
		p = self
		for i in numpy.random.permutation(4):
			yield Point(p.x+nbordx[i], p.y+nbordy[i])

class Grid:
	W = 1
	H = 1
	grid = None

	def __init__(self,_W, _H):
		self.W = _W
		self.H = _H
		self.grid = range(self.W*self.H)
		for i in range(self.W*self.H):
			self.grid[i] = ' '

	def check(self,p):
		return p.x >= 0 and p.x < self.W and p.y >= 0 and p.y < self.H

	def get(self,x,y):
		return self.grid[self.W*y + x]

	def pget(self,p):
		return self.grid[self.W*p.y + p.x]

	def set(self,x,y,value):
		self.grid[self.W*y+x] = value

	def pset(self,p,value):
		self.grid[self.W*p.y+p.x] = value

	def printgrid(self):
		for y in range(self.H):
			for x in range(self.W):
				print self.get(x,y),
			print ''

	def iter(self):
		for y in range(self.W):
			for x in range(self.H):
				yield (x,y)

def testBasic():
	g = Grid(80,80)

	for (x,y) in g.iter():
		if random.random() < 0.2:
			g.set(x,y,'X')

	g.printgrid()


def gaussian(frac, stdev):
	return math.exp(-1 * (frac*frac)/(stdev*stdev))

class InterestCurve:
	firstBump = 0.5
	lastBump = 1.0
	def eval(self, frac):
		rv = 0
		rv += self.firstBump * gaussian(frac-0.1, 0.10)
		for center in [ 0.3, 0.5, 0.7]:
			rv += (center*0.8) * 1.0 * gaussian(frac-center, 0.075)
		rv += 1.0 * gaussian(frac-0.95, 0.1)
		return rv

curve = InterestCurve()
ff = range(200)
ii = range(200)
for i in range(len(ff)):
	ff[i] = i*1.0/(len(ff)-1)
# ii[i] = gaussian(ff[i], 0.2)
	ii[i] = curve.eval(ff[i])

plot(ff,ii,'.-')
xlim([0, 1])
ylim([0, 1])
# show()

g = Grid(40,40)

def tunnel(g, p, c):
	# only dig if one nbor only is 
	foundOne = False
	for nbor in p.yield_4nbors():
		if not g.check(nbor): continue
		if g.pget(nbor) == c:
			if foundOne:
				return
			else:
				foundOne = True
	# ok, dig it
	g.pset(p, c)
	# recurse
	for nbor in p.yield_4nbors_rand():
		if not g.check(nbor): continue
		if g.pget(nbor) != c:
			tunnel(g, nbor, c)

tunnel(g, Point(0, 0), 'X')
g.printgrid()
