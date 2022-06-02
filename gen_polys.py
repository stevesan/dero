"""
"""

import pygame
import random
from dataclasses import dataclass

@dataclass
class PolyGenArgs(object):
  symmetry: float = 1.0
  num_lines: int = 6
  size: tuple[int, int] = (100, 100)
  # 0 means no concavity
  concavity: float = 1.0
  # 1.0 means all sides will be the same length and all angles will be equal.
  regularity: float = 0.5


def gen_poly(args):
  """Returns a list of points meant to define a closed polygon"""
  # just random for now haha
  assert args.num_lines >= 3
  W = args.size[0]
  H = args.size[1]
  return [(random.random()*W, random.random()*H) for i in range(args.num_lines)]

def main():
  rows = 5
  cols = 10
  count = rows * cols

  L = 100

  def GenRandomPolyArgs():
    return PolyGenArgs(
      num_lines=random.randrange(3,10),
      size=(random.randrange(L/2, L), random.randrange(L/2, L))
      )

  polys = [gen_poly(GenRandomPolyArgs()) for i in range(count)]

  width = cols * L
  height = rows * L
  screen_color = (0, 0, 0)
  line_color = (255, 0, 0)

  screen = pygame.display.set_mode((width,height))
  screen.fill(screen_color)

  xofs = 0
  yofs = 0
  for poly in polys:
    pts = [(p[0]+xofs, p[1]+yofs) for p in poly]
    pygame.draw.polygon(screen, line_color, pts, 0)
    xofs += L

    if xofs >= width:
      xofs = 0
      yofs += L

  pygame.display.flip()

  while True:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return
main()
