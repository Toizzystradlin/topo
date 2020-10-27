#import dataclasses
import numpy as np

X, Y = 0, 1

class ForceMap:
    def __init__(self, normal_forces, external_forces, density):
        self.normal_forces = normal_forces
        self.external_forces = external_forces
        self.density = density
        self.width = self.normal_forces.shape[0] - 1
        self.height = self.normal_forces.shape[1] - 1


def forcemap(width=30, height=30, density=0.5):

  normal_forces = np.zeros((width + 1, height + 1, 2))
  normal_forces[0, -1, 1] = 1
  normal_forces[-1, -1, 1] = 1

  external_forces = np.zeros((width + 1, height + 1, 2))
  external_forces[7, 0, 1] = -1

  return ForceMap(normal_forces, external_forces, density)