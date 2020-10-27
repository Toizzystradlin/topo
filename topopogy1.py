import autograd.numpy as np
from neural_structural_optimization import topology2

def define_task(problem):
  fixdofs = np.flatnonzero(problem.normal_forces.ravel())
  alldofs = np.arange(2 * (problem.width + 1) * (problem.height + 1))
  freedofs = np.sort(list(set(alldofs) - set(fixdofs)))

  params = {
      'young': 1,
      'young_min': 1e-9,
      'poisson': 0.3,
      'volfrac': problem.density,
      'nelx': problem.width,
      'nely': problem.height,
      'freedofs': freedofs,
      'fixdofs': fixdofs,
      'forces': problem.external_forces.ravel(),
      'penal': 3.0,
  }
  return params

class Environment:
  def __init__(self, args):
    self.args = args

  def reshape(self, params):
    return params.reshape(self.args['nely'], self.args['nelx'])

  def render(self, params):
    return topology2.real_density(self.reshape(params), self.args)

  def objective(self, params, ke):
    return topology2.objective(self.reshape(params), ke, self.args)
