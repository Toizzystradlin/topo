
import autograd
import autograd.core
import autograd.numpy as np
from neural_structural_optimization import topopogy1
import tensorflow as tf
from neural_structural_optimization import problems
from neural_structural_optimization import topology2

def grads_to_tensors(func):
    @tf.custom_gradient
    def loss_func(x):
        vjp, end_value = autograd.core.make_vjp(func, x.numpy())
        return end_value, vjp
    return loss_func

class myModel(tf.keras.Model):
    def __init__(self, args=None):
        super().__init__()
        self.env = topopogy1.Environment(args)
        form = (1, self.env.args['nely'], self.env.args['nelx'])
        layer = np.broadcast_to(args['volfrac'], form)
        self.layer = tf.Variable(layer, trainable=True)
        self.args = args

    def loss_func(self, logits):
        def f(params):
            ke = topology2.make_stiffness_matrix(args['young'], args['poisson'])
            losses = self.env.objective(params, ke)
            #losses = topo_physics.objective(params, ke, args)
            return losses
        final_loss_list = grads_to_tensors(f)(logits)
        return tf.reduce_mean(final_loss_list) #возвращает среднее значение массива

    def call(self, inputs=None):
        return self.layer

problem = problems.forcemap()
max_iterations = 200
args = topopogy1.define_task(problem)
model = myModel(args=args)
model(None)
model.summary()