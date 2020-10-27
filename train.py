import autograd.numpy as np
from neural_structural_optimization import topology2
import tensorflow as tf
import xarray

def trainer(model, max_iterations, optimizer=tf.keras.optimizers.Adam(1e-2)):
    tvars = model.trainable_variables
    losses = []
    frames = []
    for i in range(max_iterations + 1):
        with tf.GradientTape() as t:
            t.watch(tvars)
            logits = model(None)
            loss = model.loss_func(logits)
        losses.append(loss.numpy().item())
        frames.append(logits.numpy())
        if i < max_iterations:
            grads = t.gradient(loss, tvars)
            optimizer.apply_gradients(zip(grads, tvars))
    #for i in frames:
    #    print(i)
    losses = np.array(losses)
    #designs =
    designs = [topology2.real_density(np.reshape(x, (model.args['nely'], model.args['nelx'])), model.args) for x in frames]
    frames = np.array(designs)

    data = xarray.Dataset({
        'loss': (('step',), losses),
        'design': (('step', 'y', 'x'), frames),
    }, coords={'step': np.arange(len(losses))})

    return data