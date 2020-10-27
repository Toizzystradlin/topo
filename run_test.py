import matplotlib.pyplot as plt
from neural_structural_optimization import problems
from neural_structural_optimization import my_models
from neural_structural_optimization import topopogy1
from neural_structural_optimization import train
import matplotlib.cm
import matplotlib.colors
import numpy as np
from PIL import Image
import xarray

def main(problem, max_iterations):
    args = topopogy1.define_task(problem)
    model = my_models.myModel(args=args)
    ds_pix = train.trainer(model, max_iterations)
    return ds_pix

task = problems.forcemap()
n = 200

data = main(task, n)

data.loss.transpose().to_pandas().cummin().loc[:2000].plot(linewidth=2)
plt.ylabel('Податливость')
plt.xlabel('Шаг оптимизации')
plt.show()

def make_gif(images, path, duration=100, loop=0):
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=duration, loop=loop)

def layer_to_image(design):
  imaged_designs = []
  imaged_designs.append(design)
  #print('design: ' + str(design))
  #print('imaged_design: ' + str(imaged_designs))
  norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
  mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap='Greys')
  frame = np.ma.masked_invalid(xarray.concat(imaged_designs, dim='x').data)
  image = Image.fromarray(mappable.to_rgba(frame, bytes=True), mode='RGBA')
  return image

images = [layer_to_image(design) for design in data.design.sel()[:1000:5]]

make_gif([im.resize((30 * 30, 30 * 30)) for im in images], 'movie.gif')
