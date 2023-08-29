import numpy as np
from ifigure.interactive import figure

fig = figure()

def func(nc = model.cql3d_rfnc, bins = 20, style = 'std'):  
  delpwr = nc.get_contents("variables","delpwr")
  spsi = nc.get_contents("variables","spsi")
  wnpar = nc.get_contents("variables","wnpar")
  dspsi = np.gradient(spsi)[1]  #derivative along the path
  ddelpwr = -np.gradient(delpwr)[1]  #derivative along the path


  nray = delpwr.shape[0]
  nlen = delpwr.shape[1]-1

  ndelpwr = delpwr/delpwr[:,0].reshape(delpwr.shape[0], -1)
  
  ddelpwr = ddelpwr[np.logical_and(spsi < 1.0, delpwr > 0.0)]
  wnpar = wnpar[np.logical_and(spsi < 1.0, delpwr > 0.0)]
  
  if style == 'std':
     d, x = np.histogram(wnpar, weights = ddelpwr, bins = bins)
     x  = (x[1:] + x[:-1])/2.0
  else:
     d, x = np.histogram(1./wnpar, weights = ddelpwr, bins = bins)
     x  = (x[1:] + x[:-1])/2.0
     x  = 1./x


  fig.plot(x, d)
  return d, x
#  fig.plot(a[:,0], a[:,1], 'r')
#  fig.plot(b[:,0], b[:,1], 'b')


ans(func(*args, **kwargs))