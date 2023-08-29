import numpy as np

def func(nc = model.cql3d_rfnc, bins = 20, style = 'std'):  
  delpwr = nc.get_contents("variables","delpwr")
  spsi = nc.get_contents("variables","spsi")
  wnpar = nc.get_contents("variables","wnpar")
  dspsi = np.gradient(spsi)[1]  #derivative along the path
  ddelpwr = -np.gradient(delpwr)[1]  #derivative along the path
  print ddelpwr.shape

  nray = delpwr.shape[0]
  nlen = delpwr.shape[1]-1

  ndelpwr = delpwr/delpwr[:,0].reshape(delpwr.shape[0], -1)


  idxp = wnpar[:,0]>1.0
  idxm = wnpar[:,0]<1.0

  a =  np.sum(ddelpwr[idxp,:][spsi[idxp,:]  > 1.0])
  b =  np.sum(ddelpwr[idxp,:][spsi[idxp,:]  < 1.0])
  c =  np.sum(ddelpwr[idxm,:][spsi[idxm,:]  > 1.0])
  d =  np.sum(ddelpwr[idxm,:][spsi[idxm,:]  < 1.0])

  print 'SOL  absorption (N//>0)', a
  print 'Core absorption (N//>0)', b
  print 'SOL  absorption (N//<0)', c
  print 'Core absorption (N//<0)', d

  print (a+b+d)/(a+b+c+d)
  print (a+b)/(a+b+c+d)
  print b/(a+b+c+d)

#  fig.plot(a[:,0], a[:,1], 'r')
#  fig.plot(b[:,0], b[:,1], 'b')


ans(func(*args, **kwargs))