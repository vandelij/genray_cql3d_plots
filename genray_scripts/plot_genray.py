
#   Run()
#   Run(nc = proj.model.genray_nc) : arbitray nc file
#   Run(allrays = True) : plot all rays
#   Run(raystop = '> 0.6'):
#   Run(color = 'power' or 'npara')
#   Run(ridx = [1,2,3])
#   Run(negative = True)
#   Run(positive = True) 
#   Run(npower = True)  : normalize power


import numpy as np
import ifigure.interactive as plt
from scipy.signal import argrelextrema

def func(allrays = True, showpeak = True, shownegative = False,
         showpositive = False, raystop = None, color= 'power', 
         npower = True, nc = model.genray_nc, ridx = None, **kwargs):
    
    gfile = param.eval('gfile')
    xlim = gfile.get_contents("table","xlim")
    ylim = gfile.get_contents("table","ylim")
    rbbbs = gfile.get_contents("table","rbbbs")
    zbbbs = gfile.get_contents("table","zbbbs")
    
    wr  = kwargs['wr'] if 'wr' in kwargs else nc.get_contents("variables","wr")
    wz  = kwargs['wz'] if 'wz' in kwargs else nc.get_contents("variables","wz")
    wnpar = nc.get_contents("variables","wnpar")
    wnper = nc.get_contents("variables","wnper")
    wn_phi = nc.get_contents("variables","wn_phi")
    spsi  = nc.get_contents("variables","spsi")
    ste   = nc.get_contents("variables","ste")
    delpwr= nc.get_contents("variables","delpwr")
    delpwr0= nc.get_contents("variables","delpwr")
    nrayelt= nc.get_contents("variables","nrayelt")
    
    
    if showpositive and ridx is None:
        ridx = np.where(wnpar[:,0] > 0)[0]
    if shownegative and ridx is None:
        ridx = np.where(wnpar[:,0] < 0)[0]
    if allrays:
        ridx = np.arange(len(wnpar[:,0]))
    if raystop is not None:
        ridx = [x for x in range(wnpar.shape[0])
                if eval('spsi[x,nrayelt[x]-1]' + raystop)]
    if allrays:
        ridx = range(len(wnpar[:,0]))
    if showpeak and ridx is None:
        print('ploting spectrum peak')
        ridx = argrelextrema(delpwr[:,0], np.greater_equal)[0]
    #wnpar[np.where(abs(wnpar) < 0.1)]= 1
    
    if 'wr' in kwargs:
        for x in range(wnpar.shape[0]):
            idx = np.where(wr[x,:] == 0)[0]
            if len(idx) != 0:
                nrayelt[x] = np.min(idx)
                wnpar[x,np.min(idx)] = 0.0
    num = 0
    for x in range(wnpar.shape[0]):
       idx = np.where(wnpar[x,:] == 0)
       if delpwr[x,0]!=0.0 and npower:
          delpwr[x,:] = delpwr[x,:]/delpwr[x,0]
       if len(idx[0]) != 0:
          wnpar[x,nrayelt[x]:] = wnpar[x, nrayelt[x]-1]
          wz[x,nrayelt[x]:] = wz[x, nrayelt[x]-1]
          wr[x,nrayelt[x]:] = wr[x, nrayelt[x]-1]
          spsi[x,nrayelt[x]:] = spsi[x, nrayelt[x]-1]
          ste[x,nrayelt[x]:] = ste[x, nrayelt[x]-1]
    
       if wnpar[x,0] == 0.0:
          wnpar[x,:] = 1.0
          num = num + 1
       
    print('Number of rays which was not computed ' + str(num))
    if not allrays: print('Selected rays ' + str(ridx))
    
    if color == 'power':
       zdata = delpwr
    elif color == 'power0':
       zdata = delpwr0
    elif color == 'npara':
       zdata = wnpar
    elif color == 'dnpara':
       zdata = wnpar[:, 1:]- wnpar[:, :-1]
       zdata = np.hstack((zdata, np.zeros((zdata.shape[0], 1))))
    else: zdata = color
    
    viewer = plt.figure()
    viewer.update(False)
    plt.nsec(2,4,(0,1))
    plt.hold(1)
    plt.isec(6)
    if ridx is None:
       plt.plot(wr*0.01, wz*0.01, zdata, cz = True)
    else:
       plt.plot(wr[ridx]*0.01, wz[ridx]*0.01, zdata[ridx], cz = True)
    plt.plot(xlim, ylim, 'r')
    plt.plot(rbbbs, zbbbs, 'b')
    
    if ridx is None: ridx = np.arange(wnpar.shape[0])
    plt.isec(0)
    plt.plot(1/wnpar[ridx], spsi[ridx], zdata[ridx], cz=True)
    
    v = 2.5*np.sqrt(2.*1.6e-19*1000*ste/9.1e-31)/3e8
    plt.plot(v, spsi)
    plt.plot(-v, spsi)
    
    plt.isec(1)
    if ridx is None:
        plt.plot(delpwr);plt.title('delpwr normalized')
    else:
        plt.plot(delpwr[ridx]);plt.title('delpwr normalized')
    plt.isec(2)
    if allrays:
        plt.plot(wnpar);plt.title('N_parallel;N_perp')
        plt.plot(wnper)
    else:
        plt.plot(wnpar[ridx]);plt.title('N_parallel')
        plt.plot(wnper[ridx])
    
    plt.isec(3) 
    if allrays:
        plt.plot(wn_phi);plt.title('N_toroidal')
    else:
        plt.plot(wn_phi[ridx]);plt.title('N_toroidal')
    
    plt.isec(4)
    plt.plot(spsi[ridx], delpwr[ridx]);plt.title('Power vs rho')
    plt.isec(5)
    plt.plot(wnpar[:, 0], delpwr0[:, 0]);plt.title('Launcher Spectrum')
   
    viewer.update(True)
#ans(ridx)

ans(func(*args, **kwargs))





