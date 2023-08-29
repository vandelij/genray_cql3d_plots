from ifigure.interactive import figure
import numpy as np

urfb = model.cql3d_rfnc.get_contents("variables","urfb")
rya = model.cql3d_nc.get_contents("variables","rya")
x = model.cql3d_nc.get_contents("variables","x")
vn = model.cql3d_nc.get_contents("variables","vnorm") 
en = model.cql3d_nc.get_contents("variables","enorm")
c = 2.99792458e8
mr = 510.998910
gn = en/mr  + 1. # gamma norm
uc = np.sqrt(gn**2 - 1)   # u/c
   
u = vn*x/100.      # this is u ( = p/m_r, = gamma*v)
gamma = np.sqrt( 1.+ u*u/c/c)
vn = u/gamma/c   

v = figure();v.image(vn, rya, urfb[:,:,0])