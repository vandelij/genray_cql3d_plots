import sys, matplotlib.pyplot as plt
import numpy as np
sys.path.insert(1,'/nobackup1/jamalj/nt_arc/run1_BALOO_v1_nominal_new')
import coordinate_tools as ct

#-------------------------------------------------------------------------------
#LH access condition plotting script
#-------------------------------------------------------------------------------
#
# This is a Python script that allows you to calculate the wave accessibility 
# versus r/a at the midplane for both the high-field and the low-field side
# of the tokamak.

# density calculator in trin.in
def density_at_r(r):
    
    n_avg  = 1.7   # 10^20 m^-3 (volume average)
    n_edge  = 0.9
    n_peaking  = 2.3
    
    return n_avg * ( (1 - n_edge/n_avg) * n_peaking * ( 1 - r**2 )**(n_peaking-1)) + n_edge

def temp_at_r(r):

    #Te_avg  = 12.5   # keV (volume average)
    Te_edge = 4 # force for r/a=1 since profile wasn't evolved
    #Te_peaking  = 2 # wasn't iteratively evolved so still valid for r/a=0
    Te_core=21
    return (Te_core - Te_edge)*(1 - r**2) + Te_edge

# points in profile
npts = 100

# wave frequency
f_0 = 10e9 #Hz
w = 2*np.pi * f_0 #rad/s

# Te, temperature profile
T_0      = 25   #keV on-axis
T_sep    = 4  #keV
T_alpha1 = 1.5
T_alpha2 = 1.5

# ne, density profile 
#n_0      = 4    #10^20 m^-3 on-axis
#n_sep    = 0.8  #10^20 m^-3   
#n_alpha1 = 1.1
#n_alpha2 = 1.5

# B-field on axis
B_0      = 11 #Tesla

# major and minor radius
R_0      = 4.55 # m
a        = 1.2 # m

# parabolic profiles
#def f_parab(A_0,A_s,a_1,a_2,x):
#  return (A_0 - A_s)*(1-x**a_1)**a_2 + A_s

# setup alternate parabolic profile using trin.in form ! n(r) = n_avg * ( (1 - n_edge/n_avg) * n_peaking * ( 1 - r**2 )**(n_peaking-1)) + n_edge

# create plasma profiles (let rho=r/a here)
r_a_access = np.linspace(-1,1,npts)

#nprof = f_parab(n_0,n_sep,n_alpha1,n_alpha2,abs(rho))
nprof = density_at_r(r_a_access)
#print("\nne profile ", nprof)
#tprof = f_parab(T_0,T_sep,T_alpha1,T_alpha2,abs(rho))
tprof = temp_at_r(r_a_access)
#print("\nTe profile ", tprof)
bprof = B_0 * R_0 / (R_0 + a*r_a_access)

# calculate plasma parameters (100,)
# plasma freq
wpe = 5.64e4 * np.sqrt(nprof*1e14)
wpi = 1.32e3 * np.sqrt(nprof/2.5*1e14)

# cyclotron frequencies (100,)
wce = 1.76e7 * bprof*1e4
wci = 9.58e3 * bprof*1e4 /  2.5

# thermal velocity
clight = 3e10 #cm/s
vte = np.sqrt(2)*4.19e7 * np.sqrt(tprof*1e3) # v_th,e (100,)

#calculate upper and lower access bounds
n_ac = (wpe/wce) + np.sqrt(1+(wpe/wce)**2-(wpi/w)**2)
n_ld = clight/(3*vte) # ld : landau-damped

fsz=20
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['xtick.labelsize'] = fsz-3
plt.rcParams['ytick.labelsize'] = fsz-3
plt.plot(r_a_access,n_ld,color='tab:blue',linestyle="--", label='$N_{ELD}$ : electron Landau damping')
plt.plot(r_a_access,n_ac,color='tab:red',label='$N_a$ \t : access cut-off')
plt.fill_between(r_a_access,n_ac,color= 'tab:red', alpha= 0.1)

plt.ylim(1,2.75)
plt.legend(fontsize = fsz)
plt.xlabel('r/a',fontsize = fsz)
plt.ylabel(r'$N_\parallel$',fontsize = fsz)
plt.title("Lower hybrid access at "+str(int(f_0/1e9))+r' GHz for refractive indices $N$', fontsize = fsz)
#plt.title("run1_BALOO_v1_nominal_new")


fig = plt.gcf()
fig.set_size_inches(10, 12)
plt.gca().set_aspect('equal')
#fig.savefig('/home/jamalj/image_backup/1D_access_.png', dpi=300)
plt.show()