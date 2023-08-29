import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d

pi = np.pi
m_e = 9.109e-31
m_i = 2*1.673e-27
q = 1.602e-19
c = 3e8

def getMidplaneB(ryain):
    rgrid = proj.model1.efit_gfile1.get_contents("table","rgrid")
    zgrid = proj.model1.efit_gfile1.get_contents("table","zgrid")

    psirz = proj.model1.efit_gfile1.get_contents("table", "psirz")
    psi_mag_axis = proj.model1.efit_gfile1.get_contents("table", "ssimag")
    psi_boundary = proj.model1.efit_gfile1.get_contents("table", "ssibdry")
    
    psirzNorm = (psirz - psi_mag_axis)/(psi_boundary-psi_mag_axis)
    #interpolated function for poloidal flux
    psirzNormFunc = interp2d(rgrid, zgrid[9:-10], psirzNorm[9:-10, :])
   
    R_mag = proj.model1.efit_gfile1.get_contents("table","rmaxis")
    B_magaxis = proj.model1.efit_gfile1.get_contents("table","bcentr")

    Rs = np.linspace(1,R_mag,100)
    Bs = B_magaxis * R_mag / Rs

    rhos = np.zeros(len(Rs))
    for index in range(len(Rs)):
        rhos[index] = np.sqrt(psirzNormFunc(Rs[index],0))
      
    rhos = np.append(rhos, [0])

    Bs = np.append(Bs, [B_magaxis])

    getB_rho = interp1d(rhos, Bs, kind = 'linear')
    return getB_rho(ryain)

def main():
    cqlinput = proj.model1.genray_cql3d.cql3d.cqlinput
    ryain = np.array(cqlinput.get_contents("setup","ryain"))
    te = cqlinput.get_contents("setup","tein")
    ti = cqlinput.get_contents("setup","tiin")
    ne = np.array([x*1e6 for x in cqlinput.get_contents("setup","enein(1,1)")])

    Bmidplane = getMidplaneB(ryain)

    w_pe = 5.64e4*np.sqrt(ne/1e6)

    w  = 2*pi*4600000000
    #n_para = -2.7
    w_ce = q*Bmidplane/m_e; w_ci = q*Bmidplane/m_i
    v_Te = 4.19e5*np.sqrt(te)

    w_hat = w/np.sqrt(w_ce*w_ci)
    alpha = w_hat**4/(1-w_hat**2) #wpe[0]**2/w_ce[0]**2
    n_para = 1/(1-w_hat**2)

    N = w_pe**2/(alpha*w_ce**2)
    zeta_0 = c**2/(n_para**2*v_Te[0]**2)
    zeta = c**2/(n_para**2*v_Te**2)
    tau = zeta_0/zeta
    n_perp = -np.sqrt(m_i/m_e)*(w_hat**2/(1-w_hat**2))

    D = 1 + (1-w_hat**4)*N + (1-w_hat**2)**2*N**2
    k_perpi = np.sqrt(pi)*(w_ce/c)*(w_hat**3/(1-w_hat**2))*(zeta**3*np.exp(-zeta**2)/(1-N))/D
    
    print(f"{w:.3e}")


    fig, ax = plt.subplots()
    
    ax.plot(ryain, N)

    fig.show()

main()

