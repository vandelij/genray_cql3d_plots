import os, sys, numpy as np
#from scipy.io import netcdf_file 
import netCDF4
import matplotlib.pyplot as plt
#from scipy.ndimage.filters import uniform_filter1d
import argparse, textwrap

# get parent directory name only of directory containing the subdirecrtory that contains this script
currentdir = os.path.dirname(os.path.realpath(__file__))
print("\ncurrent directory: ", currentdir)

parser = argparse.ArgumentParser(description='Allows commandline modification of the defualt file source and output directories;\n\
    Ensure copy of eqdsk is in the same input directory provided for the .nc files', formatter_class=argparse.RawTextHelpFormatter, \
    add_help=False)

parser.add_argument('-input', '-i', '-inDir', metavar='<dir_path>', type=str, \
    default='/nobackup1/jamalj/nt_arc/run7_zeta_iter1b__small_pedestal_n_17/out', help=textwrap.dedent('''\nthe absolute path of the directory containing the eqdsk, cql3d.nc, and cql3d_krf001.nc files needed;\ndefaults to the directory path of this script\n\n'''))

#parser.add_argument('-eqdsk','-gfile', metavar='<file_name>', type=str, \
#    default='gNARC_k14d05R455a120', help=textwrap.dedent('''\nthe name of the equilibrium file used for the trinity run;\ndefaults to the recent nominal case: gNARC_k14d05R455a120\n\n'''))

parser.add_argument('--ck','--check_keys', action='store_true', \
    default=False, help=textwrap.dedent('''prints the top level dictionary keys extracted from cql3d.nc and cql3d_krf001.nc files;\ndefaults to doing nothing\n\n'''))

# obtain dictionary-like parser object and assign choice or default values
args=parser.parse_args()

input_path=args.input
#eqdsk_name=args.eqdsk
check_keys=args.ck

# get path to directory:  /nobackup1c/users/jamalj/nt_arc
parentdir = os.path.dirname(currentdir)

# open .nc file and get data reference object

cql3d_nc_path = os.path.join(input_path,'cql3d.nc')
cql3d_nc_data = netCDF4.Dataset(cql3d_nc_path,'r')

cqlkrf_nc_path = os.path.join(input_path,'cql3d_krf001.nc')
cqlkrf_nc_data = netCDF4.Dataset(cqlkrf_nc_path,'r')

# copy data from reference object
rho_tor=np.copy(cql3d_nc_data.variables['rya'][:])
dV = np.copy(cql3d_nc_data.variables['dvol'][:]) # cm^3
power_dens_absorbed = np.copy(cql3d_nc_data.variables['powrft'][-1,:]) # W/cm^-3
cumulative_current_driven = np.copy(cql3d_nc_data.variables['ccurtor'][-1,:])
q_safety = np.copy(cql3d_nc_data.variables['qsafety'][:])
rf_power = np.copy(cql3d_nc_data.variables['powers_int'][-1,0,4]) # (36 x 1 x 13) ndarray

print("\nrf power: ", rf_power * 1e-6, ' [MW]')
#print("length of qsafety: ",len(q_safety))
#print("length of rho_tor: ",len(rho_tor))
#print("length of dV: ",len(dV))


if check_keys:

    print("\nkeys in cql3d_krf001.nc")
    cql3d_krf_keys_str=[]
    [cql3d_krf_keys_str.append(key) for key in cqlkrf_nc_data.variables.keys()]
    cql3d_krf_keys_str=np.sort(cql3d_krf_keys_str)
    [print(key) for key in cql3d_krf_keys_str]
    cqlkrf_nc_data.close()

    print("\nkeys in cql3d.nc")
    cql3d_keys_str=[]
    [cql3d_keys_str.append(key) for key in cql3d_nc_data.variables.keys()]
    cql3d_keys_str=np.sort(cql3d_keys_str)
    [print(key) for key in cql3d_keys_str]


# ID the rho_tor for that q=2
def findNearestIndex(value, array):
    idx = (np.abs(array - value)).argmin()

    return idx

q2_idx=findNearestIndex(2,q_safety)
print("indice: ", q2_idx)
print("rho_tor at q=2 surface: ", rho_tor[q2_idx])
rho_at_q2 = rho_tor[q2_idx]


sz=32
tcks=sz-6
plt.style.use('seaborn-darkgrid')
plt.rc('axes',labelsize=sz, titlesize=sz, grid=True)
plt.rc('xtick', labelsize=tcks)
plt.rc('ytick', labelsize=tcks)
plt.rc('figure',titlesize=sz)
plt.rc('lines', linewidth=6)
plt.rcParams['grid.linewidth'] = 3

def find_q2():
    fig, ax = plt.subplots(figsize=(11,10))
    ax.plot(rho_tor,q_safety)
    ax.set_xlabel(r'$\rho _{tor}$')
    ax.set_ylabel("q")
    plt.savefig("my_q2_location_check.png", dpi=300)
    plt.show()

def cumulative_driven_current():

    curr_drive_eff=np.max(np.abs(cumulative_current_driven)) / rf_power
    print("\ntotal current drive efficiency: ", '{:.3f}'.format(curr_drive_eff)+ ' [A/W]')

    fig, ax = plt.subplots(figsize=(11.4,10))
    ax.plot(rho_tor, cumulative_current_driven*1e-6)
    ax.axvline(rho_at_q2, linestyle='--', color="tab:red", linewidth=4)
    ax.set_xlabel(r'$\rho _{tor}$')
    ax.set_ylabel(r'Increase in I$_p$ [MA]')
    ax.set_title("Cumulative RF Driven Current at Flattop\nwith drive efficiency: " + \
        '{:.3f}'.format(curr_drive_eff)+ ' [A/W]')
    plt.savefig("cumulative_driven_current.png", dpi=300)
    plt.show()

def power_density_dep():
    # note: W/cm^3 = MW/m^3
    fig, ax = plt.subplots(figsize=(11.4,10))
    ax.plot(rho_tor, power_dens_absorbed)
    ax.axvline(rho_at_q2, linestyle='--', color="tab:red", linewidth=4)
    ax.set_xlabel(r'$\rho _{tor}$')
    ax.set_ylabel(r'P [MW/m$^3$]')
    ax.set_title('Absorbed Power Density')
    #ax.minorticks_on()
    #ax.grid(True, which='both', axis='both')

#    secax = ax.secondary_xaxis('top', functions=(qSafety, qInverseFunc))
#    secax.set_xlabel('q')

    plt.savefig("power_density_dep.png", dpi=300)
    plt.show()
    print("\npeak power density: ", str(np.max(power_dens_absorbed)) + " [MW/m^3]")

# CONFIRM TIME INTERVAL DETAILS WITH SAM, THEN COMPLETE THE CALCULATION FOR DELTA_Te vs rho;
# plot 1x2 along side the original Te profile to see change
def total_power_dep():

    fig, ax = plt.subplots(figsize=(11.4, 10))
    power_dep_profile = power_dens_absorbed * dV
    ax.plot(rho_tor,power_dep_profile*1e-6)
    ax.axvline(rho_at_q2, linestyle='--', color="tab:red", linewidth=4)
    ax.set_xlabel(r'$\rho _{tor}$')
    ax.set_ylabel(r'Total Absorbed Power [MW]')
    plt.savefig("total_power_dep.png", dpi=300)
    plt.show()

cql3d_nc_data.close()

cumulative_driven_current()
power_density_dep()
total_power_dep()
find_q2()