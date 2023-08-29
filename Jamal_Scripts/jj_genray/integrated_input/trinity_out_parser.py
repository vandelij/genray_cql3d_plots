# Trinity-to-Genray/CQL3D automated integrator
# Jamal Johnson
# 12/1/2022

import os, sys, argparse, textwrap, numpy as np
import torch, subprocess as sp
from omfit_classes import omfit_eqdsk

# import custom modules
import coordinate_tools as ct
import cqlinput_template as cTmp
import genray_in_template as genTmp

# DO NOT USE THE RUN GENRAY OR RUN CQL3D OPTIONS FOR SUBMITTING JOBS BUT INSTEAD SUBMIT THEM FROM THE TERMINAL

# -------------------------------------------------------------------------------------------------------------------------------
#                                                     Notes
#--------------------------------------------------------------------------------------------------------------------------------
# Required: input trinity .npy file must be named "log_trinity.npy", and the eqdsk used in trinity run;
# edge model data supplimented with BALOO adjusted, CHEASE points for n,Ti,Te profiles;
# the density gradients from trinity were calculated at midpoints of the 5 r/a and only have 4 points while other arrays have 5;
# aLn gradients are all positive despite n,T vs r/a plots clearly showing negative gradients

#------------------------------------------------------------------------------------------------------
# create switches that will control major output class objects with command line control over defualts
#------------------------------------------------------------------------------------------------------
# get the path of the directory that this script was placed into; by defualt only 1 arg can be given for -i, -o, and -eqdsk
currentdir = os.path.dirname(os.path.realpath(__file__))
print("\ncurrent directory: ", currentdir)

parser = argparse.ArgumentParser(description='Allows commandline modification of the defualt file source and output directories', formatter_class=argparse.RawTextHelpFormatter, \
    add_help=False)

parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help=textwrap.dedent('''Show this help message and exit.\n\n'''))

parser.add_argument('--wi', '--write_inputs', action='store_true', \
    default=False, help=textwrap.dedent('''allows for genray.in, cqlinput, .pbs, and .sh files to be written;\ndefaults to doing nothing\n\n'''))

parser.add_argument('-input', '-i', '-inDir', metavar='<dir_path>', type=str, \
    default=currentdir, help=textwrap.dedent('''\nthe absolute path of the input directory containing trinity\'s .npy file;\ndefaults to the directory path of this script\n\n'''))

parser.add_argument('-output', '-o', '-outDir', metavar='<dir_path>', type=str, \
    default=currentdir, help=textwrap.dedent('''\nthe absolute path of the output directory where genray.in, cqlinput, and .pbs files will be written;\ndefaults to the directory path of this script\n\n'''))

parser.add_argument('-eqdsk','-gfile', metavar='<file_name>', type=str, \
    default='gNARC_k14d05R455a120', help=textwrap.dedent('''\nthe name of the equilibrium file used for the trinity run;\ndefaults to the recent nominal case: gNARC_k14d05R455a120\n\n'''))

parser.add_argument('--ce','--core_edge', action='store_true', \
    default=False, help=textwrap.dedent('''prints the n, Ti, and Te values given at r/a=0.1 and r/a=0.9 in trnity\'s log file;\ndefaults to doing nothing\n\n'''))

parser.add_argument('--pp','--print_profiles', action='store_true', \
    default=False, help=textwrap.dedent('''\nprints the genray and cql3d input profiles for n, Ti, and Te in rho_tor coordinates spanning 0-1;\ndefaults to doing nothing\n\n'''))

parser.add_argument('--ck','--check_keys', action='store_true', \
    default=False, help=textwrap.dedent('''prints the first level keys and the keys for the \'profiles\' nested dict in the trinity log file;\ndefaults to doing nothing\n\n'''))

#parser.add_argument('--rgen','--run_genray', action='store_true', \
#    default=False, help=textwrap.dedent('''submits the genray .pbs job file to engaging\'s scheduler;\nmust specify -output directory if genr.pbs is not in this script\'s directory\ndefaults to doing nothing\n\n'''))
#
#parser.add_argument('--rcql','--run_cql3d', action='store_true', \
#    default=False, help=textwrap.dedent('''submits the cql3d .pbs job file to engaging\'s scheduler;\nmust specify -output directory if cql3d.pbs is not in this script\'s directory\ndefaults to doing nothing\n\n'''))

# obtain dictionary-like parser object and assign choice or default values
args=parser.parse_args()

source_path = args.input
output_path = args.output
write_input_files=args.wi
eqdsk=args.eqdsk
print("\nReferencing eqdsk file: ", eqdsk)

# quick value check controls
check_core_edge=args.ce
print_profiles=args.pp
check_keys=args.ck

#submit_gen_job=args.rgen
#submit_cql_job=args.rcql

eqdskPath=os.path.join(source_path,eqdsk)

# create equilibrum dict-like obj
geqdsk = omfit_eqdsk.OMFITgeqdsk(eqdskPath)

fileName='log_trinity.npy'
filePath=os.path.join(source_path,fileName)
trinity_data = np.load(filePath, allow_pickle=True).tolist() # remove .tolist() if using .item(0).keys() method on loaded obj

# check r/a at rho_tor=084 in or eqlibrium
# 

if check_keys:
    print("available primary level data keys from trinity output\n")
    print(trinity_data.keys())
    print("\navailable 'profiles' dict keys\n")
    print(trinity_data['profiles'].keys())

# in the trinity output, rho = r/a at gridpoints
profile_data = trinity_data['profiles']
ra_nT = profile_data['rho_axis'] # r/a array where the 5 profile datapoints were calculated; r/a=[0.1, 0.3, 0.5, 0.7, 0.9]

# get density, temperatures, and pressures at last timestep
n     = np.array(trinity_data['n'])[-1]
pi    = np.array(trinity_data['pi'])[-1]
pe    = np.array(trinity_data['pe'])[-1]
a_Ln  = np.array(trinity_data['aLn'])[-1] # (4,)
a_Lpi = np.array(trinity_data['aLpi'])[-1]
a_Lpe = np.array(trinity_data['aLpe'])[-1]
a_LTi = a_Lpi - a_Ln
a_LTe = a_Lpe - a_Ln
Ti = pi/n
Te = pe/n

if check_core_edge:

    # check most significant values present in trinity output data
    print("Values at r/a=0.1\n")
    print("n=", n[0], " 10^20 m^-3")
    print("Te=", Te[0], " keV")
    print("Ti=", Ti[0], " keV")
    print("\nValues at r/a=1.0\n")
    print("n=", n[-1], " 10^20 m^-3")
    print("Te=", Te[-1], " keV")
    print("Ti=", Ti[-1], " keV")

# prepend zeros to all inverse gradient scale lengths
a_Ln=np.insert(a_Ln,0,0)
a_LTi=np.insert(a_LTi,0,0)
a_LTe=np.insert(a_LTe,0,0)

ra_midpoints = [0, 0.2, 0.4, 0.6, 0.8] # prepended with 0
mid_ra = torch.Tensor([ra_midpoints]) # inputs to .Tensor must be in list of list format for Pablo's function

# get values of n,Ti,Te at r/a=0.8, then use as the boundary conditions in pablos script
ra_prof=[0.1, 0.3, 0.5, 0.7, 0.9]
n_08=np.interp(0.8, ra_prof, n)
Ti_08=np.interp(0.8, ra_prof, Ti)
Te_08=np.interp(0.8, ra_prof, Te)

aLn = torch.Tensor([a_Ln.tolist()])
aLTi = torch.Tensor([a_LTi.tolist()])
aLTe = torch.Tensor([a_LTe.tolist()])
n0_to_8 = ct.integrateGradient(mid_ra,aLn,n_08)
Ti0_to_8 = ct.integrateGradient(mid_ra,aLTi,Ti_08)
Te0_to_8 = ct.integrateGradient(mid_ra,aLTe,Te_08)

# only single element tensors can be be converted to python scalars, not the 1D array

# obtain parallel rho_n array as required by genray/cql3d
ra_q2 = ct.rho_to_ra_converter([0.84], geqdsk)
print("\nlocation of q=2 in r/a: ", ra_q2)

# obtain parallel rho_n array as required by genray/cql3d
rhon0_to_8 = ct.ra_to_rho_converter(ra_midpoints, geqdsk)

# convert r/a=0.9 to rhon, then append it along with the n,T values to respective arrays
rhon9=ct.ra_to_rho_converter([0.9],geqdsk)
4
rhon0_to_9 = np.append(rhon0_to_8, rhon9[0])
n0_to_9 = np.append(n0_to_8, n[-1])
Ti0_to_9 = np.append(Ti0_to_8, Ti[-1])
Te0_to_9 = np.append(Te0_to_8, Te[-1])

# get profile values at rho_n=1, extract 1D tensor objects
n1 = ct.edge_values(rhon0_to_9, n0_to_9)
T1=0.5 # enforcing 500 eV edge for Te,sep ~ 473 keV

#check n at LCFS
print("n at LCFS: ", n1)
# append all arrays, then increase resolution for genray/cql3d
rho_full = np.append(rhon0_to_9, 1)
n_full = np.append(n0_to_9, n1)
Ti_full = np.append(Ti0_to_9, T1)
Te_full = np.append(Te0_to_9, T1)

# 101 produces a consistent input rho_tor coordinate spectrum [0,1] with 0.01 steps that are beneficial to code stability
res=101 # 'ndens' in genray

# get input array matching uniform ryain points enforced from 0 to 1
rho_tor_hi_res=np.linspace(0, 1,res)

# obtain n Ti Te arrays at <ndens> uniformly spaced rho_tor locations
n_hi_res=np.interp(rho_tor_hi_res, rho_full, n_full) * 1e20 # m^-3
Ti_hi_res=np.interp(rho_tor_hi_res, rho_full, Ti_full)
Te_hi_res=np.interp(rho_tor_hi_res, rho_full, Te_full)

# assemble dentab and tentab arrays to include 3 species as given in defualt genray.in &species settings); 101 rows
dentab = np.zeros(3*res)
temtab = np.zeros(3*res)

for i in range(res):

    # # convert density back to full order, 10^20 m^-3
    dentab[i*3] = n_hi_res[i]
    dentab[i*3+1] = n_hi_res[i]
    dentab[i*3+2] = n_hi_res[i]
    
    # keV
    temtab[i*3] = Te_hi_res[i]
    temtab[i*3+1] = Ti_hi_res[i]
    temtab[i*3+2] = Ti_hi_res[i]

if print_profiles:
    
    #-------------------------------------------
    # print arrays in an input format for genray.in
    #-------------------------------------------
    print("\ndentab:\n")
    for i in range(res):
        print(' '+'{:.9e}'.format(dentab[3*i]) + ' ' + '{:.9e}'.format(dentab[3*i+1]) + ' ' + '{:.9e}'.format(dentab[3*i+2]))

    print("\ntemtab:\n")
    for i in range(res):
        print(' '+'{:.9e}'.format(temtab[3*i]) + ' ' + '{:.9e}'.format(temtab[3*i+1]) + ' ' + '{:.9e}'.format(temtab[3*i+2]))

    #-------------------------------------------
    # print arrays in an input format for cql3d
    #-------------------------------------------
    print("\nenein(1)(4) electrons:\n")
    for i in range(int(len(n_hi_res)/5)):
        print(' '+'{:.6e}'.format(n_hi_res[5*i]) + ' ' + '{:.6e}'.format(n_hi_res[5*i+1]) + \
            ' ' + '{:.6e}'.format(n_hi_res[5*i+2]) + ' ' + '{:.6e}'.format(n_hi_res[5*i+3]) + \
                ' ' + '{:.6e}'.format(n_hi_res[5*i+4]))
    print(' '+'{:.6e}'.format(n_hi_res[-1]))

    print("\nenein(2)(3) ion split for 50/50 DT:\n")
    for i in range(int(len(n_hi_res)/5)):
        print(' '+'{:.6e}'.format(n_hi_res[5*i]/2) + ' ' + '{:.6e}'.format(n_hi_res[5*i+1]/2) + \
            ' ' + '{:.6e}'.format(n_hi_res[5*i+2]/2) + ' ' + '{:.6e}'.format(n_hi_res[5*i+3]/2) + \
                ' ' + '{:.6e}'.format(n_hi_res[5*i+4]/2))
    print(' ' + '{:.6e}'.format(n_hi_res[-1]/2))

    print("\ntein:\n")
    for i in range(int(len(Te_hi_res)/5)):
        print(' '+'{:.6e}'.format(Te_hi_res[5*i]) + ' ' + '{:.6e}'.format(Te_hi_res[5*i+1]) + \
            ' ' + '{:.6e}'.format(Te_hi_res[5*i+2]) + ' ' + '{:.6e}'.format(Te_hi_res[5*i+3]) + \
                ' ' + '{:.6e}'.format(Te_hi_res[5*i+4]))
    print(' ' + '{:.6e}'.format(Te_hi_res[-1]))            

    print("\ntiin:\n")
    for i in range(int(len(Ti_hi_res)/5)):
        print(' '+'{:.6e}'.format(Ti_hi_res[5*i]) + ' ' + '{:.6e}'.format(Ti_hi_res[5*i+1]) + \
            ' ' + '{:.6e}'.format(Ti_hi_res[5*i+2]) + ' ' + '{:.6e}'.format(Ti_hi_res[5*i+3]) + \
                ' ' + '{:.6e}'.format(Ti_hi_res[5*i+4]))
    print(' ' + '{:.6e}'.format(Ti_hi_res[-1]))

if write_input_files:
    # Setup formatted string arrays for writing to dictionaries for templates
    dentab_str=''
    temtab_str=''
    enein_e_str=''
    enein_DT_str=''
    tein_str=''
    tiin_str=''
    for i in range(res):
        dentab_str += ' '+'{:.9e}'.format(dentab[3*i]) + ' ' + '{:.9e}'.format(dentab[3*i+1]) + ' ' + \
            '{:.9e}'.format(dentab[3*i+2]) + '\n'

        temtab_str += ' '+'{:.9e}'.format(temtab[3*i]) + ' ' + '{:.9e}'.format(temtab[3*i+1]) + ' ' + \
            '{:.9e}'.format(temtab[3*i+2]) + '\n'

        enein_e_str += ' '+'{:.6e}'.format(n_hi_res[i]) + '\n'

        enein_DT_str += ' '+'{:.6e}'.format(n_hi_res[i]/2) + '\n'

        tein_str += ' '+'{:.6e}'.format(Te_hi_res[i]) + '\n'

        tiin_str += ' '+'{:.6e}'.format(Ti_hi_res[i]) + '\n'

    # setup dictionaries for filling templates
    genray_dict = { 'eqdsk' : eqdsk, 'temtab' : temtab_str, 'dentab' : dentab_str}
    cqlinput_dict = { 'eqdsk' : eqdsk, 'enein_e' : enein_e_str, 'enein_DT' : enein_DT_str, 'tein' : tein_str, 'tiin' : tiin_str}

    # import templates
    genray_template=genTmp.genray_in
    cqlinput_template=cTmp.cqlinput
    genray_pbs=genTmp.genray_pbs
    cql3d_pbs=cTmp.cql3d_pbs


    # write new files
    genray_filled=genray_template.format(**genray_dict)
    cql_filled=cqlinput_template.format(**cqlinput_dict)

    genray_pbs_path=os.path.join(output_path,'genr.pbs')
    cql3d_pbs_path=os.path.join(output_path,'cql3d.pbs')

    for i, filename in enumerate(['genray.in', 'cqlinput','genr.pbs','cql3d.pbs', 'run_genray.sh', 'run_cql3d.sh']):
        with open(os.path.join(output_path,filename), 'w') as f:
            
            if i == 0:            
                f.write(genray_filled)
            
            elif i == 1:
                f.write(cql_filled)

            elif i == 2:
                f.write(genray_pbs)

            elif i == 3:
                f.write(cql3d_pbs)
            
            elif i == 4:
                f.write("sbatch "+genray_pbs_path)

            elif i == 5:
                f.write("sbatch "+cql3d_pbs_path)

    # make copy of eqdsk file for the output dir if it's not already there
    if not os.path.isfile(os.path.join(output_path,eqdsk)):

        sp.Popen('cp '+eqdskPath+' '+ output_path, shell=True).wait()
'''
# after genray.in and cqlinput files have been written, first run genray then cql3d 
if submit_gen_job:

    source_file_path=os.path.join(output_path,"run_genray.sh")
    os.system("source "+source_file_path)

if submit_cql_job:

    source_file_path=os.path.join(output_path,"run_cql3d.sh")
    os.system("source "+source_file_path)
'''


