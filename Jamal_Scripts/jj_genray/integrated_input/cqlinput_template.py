# Stable cqlinput file settings and profile feed template
# Jamal Johnson
# 12/1/2022 
# Fixed inputs sourced from Sam Frank "Simplified CQL3D Namelist" 10/2019

cqlinput='!------------------------------------------------------------------------------\n\
!------------------------------------------------------------------------------\n\
!A simplified version of the CQL3D namelist that has options which are not\n\
!used such as those for mirrors or some of the more exotic collision\n\
!and loss term stuff removed. This was originally set up for the rf \n\
!condensation simulations performed on ITER simulation data. Important options\n\
!have comments describing their function. - S. Frank\n\
!------------------------------------------------------------------------------\n\
\n\
 &setup0\n\
 ibox =  \'cori\'\n\
 iuser = \'sfrnk\'\n\
 ioutput =  6\n\
 lrz =  60  !Number of Flux Surfaces in Simulation\n\
 noplots =  \'enabled\'\n\
 mnemonic = \'cql3d\'\n\
 nlwritf =  \'ncdfdist\'\n\
 nlrestrt = \'disabled\' !enabled allows restart from saved distribution function\n\
 &end\n\
 \n\
 &setup\n\
!------------------------------------------------------------------------------\n\
!Species Setup\n\
!------------------------------------------------------------------------------\n\
 ngen =  1 !number of \"general\" i.e. FP\'ed species\n\
 nmax =  3 !number of maxwellian background species \n\
!Species Labels (1:2,k)\n\
 kspeci(1,1) =  \'e\'\n\
 kspeci(2,1) =  \'general\'\n\
 kspeci(1,2) =  \'D\'\n\
 kspeci(2,2) =  \'maxwell\'\n\
 kspeci(1,3) =  \'T\'\n\
 kspeci(2,3) =  \'maxwell\'\n\
 kspeci(1,4) =  \'e\'\n\
 kspeci(2,4) =  \'maxwell\'\n\
!Charge of species (k)\n\
 bnumb(1) =  -1.0\n\
 bnumb(2) =  1.0\n\
 bnumb(3) =  1.0\n\
 bnumb(4) =  -1.0\n\
!Mass of species (k)\n\
 fmass(1) =  9.1095e-28\n\
 fmass(2) =  3.3435834581e-24\n\
 fmass(3) =  5.00826765e-24\n\
 fmass(4) =  9.1095e-28\n\
!------------------------------------------------------------------------------\n\
!Simulation Numerical Setup\n\
!------------------------------------------------------------------------------\n\
!Timestep Settings\n\
 dtr =  0.0000001 !Timestep\n\
 nstop =  35 !Number of steps\n\
!Variable Timestepping Settings\n\
 dtr1 =  0.000001 0.00001 0.0001 0.001,0.01 \n\
 nondtr1 =  5 10 15 20 25 \n\
!Mesh Settings\n\
 enorm =  1500  !keV\n\
 jx =  300 	!Parallel velocity mesh points\n\
 xfac =  0.5    !x mesh spacing parameters \n\
 xlwr =  0.085  !If using geometric mesh with user specified packing\n\
 xmdl =  0.25   !these options are used\n\
 xpctlwr =  0.1\n\
 xpctmdl =  0.4\n\
 iy =  200      !Perpendicular velocity mesh points\n\
 tfac =  1.0	!y/theta mesh spacing parameters\n\
 numby =  30    !Points packed around p/t boundary must be << iy\n\
 ylower =  1.22\n\
 yupper =  1.275\n\
 meshy =  \'fixed_y\' !I don\'t know what this does but it has to do with y mesh\n\
 lz =  60	!Field line mesh points\n\
 tfacz =  1.0	!Field line mesh spacing parameter\n\
!Differencing and Solution Method\n\
!Don\'t touch these unless you know what you are doing\n\
 soln_method=\'direct\'\n\
 implct =  \'enabled\'\n\
 chang =  \'disabled\'\n\
 ineg =  \'enabled\'\n\
 manymat = \'disabled\'\n\
 lbdry(1) = \'consscale\'\n\
!------------------------------------------------------------------------------\n\
!Simulation Physics/Profiles Setup\n\
!------------------------------------------------------------------------------\n\
!Physics Options\n\
 bootst =  \'disabled\' !Bootstrap Current Calculation On/Off\n\
 colmodl =  0 !Collision Model\n\
 gamaset =  0.0 !If .ne. to zero sets coulumb log to gamaset else computed\n\
 izeff =  \'backgrnd\' !\n\
 lossmode(1) = \'simplban\'\n\
 qsineut =  \'disabled\' !Forced quasi-neutrality\n\
 relativ =  \'enabled\' !Reltavistic calculation\n\
 syncrad =  \'disabled\' !Synchrotron Radiation\n\
!\"Advanced\" Physics Options (Usually do not need to be touched)\n\
 nonel =  10000\n\
 noffel =  10000\n\
 ncoef =  1\n\
 mx =  3 \n\
!Flux Surfaces\n\
 radcoord =  \"sqtorflx\" !Radial normalization\n\
 rovera = 1.0E-06\n\
 rzset = \'enabled\'	!Leave enabled set flux surface locations w/ rya(1)\n\
 rya(1) =\n\
 5.000000000E-02 6.525423729E-02 8.050847458E-02 9.576271186E-02 1.110169492E-01\n\
 1.262711864E-01 1.415254237E-01 1.567796610E-01 1.720338983E-01 1.872881356E-01\n\
 2.025423729E-01 2.177966102E-01 2.330508475E-01 2.483050847E-01 2.635593220E-01\n\
 2.788135593E-01 2.940677966E-01 3.093220339E-01 3.245762712E-01 3.398305085E-01\n\
 3.550847458E-01 3.703389831E-01 3.855932203E-01 4.008474576E-01 4.161016949E-01\n\
 4.313559322E-01 4.466101695E-01 4.618644068E-01 4.771186441E-01 4.923728814E-01\n\
 5.076271186E-01 5.228813559E-01 5.381355932E-01 5.533898305E-01 5.686440678E-01\n\
 5.838983051E-01 5.991525424E-01 6.144067797E-01 6.296610169E-01 6.449152542E-01\n\
 6.601694915E-01 6.754237288E-01 6.906779661E-01 7.059322034E-01 7.211864407E-01\n\
 7.364406780E-01 7.516949153E-01 7.669491525E-01 7.822033898E-01 7.974576271E-01\n\
 8.127118644E-01 8.279661017E-01 8.432203390E-01 8.584745763E-01 8.737288136E-01\n\
 8.889830508E-01 9.042372881E-01 9.194915254E-01 9.347457627E-01 9.500000000E-01\n\
!Profiles Setup\n\
 njene =  101\n\
 psimodel = \'spline\'\n\
 iproelec = \'spline\'\n\
 iprone = \'spline\'\n\
 iprote = \'spline\'\n\
 iproti = \'spline\'\n\
 iprozeff = \'spline\'\n\
 tescal =  1.0\n\
 enescal = 1.0e-6\n\
 efswtch=\'method1\'\n\
 efiter =\'disabled\'\n\
 ryain =\n\
 0.0e+00 1.0e-02 2.0e-02 3.0e-02 4.0e-02\n\
 5.0e-02 6.0e-02 7.0e-02 8.0e-02 9.0e-02\n\
 1.0e-01 1.1e-01 1.2e-01 1.3e-01 1.4e-01\n\
 1.5e-01 1.6e-01 1.7e-01 1.8e-01 1.9e-01\n\
 2.0e-01 2.1e-01 2.2e-01 2.3e-01 2.4e-01\n\
 2.5e-01 2.6e-01 2.7e-01 2.8e-01 2.9e-01\n\
 3.0e-01 3.1e-01 3.2e-01 3.3e-01 3.4e-01\n\
 3.5e-01 3.6e-01 3.7e-01 3.8e-01 3.9e-01\n\
 4.0e-01 4.1e-01 4.2e-01 4.3e-01 4.4e-01\n\
 4.5e-01 4.6e-01 4.7e-01 4.8e-01 4.9e-01\n\
 5.0e-01 5.1e-01 5.2e-01 5.3e-01 5.4e-01\n\
 5.5e-01 5.6e-01 5.7e-01 5.8e-01 5.9e-01\n\
 6.0e-01 6.1e-01 6.2e-01 6.3e-01 6.4e-01\n\
 6.5e-01 6.6e-01 6.7e-01 6.8e-01 6.9e-01\n\
 7.0e-01 7.1e-01 7.2e-01 7.3e-01 7.4e-01\n\
 7.5e-01 7.6e-01 7.7e-01 7.8e-01 7.9e-01\n\
 8.0e-01 8.1e-01 8.2e-01 8.3e-01 8.4e-01\n\
 8.5e-01 8.6e-01 8.7e-01 8.8e-01 8.9e-01\n\
 9.0e-01 9.1e-01 9.2e-01 9.3e-01 9.4e-01\n\
 9.5e-01 9.6e-01 9.7e-01 9.8e-01 9.9e-01\n\
 1.0e+00\n\
 enein(1,1) =\n\
{enein_e}\n\
 enein(1,2) =\n\
{enein_DT}\n\
 enein(1,3) =\n\
{enein_DT}\n\
 enein(1,4) =\n\
{enein_e}\n\
 tein =\n\
{tein}\n\
 tiin =\n\
{tiin}\n\
 zeffin = 101*1.50\n\
 elecin = 0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
 	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00    0.000000E+00    0.000000E+00    0.000000E+00\n\
	  0.000000E+00\n\
!------------------------------------------------------------------------------\n\
!Simulation I/O\n\
!------------------------------------------------------------------------------\n\
!General Plotting and I/O\n\
 contrmin = 1e-12\n\
 iactst = \'enabled\'\n\
 idskf =  \'disabled\'\n\
 idskrf = \'disabled\'\n\
 nchec =  1\n\
 ncont =  20\n\
 nplot =  35\n\
 nplt3d = 35\n\
 pltvs =  \'rho\'\n\
 veclnth =  1.5\n\
 nrskip =  1\n\
 irzplt(1) =  2\n\
 irzplt(2) =  4\n\
 irzplt(3) =  15\n\
 irzplt(4) =  30\n\
 irzplt(5) =  40\n\
 irzplt(6) =  45\n\
 irzplt(7) =  50\n\
 irzplt(8) =  60\n\
 plt3d    =  \'enabled\'\n\
 pltd     =  \'enabled\'\n\
 pltdn    =  \'disabled\'\n\
 pltend   =  \'enabled\'\n\
 pltfvs   =  \'enabled\'\n\
 pltinput =  \'enabled\'\n\
 pltmag   =  1.0\n\
 pltsig   =  \'disabled\'\n\
 pltpowe  =  \'last\'\n\
 pltprpp  =  \'enabled\'\n\
 pltrst   =  \'disabled\'\n\
 plturfb  =  \'enabled\'\n\
 pltstrm  =  \'disabled\'\n\
 pltvecal =  \'disabled\'\n\
 pltvecc  =  \'disabled\'\n\
 pltvece  =  \'disabled\'\n\
 pltvecrf =  \'disabled\'\n\
 pltvflu  =  \'disabled\'\n\
 netcdfnm =  \'enabled\'\n\
 partner =  \'disabled\'\n\
!SXR Diagnostic\n\
 softxry =  \'disabled\'\n\
 nv =  32\n\
 enmax =  205 !Max Energy\n\
 enmin =  25  !Min Energy\n\
 fds =  0.2   !Step size along viewing chord\n\
 nen =  19    !Energies at which spectral values are calculated on each chord\n\
!SXR Detector location data\n\
 x_sxr =  187.0\n\
 z_sxr =  0.0\n\
 thet1 =  75.5 76.4 77.4 78.3 79.2 80.2 81.1 82.0 83.0 83.9 84.9 85.8 86.7\n\
     87.7 88.6 89.5 90.5 91.4 92.4 93.3 94.2 95.2 96.1 97.0 98.0 98.9 99.9\n\
     100.8 101.7 102.7 103.6 104.5\n\
 thet2 =  177.2 177.2 177.2 177.2 177.2 177.2 177.2 177.2 177.2 177.2\n\
     177.2 177.2 177.2 177.2 177.2 177.2 177.2 177.2 177.2 177.2 177.2\n\
     177.2 177.2 177.2 177.2 177.2 177.2 177.2 177.2 177.2 177.2 177.2\n\
 thetd =  0.0\n\
 &end\n\
\n\
!------------------------------------------------------------------------------\n\
!Transport/Diffusion Namelist\n\
!------------------------------------------------------------------------------\n\
 &trsetup\n\
 adimeth =  \'disabled\'\n\
 difusr =  400.0\n\
 difus_rshape(1) =  1.0 0.0 0.0 0.0 0.0 0.0 0.0\n\
 difus_vshape(1) =  1.0 0.0 0.0 3.0\n\
 nontran =  11\n\
 pinch =  \'case3n\'\n\
 relaxden =  0.001\n\
 relaxtsp =  \'disabled\'\n\
 transp =  \'disabled\'\n\
 advectr =  1.0\n\
 &end\n\
\n\
!------------------------------------------------------------------------------\n\
!Particle Source Setup Namelist\n\
!------------------------------------------------------------------------------\n\
 &sousetup\n\
 noffso(1,1) =  10000\n\
 noffso(1,2) =  10000\n\
 nonso(1,1) =  0\n\
 nonso(1,2) =  0\n\
 nso =  0\n\
 nsou =  1\n\
 pltso =  \'enabled\'\n\
 soucoord =  \'disabled\'\n\
 komodel =  \'mr\'\n\
 flemodel =  \'pol\'\n\
 knockon =  \'disabled\'\n\
 jfl =  350\n\
 nkorfn =  1\n\
 nonko =  0\n\
 noffko =  10000\n\
 soffvte =  -3.0\n\
 soffpr =  0.5\n\
 isoucof =  1\n\
 faccof =  0.5\n\
 xlfac =  -0.5\n\
 xllwr =  0.0005\n\
 xlmdl =  0.05\n\
 xlpctlwr =  0.15\n\
 xlpctmdl =  0.6\n\
 &end\n\
\n\
!------------------------------------------------------------------------------\n\
!Magnetic Equilibrium Setup Namelist\n\
!------------------------------------------------------------------------------\n\
 &eqsetup\n\
 atol =  1e-08\n\
 bsign =  1.0\n\
! bsign =  -1.0\n\
 ellptcty =  0.0\n\
 eqmod =  \'enabled\'\n\
 eqpower =  2\n\
 eqsource =  \'eqdsk\'\n\
 eqdskin =  \'{eqdsk}\'\n\
 fpsimodl =  \'constant\'\n\
 methflag =  10\n\
 nconteq =  \'psigrid\'\n\
 nconteqn = 0\n\
 rbox =  92.0\n\
 rboxdst =  120.0\n\
 rmag =  166.0\n\
 rtol =  1e-08\n\
 zbox =  92.0\n\
 &end\n\
\n\
!------------------------------------------------------------------------------\n\
!RF Source Setup Namelist\n\
!------------------------------------------------------------------------------\n\
 &rfsetup\n\
 lh =  \'enabled\'\n\
 ech =  \'disabled\'\n\
 fw =  \'disabled\'\n\
 iurfl =  \'disabled\'\n\
 iurfcoll =  \'enabled\'\n\
 nbssltbl =  2000\n\
 nharms =  1\n\
 nharm1 =  0\n\
 nrfitr1 =  1000\n\
 nrfitr2 =  0\n\
 nrfitr3 =  1\n\
 nrfpwr =  0\n\
 nrfstep1(1) =  1000\n\
 nrfstep1(2) =  1000\n\
 nrfstep2 =  0\n\
 noffrf(1) =  100000\n\
 nonrf(1) =  0\n\
 nrf =  0\n\
 pwrscale(1) =  1.0\n\
 pwrscale(2) =  1.0\n\
 rfread =  \'netcdf\'\n\
 rffile(1) =  \'genray.nc\'\n\
 scaleurf =  \'enabled\'\n\
 urfdmp =  \'secondd\'\n\
 urfmod =  \'enabled\'\n\
 &end\n\
\n\
!------------------------------------------------------------------------------\n\
!Neutral Beam Setup Namelist\n\
!------------------------------------------------------------------------------\n\
 &frsetup\n\
 &end\n\
'
cql3d_pbs='#!/bin/sh\n\
#SBATCH -N 2\n\
#SBATCH -n 60\n\
#SBATCH -t 01:30:00\n\
#SBATCH -p sched_mit_psfc\n\
#SBATCH -J CQL3D\n\
\n\
cd $SLURM_SUBMIT_DIR\n\
sbcast /home/frank/dev_execs/CQL3D/xcql3d_mpi /tmp/${USER}_cql\n\
mpirun /tmp/${USER}_cql\n\
'