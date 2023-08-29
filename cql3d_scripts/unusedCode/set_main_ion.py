'''
This script edit 

  fmass(2)    = 
  kspeci(1,2) =  'd'
  bnumb(2)    = 1.0

Run()
Run(name)
Run(name, cqlinput)

name = 'H' or 'D' or 'He'
'''
if not model.param.hasvar('main_ion'):
   model.param.setvar('main_ion', 'D')
name = model.param.eval('main_ion') if len(args) < 1 else args[0]
model.param.set('main_ion', name)
cqlinput = model.cqlinput if len(args) < 2 else args[1]

import imp
mass = imp.load_source('python_lib', '/home/shiraiwa/python_lib/python_lib/mass.py')

if name  == 'H':
  charge = 1.
  dmas   = mass.mass(name = 'proton', unit = 'g')
elif name  == 'D':
  charge = 1.
  dmas   = mass.mass(name = 'deutron', unit = 'g')
elif name  == 'T':
  charge = 1.
  dmas   = mass.mass(name = 'tritron', unit = 'g')
elif name  == 'He':
  charge = 2.
  dmas   = mass.mass(name = 'alpha', unit = 'g')
else:
  exit()

d = cqlinput.get_contents("setup")
if not ("kspeci(1,2)" in d and "fmass(2)" in d and "bnumb(2)" in d):
    #unspported format
    exit()
d['bnumb(2)'] = [charge]
d['fmass(2)'] = [dmas]
d['kspeci(1,2)'] = [name.__repr__()]

cqlinput.call_method('onUpdateFile')
