'''
set zeff in calinput
   Run()
   Run(value)
   Run(value, cqlinput)

if no argument is given it uses the value in param

if zeff is a singe number it is assume to be flat profile
   zeffin(0) is set
   zeffin(1) is set
   zeffin is deleted
if zeff is not a single number, it is taken as profile.
profile is automatically interpolated to fit njene
   iprozeff is set to 'spline'
   zeffin(0) is deleted
   zeffin(1) is deleted
   zeffin    is set
'''
from ifigure.utils.cbook import isiterable
import numpy as np

#if not model.param.hasvar('zeff'):
#    model.param.setvar('zeff', 2.0)
value = model.param.eval('zeff') if len(args) < 1 else args[0]
cqlinput = model.cqlinput if len(args) < 2 else args[1]
model.param.set('zeff', value)

from ifigure.utils.arraykey_dict import clean_key
setup = cqlinput.get_contents("setup")
if not isiterable(value):
    cqlinput.set_contents("setup","iprozeff", ["'parabola'",])
    if 'zeffin' in setup: setup.pop('zeffin')
    cqlinput.set_contents("setup","zeffin(0)", [value])
    cqlinput.set_contents("setup","zeffin(1)", [value])
else:
    cqlinput.set_contents("setup","iprozeff", ["'spline'",])
    clean_key(setup, 'zeffin')
    valuex = np.linspace(0, 1, len(value))
    njene = cqlinput.get_contents("setup","njene")[0]
    value = np.interp(np.linspace(0, 1, njene), valuex, value)
    cqlinput.set_contents("setup","zeffin", list(value))

cqlinput.call_method('onUpdateFile')