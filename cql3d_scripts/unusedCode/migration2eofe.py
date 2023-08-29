from ifigure.mto.py_connection import PyConnection

# add connection object to eofe7.mit.edu
if not proj.setting.connection1.has_child('eofe7'):
    child = PyConnection()
    imodel = proj.setting.connection1.add_child('eofe7', child)
    child.setvar('server', 'eofe7.mit.edu')

# point the connection to eofe7 as genray_cql3 host
p = model
if p.name != 'genray_cql3d_loki': p = p.get_parent()
if p.name != 'genray_cql3d_loki': p = p.get_parent()
p.param.setvar('host', '=proj.setting.connection1.eofe7')

# adjust cqlinput
# the first namelist segment should be setup0 not fsetup
# lbdry(1) should be 'conserv'
from collections import OrderedDict
from ifigure.mto.py_contents import Namelist
cqlinput = model.cqlinput
d = cqlinput._var0
d2 = Namelist([('setup0', v) if k == 'fsetup' else (k, v) for k, v in d.items()])
cqlinput._var0 = d2
cqlinput.set_contents("setup","lbdry(1)", ["'conserv'"])
cqlinput.call_method('onUpdateFile')