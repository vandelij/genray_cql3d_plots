"""
    set oh electric field
    Run()   : use param
    Run(elecfield)
"""
#     **  Template for a new script  **

#   Following variabs/functions can be used
#    obj : script object
#    top. proj : = obj.get_root_parent()
#    wdir : proj.getvar('wdir')
#    model: target model to work on
#    param : model param
#    app : proj.app (ifigure application)
#    exit() : exit from script 
#    stop() : exit from script due to error
#  
#    args : parameter arguments passed to Run method (default = ())
#    kwagrs : keyward arguments passed to Run method (default = {})
    
import numpy as np
import ifigure.interactive as plt

if len(args) == 0:
    v = model.param.eval('elecfld')
else:
    v = args[0]
    model.param.set('elecfld', v)
model.cqlinput.set_contents("setup","elecfld(0)", [v])
model.cqlinput.set_contents("setup","elecfld(1)", [v])
model.cqlinput.call_method('onUpdateFile')

