#
# this script convert genray.in and cqlinput that use
# analytic profiles to spline profiles
#
# 
#
d = model.genray_loki.scripts.idens0to1.RunA()
model.cql3d_loki.scripts.set_ne.Run(d['dens'][0], d['rho'])
model.cql3d_loki.scripts.set_te.Run(d['temp'][0], d['rho'])
model.cql3d_loki.scripts.set_ti.Run(d['temp'][1], d['rho'])
model.cql3d_loki.cqlinput.set_contents("setup","iprote", ['"spline"'])
model.cql3d_loki.cqlinput.set_contents("setup","iprone", ['"spline"'])
model.cql3d_loki.cqlinput.set_contents("setup","iproti", ['"spline"'])
ans(d)