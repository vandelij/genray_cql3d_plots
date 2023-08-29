import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
from numpy.linalg import norm
from mpl_toolkits import mplot3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.cm as cm
import matplotlib


#returns interpolated versions of the magnetic fields
def test():
    angles2 = [101.81, 133.8]
    angles64 = [80.6, 155.92]

    thet1s = np.linspace(angles2[0], angles64[0], 32)
    thet2s = np.linspace(angles2[1], angles64[1], 32)

    print(str(np.round(thet1s,3).tolist())[1:-1].replace(',',''))
    print(str(np.round(thet2s,3).tolist())[1:-1].replace(',',''))

ans(test())



