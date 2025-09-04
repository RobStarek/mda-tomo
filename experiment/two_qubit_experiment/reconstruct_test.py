import numpy as np
import pandas as pd
import KetSugar as ks #my own syntatic sugar for numpy-based quantum mechanics
import MaxLik as ml #my own maxlik reconstruction library
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_data(fnIN, CH):
    if ".json" in fnIN:
        DAT = pd.read_json(fnIN, typ="series", orient="records")
        DAT["Data"] = np.array(DAT["Data"])
        Data = DAT["Data"][:, CH]
        return Data
    else:
        raise Exception('Type not supported')


#Prepare projection definition for Maxlik
Q1 = [ks.LO, ks.HI, ks.HLO, ks.HHI, ks.CLO, ks.CHI]
Q2 = [ks.LO, ks.HI, ks.HLO, ks.HHI, ks.CLO, ks.CHI]
ORDER = [Q1, Q2]
RPV = ml.MakeRPV(ORDER, False)

#Load data 
fn = "CaseEv4test.json"
datref = load_data(fn, 14)
#reshape into individual tomograms
#we have nx36 measurements in n tomograms
length = datref.size
tomogramsref = datref.reshape((length//36, 36))
reconstructionsref = [ml.Reconstruct(tomogram, RPV, 1000, 1e-9) \
    for tomogram in tomogramsref]
reconstructionsref = np.array(reconstructionsref)

fn2 = "CaseErefv3.json"
datref2 = load_data(fn2, 14)
#reshape into individual tomograms
#we have nx36 measurements in n tomograms
length = datref2.size
tomogramsref2 = datref2.reshape((length//36, 36))
reconstructionsref2 = [ml.Reconstruct(tomogram, RPV, 1000, 1e-9) \
    for tomogram in tomogramsref2]
reconstructionsref2 = np.array(reconstructionsref2)


def get_xy(rho):
    x = np.arccos((rho[0,0]+rho[1,1])**0.5)
    y = np.arccos((rho[0,0]+rho[2,2])**0.5)
    return x, y

deg = np.pi/180.
xyexp = np.array([get_xy(rho) for rho in reconstructionsref])
xyexp2 = np.array([get_xy(rho) for rho in reconstructionsref2])
Pexp = np.array([ks.Purity(rho) for rho in reconstructionsref])
Pexp2 = np.array([ks.Purity(rho) for rho in reconstructionsref2])

xs2 = np.array(sorted(list(np.arange(45,55,0.5)*deg) + list(np.arange(55,90,2)*deg)))
xs2[0] = xs2[0]+1e-4
#test version - shorter
xs = np.arange(55,90,2)*deg

ys = 0.5*np.arcsin(1/np.tan(xs))
ys2 = 0.5*np.arcsin(1/np.tan(xs2))

plt.plot(xs2/deg, xyexp2[:,0]/deg, ".", label="nominal")
plt.plot(xs/deg, xyexp[:,0]/deg, ".", label="calibrated")
plt.plot(xs2/deg,xs2/deg,"--",c='k')
plt.legend()
#points numbering
plt.xlabel('$x$ nominal [deg]')
plt.ylabel('$x$ experimental [deg]')
plt.show()

plt.plot(ys2/deg, xyexp2[:,1]/deg, ".", label="nominal")
plt.plot(ys/deg, xyexp[:,1]/deg, ".", label="calibrated")
plt.plot(ys2/deg,ys2/deg,"--",c='k')
plt.legend()
#points numbering
plt.xlabel('$y$ nominal [deg]')
plt.ylabel('$y$ experimental [deg]')
plt.show()