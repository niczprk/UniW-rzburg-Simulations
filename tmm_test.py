# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:05:45 2025

@author: jod52fr
"""
import sys
sys.path.append(r"C:\Users\s531596\Documents\GitHub\tmm\\")
from tmm import Medium, Structure, PeriodicStructure,to_energy
from official_index_list import sio2_sputter as sio2_r
from official_index_list import tio2_sputter as tio2_r 
from official_index_list import air as air_r
from refractives import Const
import numpy as np
import matplotlib.pyplot as plt



#%% Device Setup

cwl_bottom = 0.844 #Wavelength in um
ce_bottom = to_energy(cwl_bottom)
GaAs_ref = Const([3.6,0])

d_sio2 = cwl_bottom/(4*sio2_r.get_index(ce_bottom)[0]) #quarter wave layers thickness
d_tio2 = cwl_bottom/(4*tio2_r.get_index(ce_bottom)[0]) #quarter wave layers thickness

#Definition of Media
gaas = Medium("gaas",500,"asd",GaAs_ref) 
sio2 = Medium("sio2",d_sio2,"dbr",sio2_r)
cavlayer = Medium("sio2",d_sio2*2,"dbr",sio2_r)
tio2 = Medium("tio2",d_tio2,"dbr",tio2_r)
air = Medium("air",0,"air",air_r)

#Definition of Upper and Lower Mirror
lower = PeriodicStructure([tio2,sio2],periodicity=9)
upper = PeriodicStructure([sio2,tio2],periodicity=9)

#Definition of Cavity
cav = Structure([air,upper,cavlayer,lower,gaas])

#%% Simulation 1d Spectrum
cav.angle = 0
cav.wavelength = np.linspace(0.7,1.05,500)

output = cav.spectrum()
fig,ax = plt.subplots(dpi = 500)
ax.plot(cav.wavelength,output[0,:]) #TE Reflectivity
ax.set_xlim(0.7,1.05)
ax.set_xlabel("Wavelength (um)")
ax.set_ylabel("Reflectivity")

#%% Simulation 2d Spectrum
cav.angle = np.linspace(-30,30,1000)*np.pi/180
cav.wavelength = np.linspace(0.7,1.05,1000)

output = cav.spectrum()
fig,ax = plt.subplots(dpi = 500)
ax.pcolormesh(cav.angle*180/np.pi,cav.wavelength,output[0,:,:]) #TE Reflectivity
# ax.set_xlim(0.7,1.05)
ax.set_ylabel("Wavelength (um)")
ax.set_xlabel("Incidence Angle (°)")
plt.show()
