"""
Created on Wed May 21 16:25:45 2025

@author: niczprk
"""
import sys
sys.path.append(r"C:\Users\s531596\Documents\GitHub\tmm\\")
from tmm import Medium, Structure, PeriodicStructure,to_energy
from official_index_list import sio2_sputter as sio2_r
from official_index_list import tio2_sputter as tio2_r 
from official_index_list import air as air_r
from refractives import Const
from refractives import GenOsc
from refractives import EMA
from refractives import Sellmeier
import numpy as np
import matplotlib.pyplot as plt

#%% Device Setup

cwl_bottom = 0.560 #Wavelength in um
ce_bottom = to_energy(cwl_bottom)
wavelength_range = np.linspace(0.200, 1.24, 500)
energies = to_energy(wavelength_range)

Ag_ref = Const([0.0534,3.958])

quartz_params = [
    0.6961663,
    0.4079426,
    0.8974794,
    0.004679148,  # 0.0684043^2
    0.013512063,  # 0.1162414^2
    97.934002     # 9.896161^2
]
Quartz_Index = Sellmeier(quartz_params)

params_SPI = [
    0.275, 3.45, 0.271875,    # UV edge
    0.15, 4.3, 0.421875,     # mid UV
    0.45, 4.6, 0.5625,       # mid UV
    0.6, 5.0, 1.125,         # far UV tail
    2.1
]
SPI_Index = GenOsc(params_SPI)

# MC: absorption at ~2.15 eV
params_MC = [
    0.45, 2.215, 1.25,         # main MC peak
    0.35, 3.215, 0.8625,       # shoulder
    0.225, 3.7, 0.20125,     # tail
    0.5, 5.0, 0.9625,        # far UV absorption
    2.2
]
MC_Index = GenOsc(params_MC)

# Constant refractive index for PMMA
# PMMA_Index = Const([1.49, 0])

PMMA_params = [
    1.1819,
    0.0,
    0.0,
    0.011313,
    1.0,
    1.0
]

PMMA_Index = Sellmeier(PMMA_params)

n, k = MC_Index.get_index(energies)

SPI = EMA(SPI_Index, PMMA_Index, { #applies the mix index for the specific loop with the PMMA index
        'Type': 'Bruggeman',
        #"Depolarization Factor": 0.33, # assuming equal depolarization for both components
        'Weights': (3, 2), # assuming 3 parts SPI and 2 parts PMMA
    'Densities': (1.168, 0.8724) #unsure of densities for SPI and PMMA
    })

MC = EMA(MC_Index, PMMA_Index, { #applies the mix index for the specific loop with the PMMA index
        'Type': 'Bruggeman',
        #"Depolarization Factor": 0.33, # assuming equal depolarization for both components
        'Weights': (3, 2), # assuming 3 parts SPI and 2 parts PMMA
    'Densities': (1.268, 0.8724) #according to Chemical Book
    })

Mix_Index = EMA(SPI_Index, MC_Index, { #applies the mix index for the specific loop with the PMMA index
    'Type': 'Bruggeman',
    'Weights': (1, 1), # equal weights for SPI and MC
    'Densities': (1.168, 1.268) # assuming equal densities for both components
})


# SPI_PMMA = EMA(MC_Index, PMMA_Index, {
#     'Type': 'Bruggeman',
#     'Weights': (2,2), #3:2 ratio of SPI to PMMA
#     'Densities': (1.268,  0.8724), #these need to be adjusted #Densities of SPI, PMMA 
# })

# Mix_PMMA = EMA(Mix_Index, PMMA_Index, {#Definition of Media
#     'Type': 'Bruggeman',
#     'Weights': (3,2), #3:2 ratio of SPI to PMMA
#     'Densities': (1.268, 0.8724), #these need to be adjusted #Densities of SPI, PMMA 
# })

Ag_top= Medium("Ag",0.035,"metal",Ag_ref)
Ag_bottom= Medium("Ag",0.035,"metal",Ag_ref)
SPI_Film = Medium(f"SPI_PMMA", 0.136, "polymer", SPI)
MC_Film = Medium(f"MC_PMMA", 0.136, "polymer", MC)
#Mix_Film = Medium(f"Mix_PMMA", 0.126, "polymer", Mix_PMMA)
Substrate = Medium("Substrate", 500, "glass", Quartz_Index) #fused silica
Air = Medium("Air",0,"air",air_r)


#Definition of Cavity
cav = Structure([Air, Ag_top, MC_Film, Ag_bottom, Substrate])

#%% Simulation 1d Spectrum
cav.angle = 0
cav.wavelength = np.linspace(0.2,0.8,500)

output = cav.spectrum()
fig,ax = plt.subplots(dpi = 500)
ax.plot(to_energy(cav.wavelength),output[2,:]) #TE transmission
ax.set_xlim(1.2,2.9)
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Transmission")

#%% Simulation 2d Spectrum
cav.angle = np.linspace(-30,30,1000)*np.pi/180
cav.wavelength = np.linspace(0.2,0.8,1000)

output = cav.spectrum()
fig,ax = plt.subplots(dpi = 500)
#ax.pcolormesh(cav.angle*180/np.pi,cav.wavelength,output[0,:,:], shading='auto', cmap = 'viridis') #TE Reflectivity
#ax.pcolormesh(cav.angle*180/np.pi,cav.wavelength,output[0,:,:], shading='auto', cmap = 'viridis') #TE Transmission
plt.colorbar(ax.pcolormesh(cav.angle*180/np.pi,to_energy(cav.wavelength),output[0,:,:], shading = "auto", cmap = 'viridis'), ax=ax, label="Reflectivity")# ax.set_xlim(0.7,1.05)
ax.set_ylabel("Energy (eV)")
ax.set_xlabel("Incidence Angle (°)")
ax.set_ylim(1.5,3)

#fig,ax = plt.subplots(dpi = 500)
#ax.plot(energies,k)

plt.show()