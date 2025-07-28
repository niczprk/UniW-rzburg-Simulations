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

cwl_bottom = 0.580 #Wavelength in um
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

# SPI: transparent in visible, absorbs in UV
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

Ag_top= Medium("Ag",0.035,"metal",Ag_ref)
Ag_bottom= Medium("Ag",0.035,"metal",Ag_ref)
Substrate = Medium("Substrate", 500, "glass", Const([1.46, 0])) #fused silica
Air = Medium("Air",0,"air",air_r)

#%% SPI -> MC 1D Spectrum Simulation 

fig,ax = plt.subplots(dpi = 500)
colors = plt.cm.viridis(np.linspace(0, 1, 6))
# Loop over conversion fraction from 0% to 100% MC
for i, alpha in enumerate(np.linspace(0, 1, 6)):

    # Step 1: mix SPI and MC
    mix_material = EMA(SPI, MC, {
        'Type': 'Bruggeman',
        'Weights': (1 - alpha, alpha),
        'Densities': (1.168, 1.268)
    })

    # Step 2: mix resulting molecular material with PMMA
    Final_Material = EMA(mix_material, PMMA_Index, {
        'Type': 'Bruggeman',
        'Weights': (3, 2),
        'Densities': (1.268, 0.8724)  # Densities for SPI and PMMA
    })

    # Extract n, k and plot k only
    Changing_Film = Medium(f"Mix_PMMA", 0.136, "polymer", Final_Material)
    cav = Structure([Air, Ag_top, Changing_Film, Ag_bottom, Substrate])
    cav.angle = 0
    cav.wavelength = np.linspace(0.2,0.8,500)
    output = cav.spectrum()
    transmission = 0.5 * (output[2,:] + output[3,:])  # Average TE and TM transmission
    ax.plot(to_energy(cav.wavelength), output[3,:], label=f'{int(alpha*100)}% MC', color=colors[i])

#ax.set_xlim(0.2,0.8)
ax.set_xlabel("Wavelength (um)", fontsize=4)
ax.set_ylabel("Transmission", fontsize=4)
ax.set_title("Transmission Evolution of SPI → MC Conversion", fontsize=5)
ax.legend(fontsize=2, loc='upper right')
ax.grid(True)
plt.tight_layout()

plt.show()