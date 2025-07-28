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

alpha = 0.8 # 60% MC, 40% SPI

Mix = EMA(MC_Index, SPI_Index, {
        'Type': 'Bruggeman',
        #"Depolarization Factor": 0.33, # assuming equal depolarization for
        'Weights': (alpha, 1 - alpha), # assuming 3 parts MC and 2 parts SPI
    'Densities': (1.268, 1.168) #MC density
})

Alpha = EMA(Mix, PMMA_Index, { #applies the mix index for the specific loop with the PMMA index
        'Type': 'Bruggeman',
        #"Depolarization Factor": 0.33, # assuming equal depolarization for
        'Weights': (3, 2), # assuming 3 parts Mix and 2 parts PMMA
    'Densities': (1.268, 0.8724) #MC density known // PMMA-toulene at 3.3% wt
})



MC_Film = Medium("MC_Film", 0.136, "film", MC)
SPI_Film = Medium("SPI_Film", 0.136, "film", SPI)
Mix_Film = Medium("Mix_Film", 0.136, "film", Alpha)

Ag_top= Medium("Ag",0.035,"metal",Ag_ref)
Ag_bottom= Medium("Ag",0.035,"metal",Ag_ref)
Substrate = Medium("Substrate", 500, "glass", Const([1.46, 0])) #fused silica
Air = Medium("Air",0,"air",air_r)

cav = Structure([Air, Ag_top, SPI_Film, Ag_bottom, Substrate])
cav3 = Structure([Air, Ag_top,Mix_Film, Ag_bottom, Substrate])
cav2 = Structure([Air, Ag_top, MC_Film, Ag_bottom, Substrate])

cav.angle = 0

cav.wavelength = np.linspace(0.4,1.2,1500)
output = cav.spectrum()

cav2.angle = 0

cav2.wavelength = np.linspace(0.4,1.2,1500)
output2 = cav2.spectrum()

cav3.angle = 0

cav3.wavelength = np.linspace(0.4,1.2,1500)
output3 = cav3.spectrum()

transmission = 0.5 * (output[2,:] + output[3,:])  # Average TE and TM transmission
reflection = 0.5 * (output[0,:] + output[1,:])  # Average TE and TM reflection

fig, ax = plt.subplots(dpi=500)
#ax.set_xlim(0.2,0.8)
#ax.set_xlabel("Wavelength (um)", fontsize=4)
ax.plot(to_energy(cav.wavelength), output[2,:], label="SPI TE Transmission 0.0% MC", color='darkgreen')
ax.plot(to_energy(cav3.wavelength), output3[2,:], label=f"MC TE Transmission {alpha*100}% MC", color='blue')
ax.plot(to_energy(cav2.wavelength), output2[2,:], label="MC TE Transmission 100.0% MC", color='orange')
#ax.set_xlim(2.05, 2.25)
ax.set_xlabel("Energy (eV)", fontsize=4)
ax.set_ylabel("Transmission", fontsize=4)
#ax.set_xlim(2.05, 2.25)

#add secondary x-axis for wavelength
#secax = ax.secondary_xaxis('top', functions =(
#    lambda E: 1.239841984 / E,       # bottom → top
#    lambda λ: 1.239841984 / λ        # top → bottom
#))
#secax.set_xlabel("Wavelength (µm)", fontsize=4)

#energy_ticks = ax.get_xticks()
#wavelength_ticks = 1.239841984 / energy_ticks
#secax.set_xticks(energy_ticks)
#secax.set_xticklabels([f"{wtick:.2f}" for wtick in wavelength_ticks])

# n, k = MC_Index.get_index(to_energy(np.linspace(0.4, 0.75, 1500)))
# plt.plot(to_energy(np.linspace(0.4, 1.2, 1500)), k)
# plt.title("MC Absorption (k)")
# plt.xlabel("Energy (eV)")
# plt.ylabel("k")

ax.legend(fontsize=2, loc='upper right')
plt.tight_layout()
plt.show()