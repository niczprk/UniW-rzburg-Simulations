"""
Created on Wed June 3 16:01:36 2025

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

cwl_bottom = 0.576 #Wavelength in um
ce_bottom = to_energy(cwl_bottom)
wavelength_range = np.linspace(0.155, 0.8 , 500)
energies = to_energy(wavelength_range)

Ag_ref = Const([0.0534,3.958])
# Sellmeier equation for quartz (optional; not used directly here)
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
PMMA_Index = Const([1.49, 0])

n, k = MC_Index.get_index(energies)

SPI = EMA(SPI_Index, PMMA_Index, { #applies the mix index for the specific loop with the PMMA index
        'Type': 'Bruggeman',
        #"Depolarization Factor": 0.33, # assuming equal depolarization for both components
        'Weights': (3, 2), # assuming 3 parts SPI and 2 parts PMMA
    'Densities': (1.168, 0.8724) #SPI density unknown however less than MC // PMMA-toulene at 3.3% wt
    })

MC = EMA(MC_Index, PMMA_Index, { #applies the mix index for the specific loop with the PMMA index
        'Type': 'Bruggeman',
        #"Depolarization Factor": 0.33, # assuming equal depolarization for both components
        'Weights': (3, 2), # assuming 3 parts SPI and 2 parts PMMA
    'Densities': (1.268, 0.8724) #MC density known // PMMA-toulene at 3.3% wt
    })

Ag_top= Medium("Ag",0.035,"metal",Ag_ref)
Ag_bottom= Medium("Ag",0.035,"metal",Ag_ref)
SPI_Film = Medium(f"SPI_PMMA", 0.136, "polymer", SPI)
MC_Film = Medium(f"MC_PMMA", 0.136, "polymer", MC)
Substrate = Medium("Substrate", 500, "glass", Const([1.46, 0])) #fused silica
Air = Medium("Air",0,"air",air_r)

#%% Simulation Parameters

angles_rad = np.linspace(0, 61, 1000) * np.pi / 180
wavelengths = np.linspace(0.155, 0.8, 1000)
energies = to_energy(wavelengths)

#%% Build Structures

cav_MC = Structure([Air, Ag_top, MC_Film, Ag_bottom, Substrate])
cav_SPI = Structure([Air, Ag_top, SPI_Film, Ag_bottom, Substrate])

cav_MC.angle = angles_rad
cav_SPI.angle = angles_rad
cav_MC.wavelength = wavelengths
cav_SPI.wavelength = wavelengths

#%% Run Simulations

output_MC = cav_MC.spectrum()[0, :, :]  # TE something
output_SPI = cav_SPI.spectrum()[0, :, :]  # TE something

#%% Plot Side-by-Side Dispersion

fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=500, sharey=True)

mesh1 = axs[0].pcolormesh(
    angles_rad * 180 / np.pi, energies, output_SPI, shading='auto', cmap = 'viridis'
)
axs[0].set_title("SPI Form (Uncoupled)",fontsize=4)
axs[0].set_xlabel("Incidence Angle (°)",fontsize=3)
axs[0].set_ylim(1.6, 3.0)  # Limit y-axis to 3.0 eV
axs[0].set_ylabel("Energy (eV)",fontsize=3)
plt.colorbar(mesh1, ax=axs[0])

mesh2 = axs[1].pcolormesh(
    angles_rad * 180 / np.pi, energies, output_MC, shading='auto', cmap = 'viridis'
)
axs[1].set_title("MC Form (Strong Coupling)", fontsize=4)
axs[1].set_xlabel("Incidence Angle (°)", fontsize=3)
axs[1].set_ylim(1.6, 3.0)  # Limit y-axis to 3.0 eV

plt.colorbar(mesh2, ax=axs[1])

plt.suptitle("Cavity Dispersion: SPI vs. MC", fontsize=7)
plt.tight_layout()
plt.subplots_adjust(top=0.5)
plt.show()