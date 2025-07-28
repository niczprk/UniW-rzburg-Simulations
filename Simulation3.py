# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:42:36 2025
@author: niczprk
"""

import sys
sys.path.append(r"C:\Users\s531596\Documents\GitHub\tmm\\")

from tmm import Medium, Structure, to_energy
from official_index_list import air as air_r
from refractives import Const, GenOsc, EMA, Sellmeier
import numpy as np
import matplotlib.pyplot as plt

# Wavelength range in microns
wavelength_range = np.linspace(0.248, 1.24, 500)
energies = to_energy(wavelength_range)
Ag_ref = Const([0.056,3.67]) # Refractive index for silver at 560 nm

"""
A: Oscillator Strength (area) - controls how much the oscillator contributes to absorption
E: Oscillator Energy (eV) - energy at which the oscillator is centered
gamma: Broadening (linewidth, in eV) - controls the width of the absorption peak, the larger the value the broader and flatter the peak
Final parameter: background dielectric constant (ε) - accounts for the material's inherent polarizability
"""

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
    0.2, 3.55, 0.125,    # UV edge (sharp absorption onset)
    0.45, 4.3, 0.25,    # mid UV
    0.6, 5.0, 0.5,     # far UV tail
    1.5
]
SPI_Index = GenOsc(params_SPI)

# MC: absorption at ~2.15 eV
params_MC = [
    0.5, 2.15, 0.4,     # main MC peak at 560 nm
    0.4, 3.15, 0.25,    # shoulder
    0.25, 3.7, 0.15,   # tail
    0.5, 5.1, 0.5,    # far UV absorption
    1.7
]
MC_Index = GenOsc(params_MC)


# Refractive indices of constant materials
PMMA_Index = Const([1.49, 0]) # Refractive index for PMMA at 560 nm
Air = Medium("Air", 0, "air", air_r)
Substrate = Medium("Substrate", 500, "glass", Quartz_Index) #fused silica
Ag_top= Medium("Ag",0.035,"metal",Ag_ref)
Ag_bottom= Medium("Ag",0.035,"metal",Ag_ref)

# Setup plot
fig, ax = plt.subplots(dpi=500)
colors = plt.cm.viridis(np.linspace(0, 1, 6))

# Loop over conversion fraction
for i, alpha in enumerate(np.linspace(0, 1, 6)): #Loops over 6 values from 0 to 1
    
    mix_medium = EMA(SPI_Index, MC_Index, { #applies the mix index for the specific loop with the PMMA index
        'Type': 'Bruggeman',
        #"Depolarization Factor": 0.33, # assuming equal depolarization for both components
        'Weights': (1-alpha, alpha), # assuming 3 parts SPI and 2 parts PMMA
    'Densities': (1.3, 1.3) #unsure of densities for SPI and PMMA
    })

    final_medium = EMA(mix_medium, PMMA_Index, {
        'Type': 'Bruggeman',
        #"Depolarization Factor": 0.33, # assuming equal depolarization for both components
        'Weights': (3, 2), # assuming 3 parts SPI and 2 parts PMMA
        'Densities': (1.3, 1.19) #unsure of densities for SPI and PMMA
    })

    Film = Medium(f"SPI_PMMA_{int(alpha*100)}", 0.126, "polymer", final_medium)
    cav = Structure([Air, Ag_top, Film, Ag_bottom, Substrate])
    cav.angle = 0
    cav.wavelength = wavelength_range

    output = cav.spectrum() 
    T_percent = output[1, :] * 100  # Convert to percentage

    ax.plot(wavelength_range*1000 , T_percent , label=f'{int(alpha*100)}% MC', color=colors[i])

# Plot settings
ax.set_xlim(250, 750)
ax.set_xlabel("Wavelength (nm)", fontsize=5)
ax.set_ylabel("Transmission (%)", fontsize=5)
#ax.legend(fontsize=4, loc='upper right')
ax.set_title("Transmission of SPI-PMMA Blend with Varying MC Fraction", fontsize = 5)
plt.grid(True, linewidth = 0.25, linestyle='--', alpha=0.5)
plt.show()
