# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:25:45 2025
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
    0.2, 3.6, 0.05,   # Shifted to avoid strong absorption at 560 nm (3.6 eV)
    0.3, 4.5, 0.05,
    1.5              # slightly raised background ε 
]
SPI_Index = GenOsc(params_SPI)

# MC: main visible absorption at ~2.15 eV
params_MC = [
    0.3, 4.5, 0.05,  # A, E, gamma 
    0.1, 2.5833, 0.1,
    0.1, 1.8235, 0.1,
    1.7  #background dielectric constant increased to account for MC polarizability
]
MC_Index = GenOsc(params_MC)


Ag_ref = Const([0.056,3.67])

# Refractive indices of constant materials
PMMA_Index = Const([1.49, 0])
Air = Medium("Air", 0, "air", air_r)
Substrate = Medium("Substrate", 500, "glass", Const([1.458, 0]))
Ag_top= Medium("Ag",0.035,"metal",Ag_ref)
Ag_bottom= Medium("Ag",0.035,"metal",Ag_ref)
Substrate = Medium("Substrate", 500, "glass", Quartz_Index)



"""
A: Oscillator Strength (area) - controls how much the oscillator contributes to absorption
E: Oscillator Energy (eV) - energy at which the oscillator is centered
gamma: Broadening (linewidth, in eV) - controls the width of the absorption peak, the larger the value the broader and flatter the peak
Final parameter: background dielectric constant (ε) - accounts for the material's inherent polarizability
"""

n_MC, k_MC = MC_Index.get_index(energies)
n_SPI, k_SPI = SPI_Index.get_index(energies)

# Setup plot
fig, ax = plt.subplots(dpi=500)
colors = plt.cm.viridis(np.linspace(0, 1, 6))

# Loop over conversion fraction
for i, alpha in enumerate(np.linspace(0, 1, 6)): #Loops over 6 values from 0 to 1

    mix_medium = EMA(SPI_Index, MC_Index, { #applies the mix index for the specific loop with the PMMA index
        'Type': 'Bruggeman',
        'Weights': (1-alpha, alpha), # assuming 3 parts SPI and 2 parts PMMA
    'Densities': (1.3, 1.3) #unsure of densities for SPI and PMMA
    })

    SPI_PMMA = EMA(mix_medium, PMMA_Index, { #applies the mix index for the specific loop with the PMMA index
        'Type': 'Bruggeman',
        'Weights': (3, 2), # assuming 3 parts SPI and 2 parts PMMA
        'Densities': (1.3, 1.19) #unsure of densities for SPI and PMMA
    })

    Film = Medium(f"SPI_PMMA_{int(alpha*100)}", 0.126, "polymer", SPI_PMMA)
    stack = Structure([Air, Film, Air])
    stack.angle = 0
    stack.wavelength = wavelength_range

    T = stack.spectrum()[1, :] # Transmission spectrum
    OD = -np.log10(T + 1e-12) #convert to absorbance (optical density) log(0) is avoided by adding a small value

    ax.plot(wavelength_range, OD, label=f'{int(alpha*100)}% MC', color=colors[i])

# Plot settings
ax.set_xlim(0.25, 0.65)
ax.set_xlabel("Wavelength (µm)", fontsize=5)
ax.set_ylabel("Absorbance [OD]", fontsize=5)
ax.legend(fontsize=4, loc='upper right')
ax.set_title("Absorbance of SPI-PMMA Blend with Varying MC Fraction", fontsize = 5)

# Twin x-axis for photon energy
ax2 = ax.twiny()
ax2.set_xlim(to_energy(wavelength_range[0]), to_energy(wavelength_range[-1]))
ax2.set_xlabel("Photon Energy (eV)", fontsize = 5)

fig,ax = plt.subplots(dpi = 500)
ax.plot(energies,k_MC, label='MC K', color='blue')
ax.set_xlabel("Photon Energy  (eV)", fontsize=5)
ax.set_xlabel("Extinction Coefficient (k)", fontsize=5)

plt.show()
