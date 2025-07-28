import sys
sys.path.append(r"C:\Users\s531596\Documents\GitHub\tmm\\")

from tmm import to_energy
from refractives import Const, GenOsc, EMA, Sellmeier
import numpy as np
import matplotlib.pyplot as plt

# Wavelength range in microns and energies in eV
wavelength_range = np.linspace(0.248, 1.24, 500)
energies = to_energy(wavelength_range)

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
# params_SPI = [
#     0.275, 3.45, 0.271875,    # UV edge
#     0.15, 4.3, 0.421875,     # mid UV
#     0.45, 4.6, 0.5625,       # mid UV
#     0.6, 5.0, 1.125,         # far UV tail
#     2.4
# ]

params_SPI = [
    0.275, 3.45, 1.6884,
    0.15, 4.3, 1.701,
    0.45, 4.6, 2.268,
    0.6, 5.0, 4.536,
    2.4
]

SPI_Index = GenOsc(params_SPI)


params_MC = [
    0.65, 2.215, 5.04,
    0.35, 3.215, 3.0756,
    0.225, 3.7, 0.81144,
    0.5, 5.0, 3.8796,
    2.5
]

MC_Index = GenOsc(params_MC)

# params_SPI = [
#     0.275, 3.45, 0.271875,    # UV edge
#     0.15, 4.3, 0.421875,     # mid UV
#     0.45, 4.6, 0.5625,       # mid UV
#     0.6, 5.0, 1.125,         # far UV tail
#     2.4
# ]
# SPI_Index = GenOsc(params_SPI)

# # MC: absorption at ~2.15 eV
# params_MC = [
#     0.45, 2.215, 1.25,         # main MC peak
#     0.35, 3.215, 0.8625,       # shoulder
#     0.225, 3.7, 0.20125,     # tail
#     0.5, 5.0, 0.9625,        # far UV absorption
#     2.5
# ]
# MC_Index = GenOsc(params_MC)

# Constant refractive index for PMMA
PMMA_Index = Const([1.49, 0])

# Setup plot
fig, ax = plt.subplots(dpi=500)
colors = plt.cm.viridis(np.linspace(0, 1, 6))

# Loop over conversion fraction from 0% to 100% MC
for i, alpha in enumerate(np.linspace(0, 1, 6)):

    # Step 1: mix SPI and MC
    mix_material = EMA(SPI_Index, MC_Index, {
        'Type': 'Bruggeman',
        'Weights': (1 - alpha, alpha),
        'Densities': (1.168, 1.268)
    })

    # Step 2: mix resulting molecular material with PMMA
    final_material = EMA(mix_material, PMMA_Index, {
        'Type': 'Bruggeman',
        'Weights': (3, 2),
        'Densities': (1.268, 0.8724)  # Densities for SPI and PMMA
    })

    # Extract n, k and plot k only
    n, k = final_material.get_index(energies)
    ax.plot(energies, k, label=f'{int(alpha*100)}% MC', color=colors[i])

# Plot settings
ax.set_xlim(to_energy(wavelength_range)[0], to_energy(wavelength_range)[-1])
ax.set_xlabel("Photon Energy (eV)", fontsize=4)
ax.set_ylabel("Extinction Coefficient k", fontsize=4)
ax.set_title("Absorption Evolution with SPI → MC Conversion", fontsize=5)
ax.legend(fontsize=2, loc='upper right')
ax.grid(True)

plt.show()