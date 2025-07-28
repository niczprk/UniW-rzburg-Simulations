"""
Created on Friday June 13 15:53:34 2025

@author: niczprk
"""
##% Imports
import sys
sys.path.append(r"C:\Users\s531596\Documents\GitHub\tmm\\")
from tmm import Medium, Structure, PeriodicStructure,to_energy
from official_index_list import sio2_sputter as sio2_r
from official_index_list import tio2_sputter as tio2_r 
from official_index_list import al2o3_sputter as al2o3_r   
from official_index_list import air as air_r
from refractives import Const
from refractives import GenOsc
from refractives import EMA
from refractives import Sellmeier
import numpy as np
import matplotlib.pyplot as plt


#%% Device Setup

cwl_bottom = 0.560 #Wavelength in um, roughly corresponds to 2.15 eV, the absorption edge of MC
ce_bottom = to_energy(cwl_bottom)

wavelength_range = np.linspace(0.200, 1.24, 1000)
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
    2.4
]
SPI_Index = GenOsc(params_SPI)

# MC: absorption at ~2.15 eV
params_MC = [
    0.45, 2.215, 1.25,         # main MC peak
    0.35, 3.215, 0.8625,       # shoulder
    0.225, 3.7, 0.20125,     # tail
    0.5, 5.0, 0.9625,        # far UV absorption
    2.5
]
MC_Index = GenOsc(params_MC)

# Constant refractive index for PMMA
PMMA_params = [
    1.1819,
    0.0,
    0.0,
    0.011313,
    1,
    1
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

#%% Periodic Structure Setup

index =np.arange(1.53,1.64,0.01) #index of refraction for the quarter wave layers

target_R = 0.99
tolerance = 0.01

n_MC = MC.get_index(np.array([ce_bottom]))[0][0]


d_sio2 = cwl_bottom/(4*sio2_r.get_index(ce_bottom)[0]) #quarter wave layers thickness
sio2 = Medium("Sio2", d_sio2, "dielectric", sio2_r) #sputtered SiO2

# d_al2o3 = cwl_bottom/(4*al2o3_r.get_index(ce_bottom)[0]) #quarter wave layers thickness
# al2o3 = Medium("Al2O3", d_al2o3, "dielectric", al2o3_r) #sputtered Al2O3

Substrate = Medium("Substrate", 500, "glass", Quartz_Index) #fused silica
Air = Medium("Air",0,"air",air_r)

wavelengths  = np.linspace(0.4,1.2,500)

colors = plt.cm.viridis(np.linspace(0, 1, len(index)))

ns = []

for i, value in enumerate(index):

    # plotting each index value
    # fig,ax = plt.subplots(dpi = 500)


    d_n= cwl_bottom/(4*value) #quarter wave layers thickness
    # d_mc = cwl_bottom / (4 * n_MC)

    #%% Medium Setup
    change_dex = Const([value, 0.0])
    change = Medium("Tio2", d_n, "dielectric", change_dex) #sputtered TiO2

    # q = 2.0 #  number for adjusting film thickness

    # MC_Film = Medium("MC_PMMA", d_mc*q, "polymer", MC)

    # fixed_lower = PeriodicStructure([change,sio2],periodicity=8)

    #structure setup with different pair numbers for target reflectance
    for N in range(20,120,2):
        upper = PeriodicStructure([sio2,change],periodicity=N)


        cav = Structure([Air, upper, Substrate])
        cav.angle = 0
        cav.wavelength = wavelengths
        output = cav.spectrum()

        R = output[0, :]

        if R.max() >= target_R - tolerance:
            print(f"nH={value:.3f}, N_upper={N}, Max R={R.max():.4f}")

            ns.append(N)
            break

        
        #print(f"nH={value:.3f}, N_upper={N}, Max R={R.max():.4f}")


    # ax.plot(to_energy(cav.wavelength), output[0,:], label=f'Index: {value:.2f}, N: {N}, Max_R = {R.max():.3}', color=colors[i], linewidth=0.5)
    # ax.set_title(f"Reflectivity for nH = {value:.2f}", fontsize=7)
    # ax.legend(fontsize=3, loc='upper right')
    # ax.set_xlabel("Energy (eV)", fontsize=4)
    # ax.set_ylabel("Reflectivity", fontsize=4)
    # ax.set_yscale('log')
    # ax.set_xlim(2.0,2.4)

#ax.set_xlabel("Wavelength (um)", fontsize=4)
#ax.set_yscale('log')

# plt.tight_layout()
# plt.show()

fig,ax = plt.subplots(dpi = 500)
ax.scatter(index, ns, color='blue', s=1)
ax.set_yscale('log')
plt.show()