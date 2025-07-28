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

results = [
    (1.530, 110),
    (1.540, 88),
    (1.550, 74),
    (1.560, 64),
    (1.570, 56),
    (1.580, 50),
    (1.590, 44),
    (1.600, 40),
    (1.610, 38),
    (1.620, 36),
    (1.630, 32)
]


n_MC = MC.get_index(np.array([ce_bottom]))[0][0]


d_sio2 = cwl_bottom/(4*sio2_r.get_index(ce_bottom)[0]) #quarter wave layers thickness
sio2 = Medium("Sio2", d_sio2, "dielectric", sio2_r) #sputtered SiO2

d_al2o3 = cwl_bottom/(4*al2o3_r.get_index(ce_bottom)[0]) #quarter wave layers thickness
al2o3 = Medium("Al2O3", d_al2o3, "dielectric", al2o3_r) #sputtered Al2O3

Substrate = Medium("Substrate", 500, "glass", Quartz_Index) #fused silica
Air = Medium("Air",0,"air",air_r)

wavelengths  = np.linspace(0.4,0.7,1000)


value = 1.5

d_mc = cwl_bottom / (4 * n_MC)

#%% Medium Setup


q = 2.0 #  number for adjusting film thickness

MC_Film = Medium("MC_PMMA", d_mc*q, "polymer", MC)
SPI_Film = Medium("MC_PMMA", d_mc*q, "polymer", SPI)
empty = Medium("Empty", d_mc*q, "air", air_r)#empty layer for the cavity



#structure setup with different pair numbers for target reflectance
for nH, N in results:

    fig, ax = plt.subplots(dpi=500)
    
    d_n= cwl_bottom/(4*nH) #quarter wave layers thickness
    change_dex = Const([nH, 0.0])
    change = Medium("Tio2", d_n, "dielectric", change_dex) #sputtered TiO2

    # d_mc = cwl_bottom / (4 * n_MC)  

    upper = PeriodicStructure([sio2,change],periodicity=N)
    lower = PeriodicStructure([change,sio2],periodicity=N)
    cav = Structure([Air, upper, SPI_Film, lower, Substrate])
    cav.angle = 0
    cav.wavelength = wavelengths
    output = cav.spectrum()
    

    ax.plot(to_energy(cav.wavelength), output[0, :], label=f'Index: {value:.2f}, N: {N}, Max_R = {output[0, :].max():.3}', color='blue', linewidth=0.5)
    ax.set_title(f"Reflectivity for nH = {nH:.2f} vs SiTo2: {sio2_r.get_index(ce_bottom)[0]:.3f}", fontsize=7)
    ax.legend(fontsize=3, loc='upper right')
    ax.set_xlabel("Energy (eV)", fontsize=4)
    ax.set_ylabel("Reflectivity", fontsize=4)
    #ax.set_xlim(2.175,2.255)
    #ax.set_yscale('log')
    plt.tight_layout()
    

plt.show()