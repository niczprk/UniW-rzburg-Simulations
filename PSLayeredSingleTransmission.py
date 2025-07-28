##% Imports
"""
Created on Monday June 23 11:12:34 2025

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
ce_bottom_array = np.array([ce_bottom])

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

#  refractive indeces for PMMA & Polystyrene
PMMA_params = [
    1.1819,
    0.0,
    0.0,
    0.011313,
    1,
    1
]

PMMA_Index = Sellmeier(PMMA_params)

Polystyrene_params = [
    1.4435,
    0.0,
    0.0,
    0.020216,
    1.0,
    1.0
]

Polystyrene_Index = Sellmeier(Polystyrene_params)


n, k = MC_Index.get_index(energies)

SPI = EMA(SPI_Index, Polystyrene_Index, { #applies the mix index for the specific loop with the PMMA index
        'Type': 'Bruggeman',
        #"Depolarization Factor": 0.33, # assuming equal depolarization for both components
        'Weights': (3, 2), # assuming 3 parts SPI and 2 parts PMMA
    'Densities': (1.168, 1.07) #unsure of densities for SPI and PMMA
    })

MC = EMA(MC_Index, Polystyrene_Index, { #applies the mix index for the specific loop with the PMMA index
        'Type': 'Bruggeman',
        #"Depolarization Factor": 0.33, # assuming equal depolarization for both components
        'Weights': (3, 2), # assuming 3 parts SPI and 2 parts PMMA
    'Densities': (1.268, 1.07) #according to Chemical Book
    })

 # weight factor for the MC and SPI mix, assuming 60% MC and 40% SPI
alpha = 0.35

Mix = EMA(MC_Index, SPI_Index, {
        'Type': 'Bruggeman',
        #"Depolarization Factor": 0.33, # assuming equal depolarization for
        'Weights': (alpha, 1 - alpha), # assuming 3 parts MC and 2 parts SPI
    'Densities': (1.268, 1.168) #MC density
})

Alpha = EMA(Mix, Polystyrene_Index, { #applies the mix index for the specific loop with the PMMA index
        'Type': 'Bruggeman',
        #"Depolarization Factor": 0.33, # assuming equal depolarization for
        'Weights': (3, 2), # assuming 3 parts Mix and 2 parts PMMA
    'Densities': (1.268, 1.07) #MC density known // PMMA-toulene at 3.3% wt
})


d_sio2 = cwl_bottom/(4*sio2_r.get_index(ce_bottom)[0]) #quarter wave layers thickness
d_al2o3 = cwl_bottom/(4*al2o3_r.get_index(ce_bottom)[0]) #quarter wave layers thickness
d_tio2 = cwl_bottom/(4*tio2_r.get_index(ce_bottom)[0]) #quarter wave layers thickness
d_spi = cwl_bottom / (4 * SPI.get_index(np.array([ce_bottom]))[0][0])
d_mc = cwl_bottom / (4 * MC.get_index(np.array([ce_bottom]))[0][0])
#%% Medium Setup

#Ag_top= Medium("Ag",0.035,"metal",Ag_ref)
#Ag_bottom= Medium("Ag",0.035,"metal",Ag_ref)

q = 2

MC_Film = Medium("MC_PMMA", d_spi*q, "polymer", MC)
SPI_Film = Medium("SPI_PMMA", d_spi*q, "polymer", SPI)
Mix_Film = Medium("Mix_PMMA", d_spi*q, "polymer", Alpha)
#Empty_Film = Medium("Mix_PMMA", d_spi*q, "polymer", air_r)


sio2 = Medium("Sio2", d_sio2, "dielectric", sio2_r) #sputtered SiO2
tio2 = Medium("Tio2", d_tio2, "dielectric", tio2_r) #sputtered TiO2
al2o3 = Medium("Al2O3", d_al2o3, "dielectric", Const([1.59, 0])) #sputtered Al2O3

Substrate = Medium("Substrate", 500, "glass", Quartz_Index) #fused silica
Air = Medium("Air",0,"air",air_r)

#%% Periodic Structure Setup

periodicity1 = 8  # Number of pairs in the structure
periodicity2 = 6  # Number of pairs in the structure


lower = PeriodicStructure([tio2,sio2],periodicity=periodicity1)
upper = PeriodicStructure([sio2,tio2],periodicity=periodicity2)


# Extract n, k and plot k only
cav = Structure([Air, upper, SPI_Film, lower, Substrate])
cav2 = Structure([Air, upper, MC_Film, lower, Substrate])
cav3 = Structure([Air, upper, Mix_Film, lower, Substrate])
#cav4 = Structure([Air, upper, Empty_Film, lower, Substrate])


cav.angle = 0

cav.wavelength = np.linspace(0.4,1.2,500)
output = cav.spectrum()

cav2.angle = 0

cav2.wavelength = np.linspace(0.4,1.2,500)
output2 = cav2.spectrum()

cav3.angle = 0

cav3.wavelength = np.linspace(0.4,1.2,500)
output3 = cav3.spectrum()

# cav4.angle = 0

# cav4.wavelength = np.linspace(0.4,1.2,500)
# output4 = cav4.spectrum()

transmission = 0.5 * (output[2,:] + output[3,:])  # Average TE and TM transmission
reflection = 0.5 * (output[0,:] + output[1,:])  # Average TE and TM reflection

# N = periodicity1 + periodicity2
# nL = sio2_r.get_index(ce_bottom)[0]
# nH = tio2_r.get_index(ce_bottom)[0]
# r_ratio = (nL / nH) ** (2 * N)
# R = ((1 - r_ratio) / (1 + r_ratio)) ** 2

# fig, ax = plt.subplots(dpi=500)
#ax.set_xlim(0.2,0.8)
#ax.set_xlabel("Wavelength (um)", fontsize=4)
# ax.set_yscale("log")
# #ax.plot(to_energy(cav4.wavelength), output4[2,:], label="Empty Cavity TE Transmission", color='black', linewidth=0.5)
# ax.plot(to_energy(cav.wavelength), output[2,:], label="SPI TE Transmission 0.0% MC", color='green', linewidth=0.5)
# ax.plot(to_energy(cav3.wavelength), output3[2,:], label=f"MC TE Transmission {alpha*100}% MC", color='orange', linewidth=0.5)
# ax.plot(to_energy(cav2.wavelength), output2[2,:], label="MC TE Transmission 100.0% MC", color='blue', linewidth=0.5)
# #ax.set_xlim(1.9, 2.5)
# ax.set_xlabel("Energy (eV)", fontsize=3)
# ax.set_ylabel("Transmission", fontsize=3)
# ax.set_title(f"{periodicity1} - {periodicity2} SiTi T Spectrum", fontsize=4)
# #ax.set_xlim(2.125, 2.225)

# #add secondary x-axis for wavelength
# #secax = ax.secondary_xaxis('top', functions =(
# #    lambda E: 1.239841984 / E,       # bottom → top
# #    lambda λ: 1.239841984 / λ        # top → bottom
# #))
# #secax.set_xlabel("Wavelength (µm)", fontsize=4)

# #energy_ticks = ax.get_xticks()
# #wavelength_ticks = 1.239841984 / energy_ticks
# #secax.set_xticks(energy_ticks)
# #secax.set_xticklabels([f"{wtick:.2f}" for wtick in wavelength_ticks])

# ax.legend(fontsize=4, loc='upper right')
# plt.tight_layout()
# plt.show()

print(f"Ideal PS empty Cavity Distance: {d_spi*q} ")