"""
Created on Friday June 24 14:47:34 2025

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
cwl_bottom = 0.576 #Wavelength in um, roughly corresponds to 2.15 eV, the absorption edge of MC
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
alpha = 0.1 # 60% MC, 40% SPI

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

#%% Medium Setup

#Ag_top= Medium("Ag",0.035,"metal",Ag_ref)
#Ag_bottom= Medium("Ag",0.035,"metal",Ag_ref)
MC_Film = Medium("MC_PMMA", d_sio2 * 2, "polymer", MC)
SPI_Film = Medium("SPI_PMMA", d_sio2 * 2, "polymer", SPI)
Mix_Film = Medium("Mix_PMMA", d_sio2 * 2, "polymer", Alpha)
sio2 = Medium("Sio2", d_sio2, "dielectric", sio2_r) #sputtered SiO2
tio2 = Medium("Tio2", d_tio2, "dielectric", tio2_r) #sputtered TiO2
al2o3 = Medium("Al2O3", d_al2o3, "dielectric", al2o3_r) #sputtered Al2O3
Substrate = Medium("Substrate", 500, "glass", Quartz_Index) #fused silica
Air = Medium("Air",0,"air",air_r)

#%% Periodic Structure Setup
lower = PeriodicStructure([tio2,sio2],periodicity=6)
upper = PeriodicStructure([sio2,tio2],periodicity=6)


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
    Final_Material = EMA(mix_material, Polystyrene_Index, {
            'Type': 'Bruggeman',
            'Weights': (3, 2),
            'Densities': (1.268, 1.07)  # Densities for SPI and PS
        })

    # Extract n, k and plot k only
    Changing_Film = Medium(f"SPI_MC_Mix", d_sio2*2, "polymer", mix_material)
    cav = Structure([Air, upper, Changing_Film, lower, Substrate])
    cav.angle = 0
    cav.wavelength = np.linspace(0.35,0.75,1000)
    output = cav.spectrum()
    transmission = 0.5 * (output[2,:] + output[3,:])  # Average TE and TM transmission
    reflection = 0.5 * (output[0,:] + output[1,:])  # Average TE and TM reflection
    ax.plot(to_energy(cav.wavelength), output[2,:], label=f'{int(alpha*100)}% MC', color=colors[i], linewidth=0.5)
    #ax.set_xlim(2.0,2.2)
 

#ax.set_xlabel("Wavelength (um)", fontsize=4)
ax.set_xlabel("Energy (eV)", fontsize=4)
ax.set_ylabel("Transmission", fontsize=4)
ax.set_title("Transmission Evolution of SPI → MC Conversion", fontsize=5)

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

ax.legend(fontsize=2, loc='upper right')
plt.tight_layout()
plt.show()