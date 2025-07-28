"""
Created on Friday June 17 16:48:34 2025

@author: niczprk
"""
import sys
sys.path.append(r"C:\Users\s531596\Documents\GitHub\tmm\\")
from tmm import Medium, Structure, PeriodicStructure,to_energy
from official_index_list import sio2_sputter as sio2_r
from official_index_list import tio2_sputter as tio2_r 
from official_index_list import al2o3_sputter as al2o3_r 
from official_index_list import air as air_r
from refractives import Const, GenOsc, EMA, Sellmeier
import numpy as np
import matplotlib.pyplot as plt



#%% Device Setup

cwl_bottom = 0.560 #Wavelength in um
ce_bottom = to_energy(cwl_bottom)
ce_bottom_array = np.array([ce_bottom])

Ag_ref = Const([0.0534,3.958])

GaAs_ref = Const([3.6,0])

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

# MC: absorption at ~2.15 eV
# params_MC = [
#     0.85, 2.215, 1.05,         # main MC peak
#     0.35, 3.215, 0.8625,       # shoulder
#     0.225, 3.7, 0.20125,     # tail
#     0.5, 5.0, 0.9625,        # far UV absorption
#     2.5
# ]


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

alpha = 0.15 

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

#%% Medium Setup
d_sio2 = cwl_bottom/(4*sio2_r.get_index(ce_bottom)[0]) #quarter wave layers thickness
d_tio2 = cwl_bottom/(4*tio2_r.get_index(ce_bottom)[0]) #quarter wave layers thickness
d_al2o3 = cwl_bottom/(4*al2o3_r.get_index(ce_bottom)[0]) #quarter wave layers thickness

d_ag = cwl_bottom / (4 * Ag_ref.get_index(ce_bottom)[0]) #quarter wave thickness for Ag

d_spi = cwl_bottom / (4 * SPI.get_index(np.array([ce_bottom]))[0][0]) 
d_mc = cwl_bottom / (4 * MC.get_index(np.array([ce_bottom]))[0][0])


#Definition of Media
#gaas = Medium("gaas",500,"asd",GaAs_ref) 
sio2 = Medium("sio2",d_sio2,"dbr",sio2_r)
tio2 = Medium("tio2",d_tio2,"dbr",tio2_r)
al2o3 = Medium("al2o3",d_al2o3,"dbr",al2o3_r)

Ag_bottom= Medium("Ag",d_ag,"metal",Ag_ref)

#d_sio2 * 2

q = 3.75

# SPI_Film = Medium("SPI_PMMA", d_spi * q , "polymer", SPI)
# MC_Film = Medium("MC_PMMA", d_spi * q , "polymer", MC)
# Mix_Film = Medium("Mix_PMMA", d_spi * q , "polymer", Alpha)

SPI_Film = Medium("SPI_PMMA", d_spi * q , "polymer", SPI)
MC_Film = Medium("MC_PMMA", d_spi * q , "polymer", MC)
Mix_Film = Medium("Mix_PMMA", d_spi * q , "polymer", Alpha)

Substrate = Medium("Substrate", 500, "glass", Quartz_Index)
air = Medium("air",0,"air",air_r)

#Definition of Upper and Lower Mirror

periodicity1 = 8  # Number of pairs in the structure
periodicity2 = 6  # Number of pairs in the structure


lower = PeriodicStructure([tio2,sio2],periodicity=periodicity1)
upper = PeriodicStructure([sio2,tio2],periodicity=periodicity2)

#Definition of Cavity
cav1 = Structure([air,upper,SPI_Film,lower,Substrate])
cav2 = Structure([air,upper,MC_Film,lower,Substrate])
cav3 = Structure([air,upper,Mix_Film,lower,Substrate])

#%% Simulation 1d Spectrum
cav1.angle = 0
cav1.wavelength = np.linspace(0.4,0.75,1000)

output = cav1.spectrum()
fig,ax = plt.subplots(dpi = 500)
ax.set_yscale("log")
ax.plot(to_energy(cav1.wavelength),output[2,:]) #TE 
#ax.set_xlim(2.0,2.3)
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Transmission")
ax.set_title(f" SiTi 100% SPI Transmission")

#%% Simulation 2d Spectrum
cav1.angle = np.linspace(-30,30,1000)*np.pi/180
#cav1.wavelength = np.linspace(0.4,1.0,1000)

output = cav1.spectrum()
fig,ax = plt.subplots(dpi = 500)
ax.pcolormesh(cav1.angle*180/np.pi,to_energy(cav1.wavelength),output[0,:,:]) #TE Reflectivity
ax.set_ylim(1.75,2.55)
ax.set_ylabel("Energy (um)")
ax.set_xlabel("Incidence Angle (°)")
ax.set_title(f" SiTi 100% SPI Reflectivity Spectrum")

#%% Simulation 1d Spectrum
cav2.angle = 0
cav2.wavelength = np.linspace(0.4,0.75,1000)

output = cav2.spectrum()
fig,ax = plt.subplots(dpi = 500)
ax.set_yscale("log")
ax.plot(to_energy(cav2.wavelength),output[2,:]) #TE 
#ax.set_xlim(2.0,2.3)
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Transmission")
ax.set_title(f" SiTi 100% MC Transmission")

#%% Simulation 2d Spectrum
cav2.angle = np.linspace(-30,30,1000)*np.pi/180
#cav2.wavelength = np.linspace(0.4,1.0,1000)

output = cav2.spectrum()
fig,ax = plt.subplots(dpi = 500)
ax.pcolormesh(cav2.angle*180/np.pi,to_energy(cav2.wavelength),output[0,:,:]) #TE Reflectivity
ax.set_ylim(1.75,2.55)
ax.set_ylabel("Energy (eV)")
ax.set_xlabel("Incidence Angle (°)")
ax.set_title(f" SiTi 100% MC Reflectivity Spectrum")

#%% Simulation 1d Spectrum
cav3.angle = 0
cav3.wavelength = np.linspace(0.4,0.75,1000)

output = cav3.spectrum()
fig,ax = plt.subplots(dpi = 500)
ax.set_yscale("log")
ax.plot(to_energy(cav3.wavelength),output[2,:]) #TE 
#ax.set_xlim(2.0,2.3)
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Reflectivity")
ax.set_title(f" SiTi {alpha*100}% MC TE Transmission")

#%% Simulation 2d Spectrum
cav3.angle = np.linspace(-30,30,1000)*np.pi/180
#cav3.wavelength = np.linspace(0.4,1.0,1000)

output = cav3.spectrum()
fig,ax = plt.subplots(dpi = 500)
ax.pcolormesh(cav3.angle*180/np.pi,to_energy(cav3.wavelength),output[0,:,:]) #TE Reflectivity
ax.set_ylim(1.75,2.55)
ax.set_ylabel("Energy (eV)")
ax.set_xlabel("Incidence Angle (°)")
ax.set_title(f" SiTi {alpha*100}% MC TE Reflectivity Spectrum")
plt.show()
