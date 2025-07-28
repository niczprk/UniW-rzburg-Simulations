##% Imports
import sys
sys.path.append(r"C:\Users\s531596\Documents\GitHub\tmm\\")
import numpy as np
import matplotlib.pyplot as plt

spin_settings = ['A1', 'B1', 'C1', 'A2', 'B2', 'C2']

thickness_vial1 = [203.71, 171.31, 130.53, 185.42, 225.29, 162.96]
thickness_vial2 = [247.77, 323.42, 374.38, 288.32, 518.36, 391.8]
thickness_vial3 = [442.58, 627.34, 741.98, 545.78, 633.7, 763.57]
thickness_vial4 = [1071.68, 896.84, 1380.54, 1180.25, 1279.06, 1336.04]

thickness_vial2_varyingRPM1 = [600.17,475.68, 430.71, 420.98, 415.96, 405.12]
thickness_vial2_varyingRPM2 = [348.80, 296.63, 256.11, 249.77, 227.69, 217.51]

RPM_settings = [1500, 2000, 2500, 3000, 3500, 4000]



fix, ax = plt.subplots(dpi = 300)
# ax.plot(spin_settings, thickness_vial1, marker='o', label='Vial 1')
# ax.plot(spin_settings, thickness_vial2, marker='o', label='Vial 2')
# ax.plot(spin_settings, thickness_vial3, marker='o', label='Vial 3')
# ax.plot(spin_settings, thickness_vial4, marker='o', label='Vial 4')
ax.plot(RPM_settings, thickness_vial2_varyingRPM1, marker = 'o', label = 'Vial 1 (Varying RPM)',)
ax.plot(RPM_settings, thickness_vial2_varyingRPM2, marker = 'o', label = 'Vial 2 (Varying RPM)', color='orange')

ax.set_title("Combined Optical Thickness of Vial 2 vs Varying RPM")
ax.set_ylabel("Optical Thickness (nm)")
ax.set_xlabel(" Spin Setting (RPM)")
ax.legend(fontsize = 7, loc = 'upper right')
ax.grid(True)
plt.tight_layout()
plt.show()
