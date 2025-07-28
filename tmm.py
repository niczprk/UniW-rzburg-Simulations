#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:33:28 2021

@author: johannes

"""

import numpy as np
from tmm_compute import calc_1d_spectrum_detuning, calc_2d_spectrum_detuning
from tmm_compute import (
    calc_1d_spectrum,
    calc_2d_spectrum,
    calc_1d_reflection_amplitude
)
# import sys
# sys.path.append(r"C:\Users\s531596\Documents\GitHub\tmm\\")
import constants as c
import builtins



def to_wav(energies):

    return c.H * c.C / (energies * c.E) * 1e6


def to_energy(wavs):

    return c.H * c.C / (wavs * 1e-6) / c.E


class Medium:

    def __init__(self, material, h, comment, refractive_index):

        self.material = material
        self.comment = comment
        self.d = h
        self.refractive_index = refractive_index

    @property
    def layers(self):

        return [self]


class Structure:
    """
    A structure defines the layer sequence and exposes the underlying tmm_compute module.
    It is possible to generate 1d/2d spectra that are energy/angle resolved and perform
    detuning series.

    !MISSING: Electric Field Intensities


    """

    def __init__(self, structures):

        self.d = []
        self.layers = []
        self._nk = None
        self.material = []
        self.comment = []
        self._angle = 0.0
        self._expand_structures(structures)

    @staticmethod
    def to_wav(energies):

        return c.H * c.C / (energies * c.E) * 1e6

    @staticmethod
    def to_energy(wavs):

        return c.H * c.C / (wavs * 1e-6) / c.E

    @property
    def wavelength(self):

        return self._wavelength

    @wavelength.setter
    def wavelength(self, vals):
        self._index_computed = False
        self._wavelength = np.asarray(vals, dtype=np.float64)
        self._energy = Structure.to_energy(self._wavelength)

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, vals):
        self._index_computed = False
        self._energy = np.asarray(vals, dtype=np.float64)
        self._wavelength = Structure.to_wav(self._energy)

    @property
    def refractive_indices(self):
        if not self._index_computed:
            if not hasattr(self, "_energy"):
                raise NameError("Set energy first")

            self._calc_refractive_indices()
            self._index_computed = True
        return self._nk

    @refractive_indices.setter
    def refractive_indices(self, vals):

        raise AttributeError(
            "Refractive Index does not support manual assignment"
        )

    def _expand_structures(self, structures):

        for struct in structures:

            self.layers.extend(struct.layers)

        for lay in self.layers:
            self.d.append(lay.d)
            self.material.append(lay.material)
            self.comment.append(lay.comment)

    def _calc_refractive_indices(self):

        self._nk = np.zeros(
            (self._energy.shape[0], len(self.layers)), dtype=np.complex128
        )

        mats = np.array(self.material)
        unique_materials = np.unique(mats)

        for mat in unique_materials:
            indices = np.where(mats == mat)[0]
            layer = self.layers[indices[0]]
            refind = layer.refractive_index
            n, k = refind.get_index(self._energy)
            for index in indices:

                self._nk[:, index] = n - 1j * k

    @property
    def angle(self):

        return self._angle

    @angle.setter
    def angle(self, val):

        if (
            (type(val) == int)
            or (type(val) == float)
            or (type(val) == np.float64)
        ):

            self._angle = float(val)

        else:

            self._angle = np.asarray(val, dtype=np.float64)

    def _thickness_by_comment(self, change_dict):

        keys = list(change_dict.keys())
        comms = np.array(self.comment)
        size = np.array(change_dict[keys[0]]).shape

        match len(size):

            case 1:
                # thicknesses = np.array(change_dict[key])
                d_sim = np.outer(np.ones(size[0]), np.array(self.d))

                for key in keys:

                    idxs = np.where(comms == key)[0]
                    for idx in idxs:
                        d_sim[:, idx] = change_dict[key]

            case 2:
                d_sim = np.outer(
                    np.ones(size[0], size[1]),
                    np.array(self.d),
                    dtype=np.float64,
                )

                for key in keys:

                    idx = np.where(comms == key)[0]
                    d_sim[:, :, idx] = change_dict[key]

        return d_sim

    def detuning(self, change_dict):
        """
        Perform a detuning series.
        Depending on the type of the angle that has been set:
            float: Sweep a thickness for this angle
            np.ndarray: Sweep a thickness for all angles

        Parameters
        ----------
        change_dict : dict
            Key is the comment of the layers to be changed.
            Value is the array of thicknesses.

        Raises
        ------
        NotImplementedError
            Ensure that the thickness-array is only 1d.

        Returns
        -------
        out : np.ndarray
            R_TE,T_TE,R_TM,T_TM for changed thicness and angle.

        """
        d_sim = self._thickness_by_comment(change_dict)
        n = self.refractive_indices
        match (len(d_sim.shape) - 1, type(self.angle)):

            case (1, builtins.float):

                out = calc_1d_spectrum_detuning(
                    self.angle, n, self.wavelength, d_sim
                )

            case (2, builtins.float):

                raise NotImplementedError(
                    "2D Detuning has not been implemented yet"
                )

            case (1, np.ndarray):

                out = calc_2d_spectrum_detuning(
                    self.angle, n, self.wavelength, d_sim
                )

            case (2, np.ndarray):

                raise NotImplementedError(
                    "2D Angular Detuning has not been implemented yet"
                )

        return out

    def wavelength_reflection_amplitude(self):
        """
        Calculate the energy dependent reflection amplitude of the structure

        Returns
        -------
        np.ndarray
            1d array containg TE and TM reflection amplitudes.

        """
        d_sim = np.array(self.d, dtype=np.float64)
        n = self.refractive_indices
        return calc_1d_reflection_amplitude(
            self.angle, n, self.wavelength, d_sim
        )
    def spectrum(self):
        """
        Calculate energy/angle dependent spectrum. Depending on type of angle:
            float: energy spectrum (1d)
            np.ndarray: energy and angle dependent spectrum (2d)

        Returns
        -------
        out : np.ndarray
            1d or 2d spectrum.

        """

        d_sim = np.array(self.d, dtype=np.float64)
        n = self.refractive_indices
        match type(self.angle):

            case builtins.float:

                out = calc_1d_spectrum(self.angle, n, self.wavelength, d_sim)

            case np.ndarray:

                out = calc_2d_spectrum(self.angle, n, self.wavelength, d_sim)

        return out
    
    
        

class PeriodicStructure(Structure):

    def __init__(self, sequence, periodicity=1):
        super().__init__(sequence)

        # self.sequence = sequence
        self.periodicity = periodicity

        self.layers = self.layers * periodicity
        self.d = self.d * periodicity
        self.material = self.material * periodicity
        self.comment = self.comment * periodicity
        self._index_computed = False
