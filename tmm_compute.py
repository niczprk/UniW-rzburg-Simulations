# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:31:17 2021

@author: johan
"""

from numpy import linalg, empty_like, empty, array, ones,zeros
from numpy import pi
from numpy import cdouble as npcomplex128, double as npreal64
from numpy import arcsin, sin, cos, exp, absolute, power, real, linspace
from numpy import sum as nsum
from numba import jit, prange, complex128, float64,int64
# from scipy.optimize import least_squares


debug = False
parallel = True

#
@jit(nopython = True, nogil = True, cache = True,signature_or_function = complex128[:,:](float64,complex128[:,:]), boundscheck = debug, debug = debug )
def calc_angle(incidence_angle, n):
    """
    Calculates the angle of light in each layer of a device.

    Parameters
    ----------
    incidence_angle : float
        Angle under which the light is incident on the first layer.
    n : 2-D array of floats (W,N)
        Refractive indices in dependence on the W Wavelenghts and N Layers.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    angle : 2-D array of floats (W,N)
        angle[o,p] is the angle of the light with the respect to the surface
        normal - at the o-th wavelength and in the p-th layer.

    """

    angle = empty_like(n)

    for o in prange(n.shape[0]):
        
        angle[o,0] = incidence_angle

        for p in range(n.shape[1]-1):

            angle[o, p+1] = arcsin(sin(angle[o, p]) * n[o, p]/n[o, p+1])

    return angle

#
@jit(nopython = True, nogil = True, cache = True, parallel = parallel,
     boundscheck = debug,signature_or_function = complex128[:,:](float64[:],complex128[:,:],complex128[:,:]), debug = debug)
def calc_k(wavelength, n, angle):
    """
    Calculates the wavevector of light in each layer of a device.

    Parameters
    ----------
    wavelength : 1-D array of floats (W,)
        Contains the W wavelengths the function is evaluated at.
    n : 2-D array of floats (W,N)
        Refractive indices in dependence on the W Wavelenghts and N Layers.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    angle : 2-D array of floats (W,N)
        angle[o,p] is the angle of the light with the respect to the surface
        normal - at the o-th wavelength and in the p-th layer.

    Returns
    -------
    k : 2-D array of floats (W,N)
        wavevectors at each layer in dependence of the W wavelengths.

    """

    k = empty_like(n)

    for o in prange(n.shape[0]):

        for p in range(n.shape[1]):

            k[o, p] = 2 * pi / wavelength[o] * n[o, p] * cos(angle[o, p])

    return k



@jit(nopython = True, nogil = True, cache = True, boundscheck = debug,parallel = parallel,debug = debug,signature_or_function = complex128[:,:,:](float64,complex128[:,:],float64[:],float64[:]))
def calc_transfer_matrix_TE(incidence_angle, n, wavelength, h):
    """
    Calculates the TE polarized transfer matrices of a device based on input
    wavelengths and an incidence angle.

    Parameters
    ----------
    incidence_angle : float
        Angle under which the light is incident on the first layer.
    n : 2-D array of floats (W,N)
        Refractive indices in dependence on the W Wavelenghts and N Layers.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    wavelength : 1-D array of floats (W,)
        Contains the W wavelengths the function is evaluated at.
    h : 1-D array of floats (N,)
        h[k] = Thickness of the k-th layer.

    Returns
    -------
    m : 3-D array of complex floats (2,2,W)
        m[:,:,k] contains the (2,2)-transfer matrix for TE polarization at the
        k-th wavelength.

    """

    m = empty((2, 2, n.shape[0]),dtype = npcomplex128)
    
    k = empty_like(n)
    angle = empty_like(n)

    angle = calc_angle(incidence_angle, n)

    k = calc_k(wavelength, n, angle)

    for i in prange(n.shape[0]):

        m_temp = array([1, 0, 0, 1],dtype = npcomplex128).reshape(2,2)

        for j in range(n.shape[1] - 1):

            d1 = array([ 1, 1 , n[i, j+1]*cos(angle[i,j+1]), -n[i, j+1]*cos(angle[i,j+1 ])],dtype = npcomplex128).reshape((2,2))       
            d0 = array([1, 1,n[i, j]*cos(angle[i,j]), -n[i, j]*cos(angle[i,j])],dtype = npcomplex128).reshape((2,2))       

            d0_inv = linalg.inv(d0)
            p = array([exp(1j * h[j+1] * k[i, j+1]), 0,
                      0, exp(-1j * h[j+1] * k[i, j+1])],dtype = npcomplex128).reshape((2,2))

            m_temp = m_temp @ d0_inv @ d1 @ p

        m[:, :, i] = m_temp

    return m



@jit(nopython = True, nogil = True, cache = True, parallel = parallel,boundscheck = debug,signature_or_function = complex128[:,:,:](float64,complex128[:,:],float64[:],float64[:]), debug = debug)
def calc_transfer_matrix_TM(incidence_angle, n, wavelength, h):
    """
    Calculates the TM polarized transfer matrices of a device based on input
    wavelengths and an incidence angle.

    Parameters
    ----------
    incidence_angle : float
        Angle under which the light is incident on the first layer.
    n : 2-D array of floats (W,N)
        Refractive indices in dependence on the W Wavelenghts and N Layers.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    wavelength : 1-D array of floats (W,)
        Contains the W wavelengths the function is evaluated at.
    h : 1-D array of floats (N,)
        h[k] = Thickness of the k-th layer.

    Returns
    -------
    m : 3-D array of complex floats (2,2,W)
        m[:,:,k] contains the (2,2)-transfer matrix for TM polarization at the
        k-th wavelength.

    """

    m = empty((2, 2, n.shape[0]),dtype = npcomplex128)
    
    k = empty_like(n)
    angle = empty_like(n)

    angle = calc_angle(incidence_angle, n)
    k = calc_k(wavelength, n, angle)
    
    for i in prange(n.shape[0]):

        m_temp = array([1, 0, 0, 1],dtype = npcomplex128).reshape((2,2))

        for j in range(n.shape[1] - 1):
            ## selber invertieren
            d1 = array([cos(angle[i, j+1]), cos(angle[i, j+1]),
                       n[i, j+1], -n[i, j+1]],dtype = npcomplex128).reshape((2,2))          
            d0 = array([cos(angle[i, j]), cos(angle[i, j]),
                       n[i, j], -n[i, j]],dtype = npcomplex128).reshape((2,2))
            d0_inv = linalg.inv(d0)
            p = array([exp(1j * h[j+1] * k[i, j+1]), 0,
                      0, exp(-1j * h[j+1] * k[i, j+1])],dtype = npcomplex128).reshape((2,2))

            m_temp = m_temp @ d0_inv @ d1 @ p

        m[:, :, i] = m_temp

    return m


    
@jit(nopython = True, nogil = True, cache = True, parallel = parallel,boundscheck = debug,signature_or_function = complex128[:,:,:,:](float64,complex128[:,:],float64[:],float64[:,:]), debug = debug)
def detuning_transfer_matrix_TE(incidence_angle, n, wavelength, h):
    """
    Calculates the TE polarized transfer matrices of a device based on input
    wavelengths and an incidence angle for a range of different cavity lengths.

    Parameters
    ----------
    incidence_angle : float
        Angle under which the light is incident on the first layer.
    n : 2-D array of floats (W,N)
        Refractive indices in dependence on the W Wavelenghts and N Layers.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    wavelength : 1-D array of floats (W,)
        Contains the W wavelengths the function is evaluated at.
    h : 2-D array of floats (D,N)
        h[p,k] = Thickness of the k-th layer of the p-th cavity length.

    Returns
    -------
    m : 4-D array of complex floats (D,2,2,W)
        m[p,:,:,k] contains the (2,2)-transfer matrix for TE polarization at the
        k-th wavelength for the p-th cavity length

    """

    m = empty((h.shape[0], 2, 2, n.shape[0]),dtype = npcomplex128)
    
    k = empty_like(n)
    angle = empty_like(n)

    angle = calc_angle(incidence_angle, n)

    k = calc_k(wavelength, n, angle)
    for b in prange(h.shape[0]):
        
        for i in range(n.shape[0]):
    
            m_temp = array([1, 0, 0, 1],dtype = npcomplex128).reshape(2,2)
    
            for j in range(n.shape[1] - 1):
    
                d1 = array([ 1, 1 , n[i, j+1]*cos(angle[i,j+1]), -n[i, j+1]*cos(angle[i,j+1 ])],dtype = npcomplex128).reshape((2,2))       
                d0 = array([1, 1,n[i, j]*cos(angle[i,j]), -n[i, j]*cos(angle[i,j])],dtype = npcomplex128).reshape((2,2))       
    
                d0_inv = linalg.inv(d0)
                p = array([exp(1j * h[b,j+1] * k[i, j+1]), 0,
                          0, exp(-1j * h[b,j+1] * k[i, j+1])],dtype = npcomplex128).reshape((2,2))
    
                m_temp = m_temp @ d0_inv @ d1 @ p
    
            m[b, :, :, i] = m_temp

    return m
    
    
@jit(nopython = True, nogil = True, cache = True, parallel = parallel,boundscheck = debug,signature_or_function = complex128[:,:,:,:](float64,complex128[:,:],float64[:],float64[:,:]), debug = debug)
def detuning_transfer_matrix_TM(incidence_angle, n, wavelength, h):
    """
    Calculates the TM polarized transfer matrices of a device based on input
    wavelengths and an incidence angle for a range of different cavity lengths.

    Parameters
    ----------
    incidence_angle : float
        Angle under which the light is incident on the first layer.
    n : 2-D array of floats (W,N)
        Refractive indices in dependence on the W Wavelenghts and N Layers.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    wavelength : 1-D array of floats (W,)
        Contains the W wavelengths the function is evaluated at.
    h : 2-D array of floats (D,N)
        h[p,k] = Thickness of the k-th layer of the p-th cavity length.

    Returns
    -------
    m : 4-D array of complex floats (D,2,2,W)
        m[p,:,:,k] contains the (2,2)-transfer matrix for TM polarization at the
        k-th wavelength for the p-th cavity length

    """
    
    m = empty((h.shape[0], 2, 2, n.shape[0]),dtype = npcomplex128)
    
    k = empty_like(n)
    angle = empty_like(n)

    angle = calc_angle(incidence_angle, n)

    k = calc_k(wavelength, n, angle)
    for b in prange(h.shape[0]):
        
        for i in range(n.shape[0]):

            m_temp = array([1, 0, 0, 1],dtype = npcomplex128).reshape((2,2))
    
            for j in range(n.shape[1] - 1):
    
                d1 = array([cos(angle[i, j+1]), cos(angle[i, j+1]),
                            n[i, j+1], -n[i, j+1]],dtype = npcomplex128).reshape((2,2))          
                d0 = array([cos(angle[i, j]), cos(angle[i, j]),
                            n[i, j], -n[i, j]],dtype = npcomplex128).reshape((2,2))
                d0_inv = linalg.inv(d0)
                p = array([exp(1j * h[b,j+1] * k[i, j+1]), 0,
                          0, exp(-1j * h[b,j+1] * k[i, j+1])],dtype = npcomplex128).reshape((2,2))
    
                m_temp = m_temp @ d0_inv @ d1 @ p
    
            m[b,:, :, i] = m_temp

    return m
    


@jit(nopython = True, nogil = True, cache = True,parallel = parallel,signature_or_function = float64[:](complex128[:,:,:]),
     boundscheck = debug, debug = debug)
def reflectance(m):
    """
    Calculates the reflectance of an array of transfer matrices.

    Parameters
    ----------
    m : 3-D array of complex numbers (2,2,W)
        Contains the transfer matrices for all W wavelengths.
        The slice m[:,:,k] contains the (2,2) transfer matrix of the
        k-th wavelength.

    Returns
    -------
    r : 1-D array of real floats (W,)
        r[k] is the Reflectance of the device at the k-th wavelength.

    """

    r = empty((m.shape[2]),dtype = npreal64)

    r = power(absolute(m[1, 0, :] / m[0, 0, :]), 2)

    return r

@jit(nopython = True, nogil = True, cache = True,parallel = parallel,signature_or_function = complex128[:](complex128[:,:,:]),
     boundscheck = debug, debug = debug)
def reflection_amplitude(m):
    """
    Calculates the reflection amplitude of an array of transfer matrices.

    Parameters
    ----------
    m : 3-D array of complex numbers (2,2,W)
        Contains the transfer matrices for all W wavelengths.
        The slice m[:,:,k] contains the (2,2) transfer matrix of the
        k-th wavelength.

    Returns
    -------
    r : 1-D array of real floats (W,)
        r[k] is the Reflectance of the device at the k-th wavelength.

    """

    r = empty((m.shape[2]),dtype = npcomplex128)

    r = m[1, 0, :] / m[0, 0, :]

    return r


#
@jit(nopython = True, nogil = True, cache = True, parallel = parallel,signature_or_function = float64[:](complex128[:,:,:]),
     boundscheck = debug, debug = debug)
def transmittance(m):
    """
    Calculates the transmittance of an array of transfer matrices.

    Parameters
    ----------
    m : 3-D array of complex numbers (2,2,W)
        Contains the transfer matrices for all W wavelengths.
        The slice m[:,:,k] contains the (2,2) transfer matrix of the
        k-th wavelength.

    Returns
    -------
    t : 1-D array of real floats (W,)
        t[k] is the Transmittance of the device at the k-th wavelength.

    """

    t = empty((m.shape[2]),dtype = npreal64)

    t = power(absolute(1 / m[0, 0, :]), 2)

    return t

@jit(nopython = True, nogil = True, cache = True, parallel = parallel,signature_or_function = float64[:](complex128[:,:,:]),
     boundscheck = debug, debug = debug)
def stopband(m):
    """
    

    Parameters
    ----------
    m : 3-D array of complex numbers (2,2,W)
        Contains the transfer matrices for all W wavelengths.
        The slice m[:,:,k] contains the (2,2) transfer matrix of the
        k-th wavelength.

    Returns
    -------
    s : 1-D array of real floats (W,)
        s[k] is the average of matrix elements t_00 and t_11 of the transfer matrix.

    """
    
    s = empty((m.shape[2]),dtype = npreal64)
    
    s = real((m[0,0,:]+m[1,1,:])/2)
    
    return s
#
@jit(nopython = True, nogil = True, cache = True,parallel = True, boundscheck = debug,signature_or_function = float64[:,:,:](float64,complex128[:,:],float64[:],float64[:,:]), debug = debug)
def calc_1d_spectrum_detuning(incidence_angle, n, wavelength, h):
    """
    Calculates a 1-D spectrum of the Reflectance and Transmittance for TE and 
    TM polarized light incident on a device - in dependence of the wavelength
    and one incident angle and different detunings.


    Parameters
    ----------
    incidence_angle : float
        Light is incident on the structure. This is the incidence angles the
        spectrum is evaluated for.
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the W wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    wavelength : 1-D array of floats (W,)
        Wavelengths the spectrum is calculated at.
    h : 2-D array of floats (D,N)
        h[p,k] is the Thickness of the k-th layer for the p-th cavity length.

    Returns
    -------
    output : 3-D array of floats (6,W,D)
        The slices output[l,:,:] contain 1-D spectra in dependence of the 
        wavelength.

        output[0,:,:] is TE polarized Reflectance
        output[1,:,:] is TM polarized Reflectance
        output[2,:,:] is TE polarized Transmittance
        output[3,:,:] is TM polarized Transmittance
        output[4,:,:] is TE polarized real average of T_00 and T_11
        output[5,:,:] is TM polarized real average of T_00 and T_11

    """

    output = empty((6, wavelength.shape[0],h.shape[0]),dtype = npreal64)
    
    te = ones((h.shape[0],2,2,wavelength.shape[0]),dtype = npcomplex128)
    tm = ones((h.shape[0],2,2,wavelength.shape[0]),dtype = npcomplex128)

    te = detuning_transfer_matrix_TE(incidence_angle, n, wavelength, h)
    tm = detuning_transfer_matrix_TM(incidence_angle, n, wavelength, h)
    
    for b in prange(h.shape[0]):
        output[0, :, b] = reflectance(te[b,:,:,:])
        output[1, :, b] = reflectance(tm[b,:,:,:])
        output[2, :, b] = transmittance(te[b,:,:,:])
        output[3, :, b] = transmittance(tm[b,:,:,:])
        output[4, :, b] = stopband(te[b,:,:,:])
        output[5, :, b] = stopband(tm[b,:,:,:])

    return output


@jit(nopython = True, nogil = True, cache = True,parallel = True, boundscheck = debug,signature_or_function = complex128[:,:](float64,complex128[:,:],float64[:],float64[:]), debug = debug)
def calc_1d_reflection_amplitude(incidence_angle, n, wavelength, h):
    """
    Calculates the Reflection Amplitude for TE and 
    TM polarized light incident on a device - in dependence of the wavelength
    and one incident angle.


    Parameters
    ----------
    incidence_angle : float
        Light is incident on the structure. This is the incidence angles the
        spectrum is evaluated for.
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the W wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    wavelength : 1-D array of floats (W,)
        Wavelengths the spectrum is calculated at.
    h : 1-D array of floats (N,)
        h[k] is the Thickness of the k-th layer.

    Returns
    -------
    output : 2-D array of floats (6, W)
        The slices output[k,:] contain 1-D spectra in dependence of the 
        wavelength.

        output[1,:] is TE polarized Reflection Amplitude
        output[2,:] is TM polarized Reflection Amplitude


    """

    output = empty((6, wavelength.shape[0]),dtype = npcomplex128)
    
    te = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)
    tm = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    te = calc_transfer_matrix_TE(incidence_angle, n, wavelength, h)
    tm = calc_transfer_matrix_TM(incidence_angle, n, wavelength, h)
    
    output[0, :] = reflection_amplitude(te)
    output[1, :] = reflection_amplitude(tm)


    return output
    
    

@jit(nopython = True, nogil = True, cache = True,parallel = True, boundscheck = debug,signature_or_function = float64[:,:](float64,complex128[:,:],float64[:],float64[:]), debug = debug)
def calc_1d_spectrum(incidence_angle, n, wavelength, h):
    """
    Calculates a 1-D spectrum of the Reflectance and Transmittance for TE and 
    TM polarized light incident on a device - in dependence of the wavelength
    and one incident angle.


    Parameters
    ----------
    incidence_angle : float
        Light is incident on the structure. This is the incidence angles the
        spectrum is evaluated for.
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the W wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    wavelength : 1-D array of floats (W,)
        Wavelengths the spectrum is calculated at.
    h : 1-D array of floats (N,)
        h[k] is the Thickness of the k-th layer.

    Returns
    -------
    output : 2-D array of floats (6, W)
        The slices output[k,:] contain 1-D spectra in dependence of the 
        wavelength.

        output[1,:] is TE polarized Reflectance
        output[2,:] is TM polarized Reflectance
        output[3,:] is TE polarized Transmittance
        output[4,:] is TM polarized Transmittance
        output[5,:] is TE polarized real average of T_00 and T_11
        output[6,:] is TM polarized real average of T_00 and T_11

    """

    output = empty((6, wavelength.shape[0]),dtype = npreal64)
    
    te = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)
    tm = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    te = calc_transfer_matrix_TE(incidence_angle, n, wavelength, h)
    tm = calc_transfer_matrix_TM(incidence_angle, n, wavelength, h)
    
    output[0, :] = reflectance(te)
    output[1, :] = reflectance(tm)
    output[2, :] = transmittance(te)
    output[3, :] = transmittance(tm)
    output[4, :] = stopband(te)
    output[5, :] = stopband(tm)

    return output

@jit(nopython = True, nogil = True, cache = True,parallel = True, boundscheck = debug,signature_or_function = float64[:,:](float64,complex128[:,:],float64[:],float64[:]), debug = debug)
def calc_1d_EField(incidence_angle, n, wavelength, h):
    """


    Parameters
    ----------
    incidence_angle : float
        Light is incident on the structure. This is the incidence angles the
        spectrum is evaluated for.
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the W wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    wavelength : 1-D array of floats (W,)
        Wavelengths the spectrum is calculated at.
    h : 1-D array of floats (N,)
        h[k] is the Thickness of the k-th layer.

    Returns
    -------
    output : 2-D array of floats (6, W)
        The slices output[k,:] contain 1-D spectra in dependence of the 
        wavelength.

        output[1,:] is TE polarized Reflectance
        output[2,:] is TM polarized Reflectance
        output[3,:] is TE polarized Transmittance
        output[4,:] is TM polarized Transmittance
        output[5,:] is TE polarized real average of T_00 and T_11
        output[6,:] is TM polarized real average of T_00 and T_11

    """

    output = empty((6, wavelength.shape[0]),dtype = npreal64)
    
    te = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)
    tm = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    te = calc_transfer_matrix_TE(incidence_angle, n, wavelength, h)
    tm = calc_transfer_matrix_TM(incidence_angle, n, wavelength, h)
    
    output[0, :] = reflectance(te)
    output[1, :] = reflectance(tm)
    output[2, :] = transmittance(te)
    output[3, :] = transmittance(tm)
    output[4, :] = stopband(te)
    output[5, :] = stopband(tm)

    return output

#,
@jit(nopython = True, nogil = True, debug = debug, cache = True,parallel = True, boundscheck = debug,signature_or_function = float64[:,:,:](float64[:],complex128[:,:],float64[:],float64[:]))
def calc_2d_spectrum(incidence_angle, n, wavelength, h):
    """
    Calculates a 2-D spectrum of Reflectance and Transmittance for TE and TM 
    polarized light incident on a device - in dependence of the wavelength and
    incident angle.

    Parameters
    ----------
    incidence_angle : 1-D array of floats (A,)
        Light is incident on the structure. This 1-D array contains all
        incidence angles the spectrum is evaluated for.
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    wavelength : 1-D array of floats (W,)
        Wavelengths the spectrum is calculated at.
    h : 1-D array of floats (N,)
        h[k] is the Thickness of the k-th layer.

    Returns
    -------
    output : 3-D array of floats (4,W,A)
        The slices output[k,:,:] contain 2-D spectra in dependence of
        wavelength and incidence angle.

        output[k,:,:]:

        ------------> incident angles
        |
        |
        |
        |
        |
        wavelengths

        output[1,:,:] is TE polarized Reflectance
        output[2,:,:] is TM polarized Reflectance
        output[3,:,:] is TE polarized Transmittance
        output[4,:,:] is TM polarized Transmittance

    """

    output = empty((4, wavelength.shape[0], incidence_angle.shape[0]),dtype = npreal64)

    for o in prange(incidence_angle.shape[0]):
        
        te = calc_transfer_matrix_TE(incidence_angle[o], n, wavelength, h)
        tm = calc_transfer_matrix_TM(incidence_angle[o], n, wavelength, h)

        output[0, :, o] = reflectance(te)
        output[1, :, o] = reflectance(tm)
        output[2, :, o] = transmittance(te)
        output[3, :, o] = transmittance(tm)

    return output

@jit(nopython = True, nogil = True, cache = True,parallel = True, boundscheck = debug,signature_or_function = float64[:,:,:,:](float64[:],complex128[:,:],float64[:],float64[:,:]), debug = debug)
def calc_2d_spectrum_detuning(incidence_angle, n, wavelength, h):
    """
    Calculates a 2-D spectrum of the Reflectance and Transmittance for TE and 
    TM polarized light incident on a device - in dependence of the wavelength,
    incident angle and different detunings.


    Parameters
    ----------
    incidence_angle : 1-D array of floats (A,)
        Light is incident on the structure. These are the incidence angles the
        spectrum is evaluated for.
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the W wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    wavelength : 1-D array of floats (W,)
        Wavelengths the spectrum is calculated at.
    h : 2-D array of floats (D,N)
        h[p,k] is the Thickness of the k-th layer for the p-th cavity length.

    Returns
    -------
    output : 4-D array of floats (6, D, W, A)
        The slices output[p,k,:] contain 1-D spectra in dependence of the 
        wavelength.

        output[0,:,:,:] is TE polarized Reflectance
        output[1,:,:,:] is TM polarized Reflectance
        output[2,:,:,:] is TE polarized Transmittance
        output[3,:,:,:] is TM polarized Transmittance
        output[4,:,:,:] is TE polarized real average of T_00 and T_11
        output[5,:,:,:] is TM polarized real average of T_00 and T_11

    """

    output = ones((6,h.shape[0], wavelength.shape[0],incidence_angle.shape[0]),dtype = npreal64)


    for o in prange(incidence_angle.shape[0]):
        
        te = detuning_transfer_matrix_TE(incidence_angle[o], n, wavelength, h)
        tm = detuning_transfer_matrix_TM(incidence_angle[o], n, wavelength, h)

        for b in range(h.shape[0]):

            output[0, b, :, o] = reflectance(te[b,:,:,:])
            output[1, b, :, o] = reflectance(tm[b,:,:,:])
            output[2, b, :, o] = transmittance(te[b,:,:,:])
            output[3, b, :, o] = transmittance(tm[b,:,:,:])
            output[4, b, :, o] = stopband(te[b,:,:,:])
            output[5, b, :, o] = stopband(tm[b,:,:,:])

    return output


@jit(nopython = True, nogil = True, debug = debug, cache = True,parallel = parallel, boundscheck = debug)
def fit_func_spectrum_T_TE(wavelength,unique_h,param_map,incidence_angle, n):
    """
    Wrapper for calc_transfer_matrix_TE in case of a fit. Used to specify the
    independent variables.

    Parameters
    ----------
    wavelength : 1-D array of floats
        Wavelength-data of the measurement (x-values).
    unique_h : 1-D arra of floats
        Thicknesses of unique layers, in this case independent variables.
    param_map : 1-D array of floats
        Used to map unique layer information to the different layers of the 
        structure. param_map.size[0] = h.size[0] - 1 because air is excluded.
    incidence_angle : float
        Parameter, incident angle of light
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    output : 1-D array of floats (W,)
        Contains the wavelength spectrum of the device, TE-Polarized in
        Reflection
        
    """
    
    h = empty((param_map.size[0]),dtype = npreal64)
    h[0] = 0
    
    for k in range(1,param_map.size[0]+1):
        
        h[k] = unique_h[param_map[k]]


    output = empty((wavelength.shape[0]),dtype = npreal64)
    
    te = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    te = calc_transfer_matrix_TE(incidence_angle, n, wavelength, h)
    
    output = transmittance(te)

    return output



@jit(nopython = True, nogil = True, debug = debug, cache = True,parallel = parallel, boundscheck = debug)
def fit_func_spectrum_T_TM(wavelength,unique_h,param_map,incidence_angle, n):
    """
    Wrapper for calc_transfer_matrix_TE in case of a fit. Used to specify the
    independent variables.

    Parameters
    ----------
    wavelength : 1-D array of floats
        Wavelength-data of the measurement (x-values).
    unique_h : 1-D arra of floats
        Thicknesses of unique layers, in this case independent variables.
    param_map : 1-D array of floats
        Used to map unique layer information to the different layers of the 
        structure. param_map.size[0] = h.size[0] - 1 because air is excluded.
    incidence_angle : float
        Parameter, incident angle of light
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    output : 1-D array of floats (W,)
        Contains the wavelength spectrum of the device, TE-Polarized in
        Reflection
        
    """
    
    h = empty((param_map.size[0]),dtype = npreal64)
    h[0] = 0
    
    for k in range(1,param_map.size[0]+1):
        
        h[k] = unique_h[param_map[k]]


    output = empty((wavelength.shape[0]),dtype = npreal64)
    
    tm = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    tm = calc_transfer_matrix_TM(incidence_angle, n, wavelength, h)
    
    output = transmittance(tm)

    return output


@jit(nopython = True, nogil = True, debug = debug, cache = True,parallel = parallel, boundscheck = debug)
def fit_func_spectrum_T_TETM(wavelength,unique_h,param_map,incidence_angle, n):
    """
    Wrapper for calc_transfer_matrix_TE in case of a fit. Used to specify the
    independent variables.

    Parameters
    ----------
    wavelength : 1-D array of floats
        Wavelength-data of the measurement (x-values).
    unique_h : 1-D arra of floats
        Thicknesses of unique layers, in this case independent variables.
    param_map : 1-D array of floats
        Used to map unique layer information to the different layers of the 
        structure. param_map.size[0] = h.size[0] - 1 because air is excluded.
    incidence_angle : float
        Parameter, incident angle of light
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    output : 1-D array of floats (W,)
        Contains the wavelength spectrum of the device, TE-Polarized in
        Reflection
        
    """
    
    h = empty((param_map.size[0]),dtype = npreal64)
    h[0] = 0
    
    for k in range(1,param_map.size[0]+1):
        
        h[k] = unique_h[param_map[k]]


    output = empty((wavelength.shape[0]),dtype = npreal64)
    
    tm = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)
    te = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    tm = calc_transfer_matrix_TM(incidence_angle, n, wavelength, h)
    te = calc_transfer_matrix_TE(incidence_angle, n, wavelength, h)
    
    output = 1/2 * transmittance(tm) + 1/2 * transmittance(te)

    return output

@jit(nopython = True, nogil = True, debug = debug, cache = True,parallel = parallel, boundscheck = debug)
def fit_func_spectrum_R_TE(wavelength,unique_h,param_map,incidence_angle, n):
    """
    Wrapper for calc_transfer_matrix_TE in case of a fit. Used to specify the
    independent variables.

    Parameters
    ----------
    wavelength : 1-D array of floats
        Wavelength-data of the measurement (x-values).
    unique_h : 1-D arra of floats
        Thicknesses of unique layers, in this case independent variables.
    param_map : 1-D array of floats
        Used to map unique layer information to the different layers of the 
        structure. param_map.size[0] = h.size[0] - 1 because air is excluded.
    incidence_angle : float
        Parameter, incident angle of light
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    output : 1-D array of floats (W,)
        Contains the wavelength spectrum of the device, TE-Polarized in
        Reflection
        
    """
    
    h = ones((param_map.shape[0]+2,),dtype = npreal64)
    h[0] = 0
    h[-1] = 5
    for k in range(1,param_map.shape[0]+1):
        
        h[k] = unique_h[param_map[k-1]]

    # print(h)
    # print(h)
    output = ones((wavelength.shape[0]),dtype = npreal64)
    
    te = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    te = calc_transfer_matrix_TE(incidence_angle, n, wavelength, h)
    
    output = reflectance(te)

    return output



@jit(nopython = True, nogil = True, debug = debug, cache = True,parallel = parallel, boundscheck = debug)
def fit_func_spectrum_R_TM(wavelength,unique_h,param_map,incidence_angle, n):
    """
    Wrapper for calc_transfer_matrix_TE in case of a fit. Used to specify the
    independent variables.

    Parameters
    ----------
    wavelength : 1-D array of floats
        Wavelength-data of the measurement (x-values).
    unique_h : 1-D arra of floats
        Thicknesses of unique layers, in this case independent variables.
    param_map : 1-D array of floats
        Used to map unique layer information to the different layers of the 
        structure. param_map.size[0] = h.size[0] - 1 because air is excluded.
    incidence_angle : float
        Parameter, incident angle of light
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    output : 1-D array of floats (W,)
        Contains the wavelength spectrum of the device, TE-Polarized in
        Reflection
        
    """
    
    h = zeros((param_map.size[0]),dtype = npreal64)
    h[0] = 0
    h[-1] = 5
    for k in range(1,param_map.size[0]+1):
        
        h[k] = unique_h[param_map[k]]


    output = empty((wavelength.shape[0]),dtype = npreal64)
    
    tm = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    tm = calc_transfer_matrix_TM(incidence_angle, n, wavelength, h)
    
    output = reflectance(tm)

    return output


@jit(nopython = True, nogil = True, debug = debug, cache = True,parallel = parallel, boundscheck = debug)
def fit_func_spectrum_R_TETM(wavelength,unique_h,param_map,incidence_angle, n):
    """
    Wrapper for calc_transfer_matrix_TE in case of a fit. Used to specify the
    independent variables.

    Parameters
    ----------
    wavelength : 1-D array of floats
        Wavelength-data of the measurement (x-values).
    unique_h : 1-D arra of floats
        Thicknesses of unique layers, in this case independent variables.
    param_map : 1-D array of floats
        Used to map unique layer information to the different layers of the 
        structure. param_map.size[0] = h.size[0] - 1 because air is excluded.
    incidence_angle : float
        Parameter, incident angle of light
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    output : 1-D array of floats (W,)
        Contains the wavelength spectrum of the device, TE-Polarized in
        Reflection
        
    """
    
    h = zeros((param_map.size[0]+2),dtype = npreal64)
    h[0] = 0
    h[-1] = 5
    for k in range(1,param_map.size[0]+1):
        
        h[k] = unique_h[param_map[k-1]]


    output = empty((wavelength.shape[0]),dtype = npreal64)
    
    tm = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)
    te = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    tm = calc_transfer_matrix_TM(incidence_angle, n, wavelength, h)
    te = calc_transfer_matrix_TE(incidence_angle, n, wavelength, h)
    
    output = 1/2 * reflectance(tm) + 1/2 * reflectance(te)

    return output


@jit(nopython = True, nogil = True, debug = debug, cache = True, boundscheck = debug)
def fit_func_spectrum_ang_T_TE(wavelength,beta,param_map, n):
    """
    Wrapper for calc_transfer_matrix_TE in case of a fit. Used to specify the
    independent variables.

    Parameters
    ----------
    wavelength : 1-D array of floats
        Wavelength-data of the measurement (x-values).
    beta : 1-D arra of floats (B,)
        Independent variables, thicknesses of unique layers for the first B-1 
        entries, incident_angle for the last entry
    param_map : 1-D array of floats
        Used to map unique layer information to the different layers of the 
        structure. param_map.size[0] = h.size[0] - 1 because air is excluded.
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    output : 1-D array of floats (W,)
        Contains the wavelength spectrum of the device, TE-Polarized in
        Reflection
        
    """
    
    h = empty((param_map.size[0]),dtype = npreal64)
    h[0] = 0
    incidence_angle = beta[-1]
    for k in range(1,param_map.size[0]+1):
        
        h[k] = beta[param_map[k]]


    output = empty((wavelength.shape[0]),dtype = npreal64)
    
    te = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    te = calc_transfer_matrix_TE(incidence_angle, n, wavelength, h)
    
    output = transmittance(te)

    return output
@jit(nopython = True, nogil = True, debug = debug, cache = True, boundscheck = debug)
def fit_func_spectrum_ang_T_TM(wavelength,beta,param_map, n):
    """
    Wrapper for calc_transfer_matrix_TE in case of a fit. Used to specify the
    independent variables.

    Parameters
    ----------
    wavelength : 1-D array of floats
        Wavelength-data of the measurement (x-values).
    beta : 1-D arra of floats (B,)
        Independent variables, thicknesses of unique layers for the first B-1 
        entries, incident_angle for the last entry
    param_map : 1-D array of floats
        Used to map unique layer information to the different layers of the 
        structure. param_map.size[0] = h.size[0] - 1 because air is excluded.
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    output : 1-D array of floats (W,)
        Contains the wavelength spectrum of the device, TE-Polarized in
        Reflection
        
    """
    
    h = empty((param_map.size[0]),dtype = npreal64)
    h[0] = 0
    incidence_angle = beta[-1]
    for k in range(1,param_map.size[0]+1):
        
        h[k] = beta[param_map[k]]


    output = empty((wavelength.shape[0]),dtype = npreal64)
    
    tm = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    tm = calc_transfer_matrix_TM(incidence_angle, n, wavelength, h)
    
    output = transmittance(tm)

    return output


@jit(nopython = True, nogil = True, debug = debug, cache = True, boundscheck = debug)
def fit_func_spectrum_ang_T_TETM(wavelength,beta,param_map, n):
    """
    Wrapper for calc_transfer_matrix_TE in case of a fit. Used to specify the
    independent variables.

    Parameters
    ----------
    wavelength : 1-D array of floats
        Wavelength-data of the measurement (x-values).
    beta : 1-D arra of floats (B,)
        Independent variables, thicknesses of unique layers for the first B-1 
        entries, incident_angle for the last entry
    param_map : 1-D array of floats
        Used to map unique layer information to the different layers of the 
        structure. param_map.size[0] = h.size[0] - 1 because air is excluded.
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    output : 1-D array of floats (W,)
        Contains the wavelength spectrum of the device, TE-Polarized in
        Reflection
        
    """
    
    h = empty((param_map.size[0]),dtype = npreal64)
    h[0] = 0
    incidence_angle = beta[-1]
    for k in range(1,param_map.size[0]+1):
        
        h[k] = beta[param_map[k]]


    output = empty((wavelength.shape[0]),dtype = npreal64)
    
    tm = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)
    te = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    tm = calc_transfer_matrix_TM(incidence_angle, n, wavelength, h)
    tm = calc_transfer_matrix_TE(incidence_angle, n, wavelength, h)
    
    output = 1/2 * transmittance(tm) + 1/2 * transmittance(te)

    return output


@jit(nopython = True, nogil = True, debug = debug, cache = True, boundscheck = debug)
def fit_func_spectrum_ang_R_TE(wavelength,beta,param_map, n):
    """
    Wrapper for calc_transfer_matrix_TE in case of a fit. Used to specify the
    independent variables.

    Parameters
    ----------
    wavelength : 1-D array of floats
        Wavelength-data of the measurement (x-values).
    beta : 1-D arra of floats (B,)
        Independent variables, thicknesses of unique layers for the first B-1 
        entries, incident_angle for the last entry
    param_map : 1-D array of floats
        Used to map unique layer information to the different layers of the 
        structure. param_map.size[0] = h.size[0] - 1 because air is excluded.
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    output : 1-D array of floats (W,)
        Contains the wavelength spectrum of the device, TE-Polarized in
        Reflection
        
    """
    
    h = empty((param_map.size[0]),dtype = npreal64)
    h[0] = 0
    incidence_angle = beta[-1]
    for k in range(1,param_map.size[0]+1):
        
        h[k] = beta[param_map[k]]


    output = empty((wavelength.shape[0]),dtype = npreal64)
    
    te = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    te = calc_transfer_matrix_TE(incidence_angle, n, wavelength, h)
    
    output = reflectance(te)

    return output

@jit(nopython = True, nogil = True, debug = debug, cache = True, boundscheck = debug)
def fit_func_spectrum_ang_R_TM(wavelength,beta,param_map, n):
    """
    Wrapper for calc_transfer_matrix_TE in case of a fit. Used to specify the
    independent variables.

    Parameters
    ----------
    wavelength : 1-D array of floats
        Wavelength-data of the measurement (x-values).
    beta : 1-D arra of floats (B,)
        Independent variables, thicknesses of unique layers for the first B-1 
        entries, incident_angle for the last entry
    param_map : 1-D array of floats
        Used to map unique layer information to the different layers of the 
        structure. param_map.size[0] = h.size[0] - 1 because air is excluded.
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    output : 1-D array of floats (W,)
        Contains the wavelength spectrum of the device, TE-Polarized in
        Reflection
        
    """
    
    h = empty((param_map.size[0]),dtype = npreal64)
    h[0] = 0
    incidence_angle = beta[-1]
    for k in range(1,param_map.size[0]+1):
        
        h[k] = beta[param_map[k]]


    output = empty((wavelength.shape[0]),dtype = npreal64)
    
    tm = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    tm = calc_transfer_matrix_TM(incidence_angle, n, wavelength, h)
    
    output = reflectance(tm)

    return output


@jit(nopython = True, nogil = True, debug = debug, cache = True, boundscheck = debug)
def fit_func_spectrum_ang_R_TETM(wavelength,beta,param_map, n):
    """
    Wrapper for calc_transfer_matrix_TE in case of a fit. Used to specify the
    independent variables.

    Parameters
    ----------
    wavelength : 1-D array of floats
        Wavelength-data of the measurement (x-values).
    beta : 1-D arra of floats (B,)
        Independent variables, thicknesses of unique layers for the first B-1 
        entries, incident_angle for the last entry
    param_map : 1-D array of floats
        Used to map unique layer information to the different layers of the 
        structure. param_map.size[0] = h.size[0] - 1 because air is excluded.
    n : 2-D array of floats (W,N)
        Refractive indices of the device in dependence of the wavelengths
        the spectrum is evaluated at.

        ------------> device layers
        |
        |
        |
        |
        |
        wavelengths

    Returns
    -------
    output : 1-D array of floats (W,)
        Contains the wavelength spectrum of the device, TE-Polarized in
        Reflection
        
    """
    
    h = empty((param_map.size[0]),dtype = npreal64)
    h[0] = 0
    incidence_angle = beta[-1]
    for k in range(1,param_map.size[0]+1):
        
        h[k] = beta[param_map[k]]


    output = empty((wavelength.shape[0]),dtype = npreal64)
    
    tm = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)
    te = ones((2,2,wavelength.shape[0]),dtype = npcomplex128)

    tm = calc_transfer_matrix_TM(incidence_angle, n, wavelength, h)
    tm = calc_transfer_matrix_TE(incidence_angle, n, wavelength, h)
    
    output = 1/2 * reflectance(tm) + 1/2 * reflectance(te)

    return output

