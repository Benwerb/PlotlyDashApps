"""
Calculate SBE63 oxygen concentration for NAVIS profiling float message files.
Given the required raw data as a matrix and a calibration structure.
Data is corrected for salt and pressure.

This function is a Python conversion of Calc_SBE63_O2.m

INPUTS:
    data -- an n x 5 matrix [CTD P, CTD T, CTD S, O2Phase, O2 Temp volts]
    cal  -- a dictionary containing calibration coefficients:
            cal['TempCoef']    [TA0 TA1 TA2 TA3]
            cal['PhaseCoef']   [A0 A1 A2 B0 B1 C0 C1 C2]

OUTPUT:
    O2_uM -- Oxygen concentration in μmol/L
    O2_T  -- O2 sensor Temperature

HISTORY:
    05/30/2017 - Implemented Henry Bittig's (2015) pressure correction
       scheme. This makes a correction in raw phase and in [O2].
       http://dx.doi.org/10.1175/JTECH-D-15-0108.1
"""

import numpy as np


def calc_sbe63_o2(data, cal):
    """
    Calculate SBE63 oxygen concentration.
    
    Parameters
    ----------
    data : numpy.ndarray
        n x 5 matrix [CTD P, CTD T, CTD S, O2Phase, O2 Temp volts]
    cal : dict
        Dictionary containing calibration coefficients:
        - 'TempCoef': array-like [TA0, TA1, TA2, TA3]
        - 'PhaseCoef': array-like [A0, A1, A2, B0, B1, C0, C1, C2]
    
    Returns
    -------
    O2_uM : numpy.ndarray
        Oxygen concentration in μmol/L
    O2_T : numpy.ndarray
        O2 sensor Temperature in degrees C
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    
    # ************************************************************************
    # BREAK OUT COEFFICIENTS AND DATA, SET SOME CONSTANTS
    Tcf = np.asarray(cal['TempCoef'])  # TA0-3, Temperature coefficients
    Acf = np.asarray(cal['PhaseCoef'][0:3])  # A0, A1, A2 Phase Coef's
    Bcf = np.asarray(cal['PhaseCoef'][3:5])  # B0, B1 Phase Coef's
    Ccf = np.asarray(cal['PhaseCoef'][5:8])  # C0, C1, C2, Ksv Coef's
    
    P = data[:, 0]  # CTD pressure
    T = data[:, 1]  # CTD temperature
    S = data[:, 2]  # CTD salinity
    OPh = data[:, 3]  # SBE63 phase delay
    OTV = data[:, 4]  # SBE63 temperature in volts
    
    Sref = 0  # instrument usually set to 0 salinity
    # E = 0.011  # Pressure correction coefficient (old value)
    E = 0.009  # Reassessed value in Processing ARGO O2 V2.2
    
    # ************************************************************************
    # CALCULATE OPTODE WINDOW TEMPERATURE (code from Dan Quittman)
    # Volts to resistance, np.log is natural log in Python
    L = np.log(100000 * OTV / (3.3 - OTV))
    denom = Tcf[0] + Tcf[1] * L + Tcf[2] * L**2 + Tcf[3] * L**3
    O2_T = 1.0 / denom - 273.15  # window temperature (degrees C)
    
    # ************************************************************************
    # CALCULATE OXYGEN FROM PHASE, ml/L (modified from Dan Quittman)
    CALC_T = O2_T  # Could also use CTD T, but SBE63 is pumped
    
    # Bittig pressure correction (2015) Part 1, T & O2 independent 05/30/2017
    OPh = OPh + (0.115 * P / 1000)
    
    V = OPh / 39.4570707
    A = Acf[0] + Acf[1] * CALC_T + Acf[2] * V**2
    B = Bcf[0] + Bcf[1] * V
    Ksv = Ccf[0] + Ccf[1] * CALC_T + Ccf[2] * CALC_T**2
    O2 = (A / B - 1) / Ksv  # ml/L, Salinity = 0, pressure = 0
    
    # ************************************************************************
    # CALCULATE SALT CORRECTION
    Ts = np.log((298.15 - CALC_T) / (273.15 + CALC_T))  # Scaled T
    # Descending powers for numpy polyval (same as MATLAB polyfit)
    SolB = np.array([-8.17083e-3, -1.03410e-2, -7.37614e-3, -6.24523e-3])
    C0 = -4.88682e-7
    
    Scorr = np.exp((S - Sref) * np.polyval(SolB, Ts) + C0 * (S**2 - Sref**2))
    
    # ************************************************************************
    # CALCULATE PRESSURE CORRECTION
    # PP = (P > 0) * P  # Clamp negative values to zero
    # pcorr = np.exp(E * PP / (CALC_T + 273.15))
    
    # SWITCH TO PRESSURE CORRECTION FROM BITTIG ET AL., (2015) - 05/30/2017
    pcorr = (0.00022 * T + 0.0419) * P / 1000 + 1
    
    # ************************************************************************
    # FINAL OUTPUT
    O2_uM = O2 * Scorr * pcorr * 44.6596  # OXYGEN μM/L
    
    return O2_uM, O2_T

