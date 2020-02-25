#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def delay_f(x, time, fs):
    N = len(x)
    x_f = np.fft.fft(x)
    y_f = np.zeros(N, dtype=np.complex_)
    w = np.array(range(0, (N / 2) + 1) + range((-N / 2) + 1, 0)) / float(N) * float(fs)
    y_f = x_f * np.exp(-1j * 2 * np.pi * w * time)
    y = np.fft.ifft(y_f).real
    return y


def to_dB_spect(magnitude, MIN_AMP, AMP_FAC):
    magnitude = np.maximum(magnitude, np.max(magnitude) / MIN_AMP)
    magnitude = 20. * np.log10(magnitude * AMP_FAC)
    return magnitude


def to_dB_mag(magnitude, MIN_AMP, AMP_FAC):
    magnitude = np.maximum(magnitude, np.max(magnitude) / float(MIN_AMP))
    magnitude = 20. * np.log10(magnitude * AMP_FAC)
    return magnitude


def add_SIR_(s1, sir1):
    s1 = 10. ** (sir1 / 20.0) * s1

    if (s1.max() > 1.0):
        s1 = s1 * (1.0) / float(s1.max())

    if (s1.min() < -1.0):
        s1 = s1 * (-1.0) / float(s1.min())
    return s1
