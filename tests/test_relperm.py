import numpy as np
import pytest

from relperm import s_eff, krw


def test_s_eff_basic():
    sw = np.array([0.3, 0.5, 0.7])
    swr = 0.2
    snwr = 0.1
    result = s_eff(sw, swr, snwr)
    expected = np.array([0.142857, 0.428571, 0.714286])
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_krw_basic():
    s_eff_values = np.array([0.0, 0.5, 1.0])
    krw0 = 0.8
    nw = 2.0
    result = krw(s_eff_values, krw0, nw)
    expected = np.array([0.0, 0.2, 0.8])
    np.testing.assert_allclose(result, expected, rtol=1e-5)
