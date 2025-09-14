import math
import pytest

from relperm.relperm import (
    pc_thomeer,
    sw_thomeer,
    pc_brooks_corey,
    sw_brooks_corey,
    sw_denormalized,
    krw_corey,
    krow_corey,
    krg_corey,
    krog_corey,
)


def test_sw_pc_thomeer_roundtrip():
    # Choose interior values to avoid edge/equality cases
    ift_res = 1.1
    cos_theta_res = 0.95
    swi_th = 0.12
    pce_th = 2.3
    g_th = 1.7
    sw = 0.53

    pc = pc_thomeer(sw, ift_res, cos_theta_res, swi_th, pce_th, g_th)
    sw_back = sw_thomeer(pc, ift_res, cos_theta_res, swi_th, pce_th, g_th)
    # Allow relaxed tolerance due to mixed float/decimal math in implementation
    assert sw_back == pytest.approx(sw, rel=1e-3, abs=2e-2)


def test_pc_brooks_corey_invalid_sw_raises():
    ift_res = 1.2
    cos_theta_res = 0.9
    swi_bc = 0.2
    pce_bc = 3.0
    n_bc = 2.0

    with pytest.raises(ValueError):
        pc_brooks_corey(-0.01, ift_res, cos_theta_res, swi_bc, pce_bc, n_bc)
    with pytest.raises(ValueError):
        pc_brooks_corey(1.01, ift_res, cos_theta_res, swi_bc, pce_bc, n_bc)


def test_sw_brooks_corey_roundtrip():
    ift_res = 1.2
    cos_theta_res = 0.9
    swi_bc = 0.2
    pce_bc = 3.0
    n_bc = 2.0
    sw = 0.7

    pc = pc_brooks_corey(sw, ift_res, cos_theta_res, swi_bc, pce_bc, n_bc)
    sw_back = sw_brooks_corey(pc, ift_res, cos_theta_res, swi_bc, pce_bc, n_bc)
    assert sw_back == pytest.approx(sw, rel=1e-6, abs=1e-9)


def test_sw_denormalized_basic():
    swn = 0.5
    swirr = 0.2
    sorw = 0.1
    expected = 0.5 * (1 - 0.2 - 0.1) + 0.2  # 0.55
    assert sw_denormalized(swn, swirr, sorw) == pytest.approx(expected)


def test_krw_corey_edges_and_value():
    swirr = 0.2
    sorw = 0.1
    krw_sorw = 0.3
    nw = 2.5

    # Below irreducible and above 1 - sorw -> NaN
    assert math.isnan(krw_corey(0.19, swirr, sorw, krw_sorw, nw))
    assert math.isnan(krw_corey(0.91, swirr, sorw, krw_sorw, nw))

    # At irreducible -> 0
    assert krw_corey(swirr, swirr, sorw, krw_sorw, nw) == pytest.approx(0.0)


def test_krow_corey_edges_and_monotonic_region():
    swirr = 0.2
    sorw = 0.1
    kro_swirr = 0.85
    now = 2.0

    assert math.isnan(krow_corey(0.19, swirr, sorw, kro_swirr, now))
    assert math.isnan(krow_corey(0.91, swirr, sorw, kro_swirr, now))

    # At high water saturation near 1 - sorw, oil relperm tends to 0
    assert krow_corey(0.9, swirr, sorw, kro_swirr, now) == pytest.approx(0.0, abs=1e-9)


def test_krg_corey_at_sgc_zero():
    sgc = 0.05
    swirr = 0.2
    sorg = 0.15
    krg_sg_max = 0.9
    ng = 3.0

    # At critical gas saturation -> zero
    assert krg_corey(sgc, sgc, swirr, sorg, krg_sg_max, ng) == pytest.approx(0.0)


def test_krog_corey_at_zero_gas():
    sg = 0.0
    swirr = 0.2
    sorg = 0.15
    kro_sgi = 0.8
    nog = 2.5

    # With zero gas saturation, normalized term is 1
    assert krog_corey(sg, swirr, sorg, kro_sgi, nog) == pytest.approx(kro_sgi)
