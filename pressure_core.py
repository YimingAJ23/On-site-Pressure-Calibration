# pressure_core.py

import math
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import fabio
import h5py

import pyFAI
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import os
import datetime as dt


CALIBRANT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "Au": {
        "hkl_default": "111",
        "crystal_default": "fcc",
        "a0": 4.0786,                 # Å
        "k0": 166.65,                 # GPa
        "k0p": 5.4823,
        "eos_default": "BM3",                 
        "reference": "EoS source: O. L. Anderson et al. J. Appl. Phys. 65, 1534 (1989).",
    },
    "Cu": {
        "hkl_default": "111",
        "crystal_default": "fcc",
        "a0": 3.6150,                 # Å
        "k0": 134.42,                 # GPa
        "k0p": 5.19,
        "eos_default": "BM3",                 
        "reference": "EoS source: JCPDS 4-0836, Eos from Holzapfel.",
    },
    "Mo": {
        "hkl_default": "110",
        "crystal_default": "bcc",
        "a0": 3.1453,                 # Å
        "k0": 261.0,                  # GPa
        "k0p": 4.06,
        "eos_default": "VINET",               
        "reference": "EoS source: A. Dewaele et al. Phys. Rev. B 78, 104102 (2008).",
    },
    "Pt": {
        "hkl_default": "111",
        "crystal_default": "fcc",
        "a0": 3.9231,                 # Å
        "k0": 266.0,                  # GPa
        "k0p": 5.81,
        "eos_default": "BM3",               
        "reference": "EoS source: JCPDS 4-0802, EOS from Holmes.",
    },
    "Re": {
        "hkl_default": "100",
        "crystal_default": "hcp",
        "a0": 2.7619,                 # Å
        "c0": 4.4605,                 # Å
        "k0": 352.6,                  # GPa
        "k0p": 4.56,
        "eos_default": "VINET",               
        "reference": "EoS source: S. Anzellini et al. J. Appl. Phys. 115, 043511 (2014).",
    },
    "MgO": {
        "hkl_default": "111",
        "crystal_default": "fcc",
        "a0": 4.2130,                 # Å
        "k0": 160.2,                 # GPa
        "k0p": 3.99,
        "eos_default": "BM3",                 
        "reference": "EoS source: JCPDS 4-0829, EOS from Jackson & Niesler.",
    },
}


# =========================
# IO
# =========================
def load_image(path: str, h5_dset: Optional[str] = None) -> np.ndarray:
    if path.lower().endswith((".edf", ".tif", ".tiff", ".cbf", ".img", ".mar3450", ".mccd", ".pnm")):
        data = fabio.open(path).data
        data = np.flipud(data)
        return data.astype(np.float32)

    if path.lower().endswith((".h5", ".hdf5", ".nxs")):
        with h5py.File(path, "r") as f:
            if h5_dset:
                data = f[h5_dset][()]
            else:
                candidate = None

                def visitor(name, obj):
                    nonlocal candidate
                    if candidate is not None:
                        return
                    if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
                        shape = obj.shape
                        if shape[-2] > 50 and shape[-1] > 50:
                            candidate = name

                f.visititems(visitor)
                if candidate is None:
                    raise ValueError("No suitable 2D/3D dataset found in H5. Please specify H5 dataset path.")
                data = f[candidate][()]
            
            data = np.asarray(data)
            if data.ndim == 3:
                data = data[0]
            if data.ndim != 2:
                raise ValueError(f"Unsupported H5 dataset shape: {data.shape}")
            return data.astype(np.float32)

    raise ValueError(f"Unsupported data file type: {path}")


def read_mask(mask_path: str, expected_shape=None) -> np.ndarray:
    from PIL import Image
    m = np.array(Image.open(mask_path))
    m = (np.asarray(m) != 0)  # True means masked
    m = np.flipud(m)
    if expected_shape is not None and m.shape != expected_shape:
        raise ValueError(f"Mask shape {m.shape} != expected {expected_shape}")
    return m


# =========================
# Integration / peak fitting
# =========================
def integrate_1d(ai, img, mask, npt, unit="2th_deg"):
    res = ai.integrate1d(img, npt, mask=mask, unit=unit, method="csr", polarization_factor=0.990)
    x = np.asarray(res.radial, dtype=np.float64)
    y = np.asarray(res.intensity, dtype=np.float64)
    return x, y


def pseudo_voigt(x, area, x0, fwhm, eta, bkg0, bkg1):
    """
    Normalized pseudo-Voigt with 'area' parameterization.
    area: peak area
    x0: center
    fwhm: full width at half maximum
    eta: mixing factor [0..1], 0=Lorentzian, 1=Gaussian
    bkg0,bkg1: linear background
    """
    fwhm = max(float(fwhm), 1e-12)
    eta = min(max(float(eta), 0.0), 1.0)
    g = (2.0 * math.sqrt(math.log(2.0)) / (math.sqrt(math.pi) * fwhm)) * \
        np.exp(-4.0 * math.log(2.0) * ((x - x0) / fwhm) ** 2)
    l = (2.0 / (math.pi * fwhm)) * \
        (1.0 / (1.0 + 4.0 * ((x - x0) / fwhm) ** 2))
    peak = area * ((1.0 - eta) * l + eta * g)
    bkg = bkg0 + bkg1 * x
    return peak + bkg


def find_and_fit_first_peak(x, y, x_min=None, x_max=None) -> dict:
    if x_min is None:
        x_min = float(x.min())
    if x_max is None:
        x_max = float(x.max())

    sel = (x >= x_min) & (x <= x_max)
    xs = x[sel]
    ys = y[sel]

    baseline = np.percentile(ys, 10)
    ys0 = ys - baseline

    peaks, _ = find_peaks(ys0, height=np.percentile(ys0, 90))
    if len(peaks) == 0:
        raise RuntimeError("No peak found in the specified range.")

    idx = peaks[np.argmin(xs[peaks])]
    x0_init = float(xs[idx])
    h_init = float(max(ys0[idx], 1.0))

    fwhm_init = 0.1
    eta_init = 0.5
    C_g = math.sqrt(math.pi / (4.0 * math.log(2.0)))
    C_l = math.pi / 2.0 
    area_init = h_init * fwhm_init * (eta_init * C_l + (1.0 - eta_init) * C_g)  
    
    bkg0_init = float(baseline)
    bkg1_init = 0.0

    p0 = [area_init, x0_init, fwhm_init, eta_init, bkg0_init, bkg1_init]
    bounds = (
        [0, x_min, 1e-6, 0.0, -np.inf, -np.inf],
        [np.inf, x_max, (x_max - x_min), 1.0, np.inf, np.inf],
    )

    popt, _ = curve_fit(pseudo_voigt, xs, ys, p0=p0, bounds=bounds, maxfev=20000)

    yfit = pseudo_voigt(xs, *popt)
    resid = ys - yfit
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "x0": float(popt[1]),
        "fwhm": float(popt[2]),
        "eta": float(popt[3]),
        "area": float(popt[0]),
        "bkg0": float(popt[4]),
        "bkg1": float(popt[5]),
        "r2": float(r2),
        "fit_x": xs,
        "fit_y": ys,
        "fit_ycalc": yfit,
        "fit_min": float(xs.min()),
        "fit_max": float(xs.max()),
    }


# =========================
# HKL / lattice / EOS
# =========================
def parse_hkl(hkl: str) -> Tuple[int, int, int]:
    s = (hkl or "").strip()
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    s = s.replace(",", " ").replace("_", " ")

    parts = [p for p in s.split() if p]

    if len(parts) == 1:
        token = parts[0]
        if len(token) != 3 or (not token.lstrip("+-").isdigit()):
            raise ValueError(f"Invalid HKL: {hkl}. Use e.g. 111 or '1 1 1'.")
        # allow sign not supported in compact form; keep simple
        if token.startswith(("+", "-")):
            raise ValueError(f"Invalid compact HKL: {hkl}. Use spaced form like '-1 1 1'.")
        return int(token[0]), int(token[1]), int(token[2])

    if len(parts) == 3 and all(p.lstrip("+-").isdigit() for p in parts):
        return int(parts[0]), int(parts[1]), int(parts[2])

    raise ValueError(f"Invalid HKL: {hkl}. Use e.g. 111 or '1 1 1'.")


def tth_to_d_ang(tth_deg: float, wavelength_ang: float) -> float:
    theta = math.radians(tth_deg / 2.0)
    return wavelength_ang / (2.0 * math.sin(theta))


def v0_from_lattice(crystal: str, a0: float, c0: Optional[float] = None) -> float:
    crystal = (crystal or "").strip().lower()
    if crystal in ("bcc", "fcc"):
        return float(a0) ** 3
    if crystal == "hcp":
        if c0 is None:
            raise ValueError("hcp V0 requires c0.")
        return (math.sqrt(3.0) / 2.0) * (float(a0) ** 2) * float(c0)
    raise ValueError(f"Unknown crystal: {crystal}. Use bcc/fcc/hcp.")


def volume_from_d_general(
    crystal: str,
    hkl: Tuple[int, int, int],
    d_ang: float,
    a0_ang: Optional[float] = None,
    c0_ang: Optional[float] = None,
) -> float:
    crystal = (crystal or "").strip().lower()
    h, k, l = hkl

    if d_ang <= 0:
        raise ValueError("d must be positive.")

    if crystal in ("fcc", "bcc"):
        s = h * h + k * k + l * l
        if s <= 0:
            raise ValueError(f"Invalid HKL for cubic: {hkl}")
        a = d_ang * math.sqrt(s)
        return a ** 3

    if crystal == "hcp":
        if a0_ang is None or c0_ang is None:
            raise ValueError("hcp requires a0 and c0 (to fix c/a).")
        ca = float(c0_ang) / float(a0_ang)

        hk = h * h + h * k + k * k
        coeff = (4.0 / 3.0) * hk + (l * l) / (ca * ca)
        if coeff <= 0:
            raise ValueError(f"Invalid HKL for hcp: {hkl}")

        # With fixed c/a, solve for a
        a = d_ang * math.sqrt(coeff)
        c = ca * a
        v = (math.sqrt(3.0) / 2.0) * a * a * c
        return v

    raise ValueError(f"Unknown crystal: {crystal}. Use bcc/fcc/hcp.")

    
def bm3_pressure_from_v(v: float, v0: float, k0: float, k0p: float) -> float:
    if v <= 0 or v0 <= 0:
        raise ValueError("V and V0 must be positive.")
    eta = (v0 / v) ** (1.0 / 3.0)
    term1 = (eta**7 - eta**5)
    term2 = 1.0 + 0.75 * (k0p - 4.0) * (eta**2 - 1.0)
    return 1.5 * k0 * term1 * term2


def vinet_pressure_from_v(v: float, v0: float, k0: float, k0p: float) -> float:
    if v <= 0 or v0 <= 0:
        raise ValueError("V and V0 must be positive.")
    x = (v / v0) ** (1.0 / 3.0)
    return 3.0 * k0 * (1.0 - x) / (x * x) * math.exp(1.5 * (k0p - 1.0) * (1.0 - x))


def pressure_from_eos(v: float, v0: float, k0: float, k0p: float, eos_model: str) -> float:
    eos = (eos_model or "").strip().upper()
    if eos in ("BM", "BM3", "BIRCH", "BIRCH-MURNAGHAN"):
        return float(bm3_pressure_from_v(v=v, v0=v0, k0=k0, k0p=k0p))
    if eos in ("VINET", "V"):
        return float(vinet_pressure_from_v(v=v, v0=v0, k0=k0, k0p=k0p))
    raise ValueError(f"Unknown eos_model: {eos_model}. Use BM3 or VINET.")


def get_material_defaults(material: str) -> Dict[str, Any]:
    if material not in CALIBRANT_DEFAULTS:
        raise ValueError(f"Unknown material: {material}")
    cfg = dict(CALIBRANT_DEFAULTS[material])
    crystal = cfg.get("crystal_default", "fcc")
    a0 = cfg.get("a0", None)
    c0 = cfg.get("c0", None)
    if a0 is None:
        raise ValueError(f"Defaults for {material} missing a0.")
    cfg["v0_default"] = v0_from_lattice(crystal, a0, c0)
    return cfg


# =========================
# Plotting
# =========================
def default_fit_png_name(prefix="Fit", out_dir="results"):
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{prefix}_{ts}.png")


def save_fit_plot(
    out_png: str,
    tth_full: np.ndarray,
    I_full: np.ndarray,
    fit: dict,
    d_ang: float,
    p_gpa: float,
    material: str,
    crystal: str,
    hkl_str: str,
):
    fit_x = fit["fit_x"]
    fit_y = fit["fit_y"]
    fit_ycalc = fit["fit_ycalc"]
    tth0 = fit["x0"]
    fit_min = fit["fit_min"]
    fit_max = fit["fit_max"]
    r2 = fit["r2"]

    resid = fit_y - fit_ycalc

    sel_zoom = (tth_full >= fit_min) & (tth_full <= fit_max)
    tth_zoom = tth_full[sel_zoom]
    I_zoom = I_full[sel_zoom]

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    fig = plt.figure(figsize=(7, 7), dpi=150)
    gs = GridSpec(
        3, 1,
        height_ratios=[2.0, 2.5, 1.0],   # ratio of [full / zoom / residual] subplot
        hspace=0.15
    )

    # Panel 1: full pattern
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(tth_full, I_full, lw=0.8, color="black", alpha=0.8)
    ax0.axvline(tth0, ls="--", lw=1.0, color="red", label="Peak position")
    ax0.set_ylabel("Intensity (a.u.)")
    ax0.set_title(
        f"{material}  ({crystal})  HKL={hkl_str}   |   "
        f"2θ={tth0:.4f}°   d={d_ang:.6f} Å   P={p_gpa:.3f} GPa"
    )
    ax0.legend(loc="upper right")

    # Panel 2: zoom + dense fit line
    ax1 = fig.add_subplot(gs[1])
    x_dense = np.linspace(fit["fit_min"], fit["fit_max"], 1500)
    y_dense = pseudo_voigt(
        x_dense,
        fit["area"],
        fit["x0"],
        fit["fwhm"],
        fit["eta"],
        fit["bkg0"],
        fit["bkg1"],
    )
    ax1.scatter(tth_zoom, I_zoom, s=10, color="black", label="Obs. data")
    ax1.plot(x_dense, y_dense, lw=2.0, color="red", label="Pseudo-Voigt fit")
    ax1.axvline(tth0, ls="--", lw=1.2, color="tab:blue", label=f"Peak @ {tth0:.4f}°")

    ax1.set_xlim(fit_min, fit_max)
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.legend(loc="upper left", frameon=True)

    info = (
        f"FWHM = {fit['fwhm']:.5f}°\n"
        f"η = {fit['eta']:.3f}\n"
        f"R² = {r2:.4f}"
    )
    ax1.text(
        0.99, 0.97,
        info,
        transform=ax1.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
            linewidth=0.5,
        ),
    )

    # Panel 3: residual
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax2.plot(fit_x, resid, lw=1, color="black")
    ax2.axhline(0.0, lw=1, ls="--", color="gray")
    ax2.set_xlabel("2θ (deg)")
    ax2.set_ylabel("Residual")

    rmax = np.nanpercentile(np.abs(resid), 99)
    if rmax > 0:
        ax2.set_ylim(-1.1 * rmax, 1.1 * rmax)

    fig.savefig(out_png)
    plt.close(fig)


# =========================
# Main API (called by GUI)
# =========================   
def run_pressure_calibration(
    poni_path: str,
    data_path: str,
    mask_path: Optional[str],
    h5_dset: Optional[str],
    material: str,
    hkl_str: str,
    crystal: str,
    fit_min: Optional[float],
    fit_max: Optional[float],
    eos_model: str,          
    b0: Optional[float],         
    b0p: Optional[float],        
    v0: Optional[float],
    npt: int = 2000,    
) -> dict:

    if material not in CALIBRANT_DEFAULTS:
        raise ValueError(f"Unsupported material: {material}")
        
    cfg = get_material_defaults(material)

    # wavelength from PONI
    ai = pyFAI.load(poni_path)
    wl = getattr(ai, "wavelength", None)
    if wl is None:
        raise ValueError("PONI has no wavelength.")
    wl_ang = wl * 1e10  # m -> Å

    # load image
    img = load_image(data_path, h5_dset)

    # load mask
    mask = None
    if mask_path:
        mask = read_mask(mask_path, expected_shape=img.shape)

    # integrate
    tth, inten = integrate_1d(ai, img, mask, npt=npt, unit="2th_deg")

    # fit window auto guess
    if fit_min is None or fit_max is None:
        fit0 = find_and_fit_first_peak(tth, inten, None, None)
        tth0_guess = fit0["x0"]
        fit_min = tth0_guess - 1.0
        fit_max = tth0_guess + 1.0
    
    fit = find_and_fit_first_peak(tth, inten, fit_min, fit_max)
    tth0 = float(fit["x0"])
    d = float(tth_to_d_ang(tth0, wl_ang))

    # HKL / crystal defaults if empty
    hkl_use_str = (hkl_str or "").strip() or cfg["hkl_default"]
    crystal_use = (crystal or "").strip().lower() or cfg["crystal_default"]
    hkl_use = parse_hkl(hkl_use_str)

    # EOS default if empty
    eos_use = (eos_model or "").strip().upper() or str(cfg.get("eos_default", "BM3")).strip().upper()

    # params: user input overrides defaults
    k0_use = float(b0) if b0 is not None else float(cfg["k0"])
    k0p_use = float(b0p) if b0p is not None else float(cfg["k0p"])

    if v0 is not None:
        v0_use = float(v0)
    else:
        # compute from lattice, but using selected crystal type
        a0 = float(cfg["a0"])
        c0 = float(cfg["c0"]) if ("c0" in cfg) else None
        v0_use = v0_from_lattice(crystal_use, a0, c0)

    # volume from measured d
    a0 = float(cfg["a0"])
    c0 = float(cfg["c0"]) if ("c0" in cfg) else None
    v = float(volume_from_d_general(crystal_use, hkl_use, d_ang=d, a0_ang=a0, c0_ang=c0))

    # pressure
    p_gpa = float(pressure_from_eos(v=v, v0=v0_use, k0=k0_use, k0p=k0p_use, eos_model=eos_use))

    out_png = default_fit_png_name(prefix=f"{material}_fit", out_dir="results")
    save_fit_plot(
        out_png=out_png,
        tth_full=tth,
        I_full=inten,
        fit=fit,
        d_ang=d,
        p_gpa=p_gpa,
        material=material,
        crystal=crystal_use,
        hkl_str=hkl_use_str,
    )

    return {
        "material": material,
        "crystal": crystal_use,
        "hkl": hkl_use_str,
        "eos_model": eos_use,
        "b0": float(k0_use),
        "b0p": float(k0p_use),
        "v0": float(v0_use),
        "tth": float(tth0),
        "d": d,
        "p_gpa": float(p_gpa),
        "fwhm": float(fit["fwhm"]),
        "r2": float(fit["r2"]),
        "reference": str(cfg.get("reference", "")),
        "plot_png": out_png,
    }