from pathlib import Path
import numpy as np
import jax.numpy as jnp
from jax import jit
from das import das
from paths import time_of_flight
from scipy.io import loadmat, savemat
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from jaxopt import OptaxSolver
import optax
from losses import *
import time


N_ITERS = 301
LEARNING_RATE = 10
ASSUMED_C = 1540  # [m/s]

# B-mode limits in m
BMODE_X_MIN = -12e-3
BMODE_X_MAX = 12e-3
BMODE_Z_MIN = 0e-3
BMODE_Z_MAX = 40e-3

# Sound speed grid in m
SOUND_SPEED_X_MIN = -12e-3
SOUND_SPEED_X_MAX = 12e-3
SOUND_SPEED_Z_MIN = 0e-3
SOUND_SPEED_Z_MAX = 40e-3
SOUND_SPEED_NXC = 19
SOUND_SPEED_NZC = 31

# Phase estimate kernel size in samples
NXK, NZK = 5, 5

# Phase estimate patch grid size in samples
NXP, NZP = 17, 17
PHASE_ERROR_X_MIN = -20e-3
PHASE_ERROR_X_MAX = 20e-3
PHASE_ERROR_Z_MIN = 4e-3
PHASE_ERROR_Z_MAX = 44e-3

# Loss options
# -"pe" for phase error
# -"sb" for speckle brightness
# -"cf" for coherence factor
# -"lc" for lag one coherence

LOSS = "pe"

# Data options:
# (Constant Phantoms)
# - 1420
# - 1465
# - 1480
# - 1510
# - 1540
# - 1555
# - 1570
# (Heterogeneous Phantoms)
# - inclusion
# - inclusion_layer
# - four_layer
# - two_layer
# - checker2
# - checker8

SAMPLE = "1570"

CTRUE = {
    "1420": 1420,
    "1465": 1465,
    "1480": 1480,
    "1510": 1510,
    "1540": 1540,
    "1555": 1555,
    "1570": 1570,
    "inclusion": 0,
    "inclusion_layer": 0,
    "four_layer": 0,
    "two_layer": 0,
    "checker2": 0,
    "checker8": 0
}


# Refocused plane wave datasets from base dataset directory
DATA_DIR = Path("./data")


def imagesc(xc, y, img, dr, **kwargs):
    """MATLAB style imagesc"""
    dx = xc[1] - xc[0]
    dy = y[1] - y[0]
    ext = [xc[0] - dx / 2, xc[-1] + dx / 2, y[-1] + dy / 2, y[0] - dy / 2]
    im = plt.imshow(img, vmin=dr[0], vmax=dr[1], extent=ext, **kwargs)
    plt.colorbar()
    return im

def load_dataset(sample):
    mdict = loadmat(f"{DATA_DIR}/{sample}.mat")
    iqdata = mdict["iqdata"]
    fs = mdict["fs"][0, 0]  # Sampling frequency
    fd = mdict["fd"][0, 0]  # Demodulation frequency
    dsf = mdict["dsf"][0, 0]  # Downsampling factor
    t = mdict["t"]  # time vector
    t0 = mdict["t0"]  # time zero of transmit
    elpos = mdict["elpos"]  # element position
    return iqdata, t0, fs, fd, elpos, dsf, t


def plot_errors_vs_sound_speeds(c0, dlc, dl7c, dslsc, dpe, sample):
    plt.clf()
    plt.plot(c0, dlc, label="Lag One Coherence")
    plt.plot(c0, dl7c, label="Lag 7 Coherence")
    plt.plot(c0, dslsc, label="SLSC M=7")
    # divided by 10 for visualization
    plt.plot(c0, dpe / 10, label="Phase Error")
    plt.grid()
    plt.axvline(x=CTRUE[SAMPLE], color='black', linestyle=':')
    plt.xlabel("Global sound speed (m/s)")
    plt.ylabel("Loss function")
    plt.title(sample)
    plt.legend()
    plt.savefig(f"images/losses_coherence_{sample}_v2.png")
    plt.clf()


def main(sample, loss_name):

    assert (
        sample in CTRUE
    ), f'The data sample string was "{sample}".\
                            \nOptions are {", ".join(CTRUE.keys()).lstrip(" ,")}.'

    # Get IQ data, time zeros, sampling and demodulation frequency, and element positions
    iqdata, t0, fs, fd, elpos, _, _ = load_dataset(sample)
    xe, _, ze = jnp.array(elpos)
    wl0 = ASSUMED_C / fd  # wavelength (λ)

    # B-mode image dimensions
    xi = jnp.arange(BMODE_X_MIN, BMODE_X_MAX, wl0 / 3)
    zi = jnp.arange(BMODE_Z_MIN, BMODE_Z_MAX, wl0 / 3)
    nxi, nzi = xi.size, zi.size
    xi, zi = np.meshgrid(xi, zi, indexing="ij")

    # Sound speed grid dimensions
    xc = jnp.linspace(SOUND_SPEED_X_MIN, SOUND_SPEED_X_MAX, SOUND_SPEED_NXC)
    zc = jnp.linspace(SOUND_SPEED_Z_MIN, SOUND_SPEED_Z_MAX, SOUND_SPEED_NZC)
    dxc, dzc = xc[1] - xc[0], zc[1] - zc[0]

    # Kernels to use for loss calculations (2λ x 2λ patches)
    xk, zk = np.meshgrid((jnp.arange(NXK) - (NXK - 1) / 2) * wl0 / 2,
                         (jnp.arange(NZK) - (NZK - 1) / 2) * wl0 / 2,
                         indexing="ij")

    # Kernel patch centers, distributed throughout the field of view
    xpc, zpc = np.meshgrid(
        np.linspace(PHASE_ERROR_X_MIN, PHASE_ERROR_X_MAX, NXP),
        np.linspace(PHASE_ERROR_Z_MIN, PHASE_ERROR_Z_MAX, NZP),
        indexing="ij")

    # Explicit broadcasting. Dimensions will be [elements, pixels, patches]
    xe = jnp.reshape(xe, (-1, 1, 1))
    ze = jnp.reshape(ze, (-1, 1, 1))
    xp = jnp.reshape(xpc, (1, -1, 1)) + jnp.reshape(xk, (1, 1, -1))
    zp = jnp.reshape(zpc, (1, -1, 1)) + jnp.reshape(zk, (1, 1, -1))
    xp = xp + 0 * zp  # Manual broadcasting
    zp = zp + 0 * xp  # Manual broadcasting

    # Compute time-of-flight for each {image, patch} pixel to each element
    def tof_image(c): return time_of_flight(
        xe, ze, xi, zi, xc, zc, c, fnum=0.5, npts=64)

    def tof_patch(c): return time_of_flight(
        xe, ze, xp, zp, xc, zc, c, fnum=0.5, npts=64)

    def makeImage(c):
        t = tof_image(c)
        return jnp.abs(das(iqdata, t - t0, t, fs, fd))

    def loss_wrapper(func, c):
        t = tof_patch(c)
        return (func)(iqdata, t - t0, t, fs, fd)
    
    def loss_wrapper_new(func, c, m=1):
        t = tof_patch(c)
        return (func)(iqdata, t - t0, t, fs, fd, m)

    # Define loss functions
    lc_loss = jit(lambda c: 1 - jnp.mean(loss_wrapper(lag_one_coherence, c)))
    lmc_loss = jit(lambda c: 1 - jnp.mean(loss_wrapper_new(lag_m_coherence, c, m=7)))
    slscm_loss = jit(lambda c: 1 - jnp.mean(loss_wrapper_new(slsc_m, c, m=7)))

    @jit
    def pe_loss(c):
        t = tof_patch(c)
        dphi = phase_error(iqdata, t - t0, t, fs, fd)
        valid = dphi != 0
        dphi = jnp.where(valid, jnp.where(valid, dphi, jnp.nan), jnp.nan)
        return jnp.nanmean(jnp.log1p(jnp.square(100 * dphi)))

    tv = jit(lambda c: total_variation(c) * dxc * dzc)

    def loss(c):
        if loss_name == "lc":  # Lag one coherence
            return lc_loss(c) + tv(c) * 1e2
        elif loss_name == "lmc":  # Lag 7 coherence
            return lmc_loss(c) + tv(c) * 1e2
        elif loss_name == "slsc":  # slsc Lag 7
            return slscm_loss(c) + tv(c) * 1e2
        elif loss_name == "pe":  # Phase error
            return pe_loss(c) + tv(c) * 1e2
        else:
            NotImplementedError

    # Initial survey of losses vs. global sound speed
    c = ASSUMED_C * jnp.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC))

    # find optimal global sound speed for initalization
    c0 = np.linspace(1340, 1740, 201)
    dlc = np.array(
        [lc_loss(cc * jnp.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC))) for cc in c0])
    dlmc = np.array(
        [lmc_loss(cc * jnp.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC))) for cc in c0])
    dslsc = np.array(
        [slscm_loss(cc * jnp.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC))) for cc in c0])
    dpe = np.array(
        [pe_loss(cc * jnp.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC))) for cc in c0])

    # Use the sound speed with the optimal phase error to initialize sound speed map
    c = c0[np.argmin(dpe)] * jnp.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC))

    # Plot global sound speed error
    plot_errors_vs_sound_speeds(c0, dlc, dlmc, dslsc, dpe, sample)

    return c


if __name__ == "__main__":
    main(SAMPLE, LOSS)

    # # Run all examples
    # for sample in CTRUE.keys():
    #     print(sample)
    #     main(sample, LOSS)

