"""
This is a refactored version of the original python code of Synaptogen

Major differences:
 - replacing unicode names by ascii names
 - additional typing and docstrings
 - removed unneeded ADC and matmul function
"""

import json
import os

from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
from numpy import polyval, sqrt, float32, abs


def array32(x):
    return np.array(x, dtype=float32)


def zeros32(n):
    return np.zeros(n, dtype=float32)


def empty32(n):
    return np.empty(n, dtype=float32)


rng = np.random.default_rng()
randn = partial(rng.standard_normal, dtype=float32)
rand = partial(rng.random, dtype=float32)

Uread = float32(0.2)
e = float32(1.602176634e-19)
kBT = float32(1.380649e-23 * 300)
# sigmaClip = float32(3.5)
HRS_INDEX, US_INDEX, LRS_INDEX, UR_INDEX = 0, 1, 2, 3

moduledir = os.path.dirname(__file__)
default_param_fp = os.path.join(moduledir, "default_params.json")


def gamma_f(gamma, x):
    y = np.zeros_like(x)
    for gammav in gamma.T:
        y = y * x + gammav[:, np.newaxis]  # for broadcasting to work..
    return y


def psi(mu, sigma, x):
    y = mu + sigma * x
    y[HRS_INDEX, :] = 10 ** y[HRS_INDEX, :]
    return y


@dataclass
class CellParams:
    Umax: float  # Maximum voltage applied during the experiment.  defines the point where HRS is reached.
    U0: float  # Voltage used in the definition of resistance R = U₀ / I(U₀)
    eta: float  # Sets the curvature of the reset transition
    nfeatures: int  # Number of features for the VAR model
    p: int  # Order of the VAR model (how many cycles of history remembered)
    K: int  # How many components in the GMM for modeling device-to-device distribution
    gamma_deg: int  # Degree of the non-linear transformation polynomials
    G_HHRS: float  # Conductance of the HHRS
    G_LLRS: float  # Conductance of the LLRS
    HHRSdeg: int  # Degree of the HHRS polynomial
    LLRSdeg: int  # Degree of the LLRS polynomial
    HHRS: np.ndarray  # HHRS coefficients.
    LLRS: np.ndarray  # LLRS coefficients
    gamma: np.ndarray  # non-linear transformation coefficients
    wk: np.ndarray  # weights of the GMM components
    mu_DtD: np.ndarray  # mean vectors for the GMM
    LDtD: (
        np.ndarray
    )  # Cholesky decomposition of covariance matrices for the GMM (lower triangular)
    VAR: np.ndarray  # VAR coefficients, including A and B


class CellArray:
    M: int  # scalar      (number of cells)
    Xhat: np.ndarray  # 4(p+1) × M  (feature history and εₙ for all cells)
    # Xbuf : np.ndarray             # 4(p+1) × M  (buffer to improve the speed of the partial shift operation)
    x: (
        np.ndarray
    )  # 4 × M       (generated normal feature vectors ̂x*ₙ, basically also a buffer)
    sigma: np.ndarray  # 4 × M       (CtC scale vectors)
    mu: np.ndarray  # 4 × M       (CtC offset vectors)
    y: np.ndarray  # 4 × M       (scaled feature vector)
    r: np.ndarray  # M × 1       (device state variables)
    n: np.ndarray  # M × 1       (cycle numbers)
    k: np.ndarray  # M × 1       (GMM component, not strictly necessary to store)
    UR: np.ndarray  # M × 1       (voltage thresholds for reset switching)
    # Umax : np.ndarray              # M × 1       (Vector of Umax, probably all the same value, just for vectorization of polyval)
    resetCoefs: (
        np.ndarray
    )  # M × 2       (polynomial coefficients for reset transitions)
    Iread: np.ndarray  # M × 1       (readout buffer)
    inHRS: (
        np.ndarray
    )  # Using BitVector does not save much memory and isn't faster either.
    inLRS: np.ndarray
    setMask: np.ndarray
    resetMask: np.ndarray
    fullResetMask: np.ndarray
    partialResetMask: np.ndarray
    resetCoefsCalcMask: np.ndarray
    drawVARMask: np.ndarray
    params: CellParams

    def __init__(self, M, params: CellParams):
        """

        :param M:
        :param params:
        """
        self.params = params
        self.M = M

        self.Xhat = zeros32((params.nfeatures * (params.p + 1), M))
        # Xhat[:nfeatures, :] = randn((nfeatures, M))
        randn((params.nfeatures, M), out=self.Xhat[: params.nfeatures, :])
        self.x = params.VAR @ self.Xhat
        self.Xhat[-params.nfeatures :, :] = self.x
        cs = np.cumsum(params.wk) / np.sum(params.wk)
        self.k = np.searchsorted(cs, rand(M))
        mu_sigma_CtC = empty32((params.nfeatures * 2, M))
        for kk in range(len(params.wk)):
            mask = self.k == kk
            Mk = np.sum(mask)
            mu_sigma_CtC[:, mask] = params.mu_DtD[:, kk, np.newaxis] + params.LDtD[
                :, :, kk
            ] @ randn((params.nfeatures * 2, Mk))
        self.mu = mu_sigma_CtC[: params.nfeatures, :]
        self.sigma = mu_sigma_CtC[params.nfeatures :, :]
        self.y = psi(self.mu, self.sigma, gamma_f(params.gamma, self.x))
        self.resetCoefs = empty32((2, M))
        self.r = (params.G_LLRS - (1 / self.y[HRS_INDEX, :])) / (
            params.G_LLRS - params.G_HHRS
        )
        self.n = np.zeros(M, dtype=np.int64)
        # Todo: this needs better naming, overlaps with self.get_UR
        self.UR = self.y[UR_INDEX, :]
        # Umax = np.repeat(Umax, M)
        self.Iread = zeros32(M)
        self.inHRS = np.zeros(M, dtype=bool)
        self.inLRS = np.zeros(M, dtype=bool)
        self.setMask = np.zeros(M, dtype=bool)
        self.resetMask = np.zeros(M, dtype=bool)
        self.fullResetMask = np.zeros(M, dtype=bool)
        self.partialResetMask = np.zeros(M, dtype=bool)
        self.resetCoefsCalcMask = np.zeros(M, dtype=bool)
        self.drawVARMask = np.zeros(M, dtype=bool)

    def __len__(self):
        return self.M

    def Imix(self, r, U):
        return (1 - r) * polyval(self.params.LLRS, U) + r * polyval(self.params.HHRS, U)

    def I(self, U):
        return self.Imix(self.r, U)

    def Ireset(self, a, c, U):
        return a * abs(self.params.Umax - U) ** self.params.eta + c

    def get_LRS(self) -> np.ndarray:
        return self.y[LRS_INDEX]

    def get_HRS(self) -> np.ndarray:
        return self.y[HRS_INDEX]

    def get_US(self):
        return self.y[US_INDEX]

    def get_UR(self):
        return self.y[UR_INDEX]

    def rIU(self, I, U):
        IHHRS_U = polyval(self.params.HHRS, U)
        ILLRS_U = polyval(self.params.LLRS, U)
        return (I - ILLRS_U) / (IHHRS_U - ILLRS_U)

    def get_rHRS(self, R):
        return (self.params.G_LLRS - 1 / R) / (self.params.G_LLRS - self.params.G_HHRS)

    def get_reset_coefs(self, x1, x2, y1, y2):
        a = (y1 - y2) / abs(x2 - x1) ** self.params.eta
        c = y2
        return np.vstack((a, c))

    def var_sample(self):
        """
        Draw the next VAR terms, updating the history matrix (c.Xhat)
        involves a shift operation for the subset of columns corresponding to drawVARMask == true
        """
        nfeatures = self.params.nfeatures
        VAR = self.params.VAR
        mask = self.drawVARMask
        randn((nfeatures, self.M), out=self.Xhat[:nfeatures, :])
        x = VAR @ self.Xhat
        self.Xhat[nfeatures:-nfeatures, mask] = self.Xhat[2 * nfeatures :, mask]
        self.Xhat[-nfeatures:, mask] = x[:, mask]

    def applyVoltage(self, Ua):
        """
        Apply voltages from array U to the corresponding cell in the CellArray
        if U > UR or if U ≤ US, cell states will be modified
        """
        if type(Ua) is np.ndarray:
            Ua = Ua.astype(float32, copy=False)
        else:
            Ua = np.repeat(float32(Ua), self.M)

        Umax = self.params.Umax
        gamma = self.params.gamma
        nfeatures = self.params.nfeatures

        self.setMask = ~self.inLRS & (Ua <= self.get_US())
        self.resetMask = ~self.inHRS & (Ua > self.UR)
        self.fullResetMask = self.resetMask & (Ua >= Umax)
        self.partialResetMask = self.resetMask & (Ua < Umax)
        self.drawVARMask = self.inLRS & self.resetMask
        self.resetCoefsCalcMask = self.drawVARMask & ~self.fullResetMask

        if any(self.setMask):
            self.r[self.setMask] = self.get_rHRS(self.get_LRS()[self.setMask])
            self.inLRS |= self.setMask
            self.inHRS = self.inHRS & ~self.setMask
            self.UR[self.setMask] = self.get_UR()[self.setMask]

        if any(self.drawVARMask):
            self.var_sample()
            self.n += self.drawVARMask
            self.y = psi(self.mu, self.sigma, gamma_f(gamma, self.Xhat[-nfeatures:, :]))

        if any(self.resetCoefsCalcMask):
            x1 = self.UR[self.resetCoefsCalcMask]
            x2 = Umax
            y1 = self.Imix(self.r[self.resetCoefsCalcMask], x1)
            r_HRS = self.get_rHRS(self.get_HRS()[self.resetCoefsCalcMask])
            y2 = self.Imix(r_HRS, x2)
            self.resetCoefs[:, self.resetCoefsCalcMask] = self.get_reset_coefs(
                x1, x2, y1, y2
            )

        if any(self.resetMask):
            self.inLRS = self.inLRS & ~self.resetMask
            self.UR[self.resetMask] = Ua[self.resetMask]

        if any(self.partialResetMask):
            Itrans = self.Ireset(
                self.resetCoefs[0, self.partialResetMask],
                self.resetCoefs[1, self.partialResetMask],
                Ua[self.partialResetMask],
            )
            self.r[self.partialResetMask] = self.rIU(Itrans, Ua[self.partialResetMask])

        if any(self.fullResetMask):
            self.inHRS |= self.fullResetMask
            self.r[self.fullResetMask] = self.get_rHRS(
                self.get_HRS()[self.fullResetMask]
            )


def load_params(param_fp: str = default_param_fp, p: int = 10):
    ### load model parameters from file
    with open(param_fp, "r", encoding="UTF-8") as f:
        json_params = json.load(f)

    gamma = array32(json_params["gamma"])
    nfeatures, gamma_deg = gamma.shape

    VAR_keys = [k for k in json_params.keys() if k.startswith("VAR")]
    available_orders = np.sort([int(k.split("_")[-1]) for k in VAR_keys])
    q = available_orders[np.where(available_orders >= p)[0][0]]
    VAR = array32(json_params[f"VAR_{q:>03}"])[:, : (p + 1) * nfeatures]
    Umax = float32(json_params["Umax"])
    U0 = float32(json_params["U_0"])
    eta = float32(json_params["eta"])
    HHRS = array32(json_params["HHRS"])
    LLRS = array32(json_params["LLRS"])
    G_HHRS = polyval(HHRS, U0) / U0
    G_LLRS = polyval(LLRS, U0) / U0
    HHRSdeg = HHRS.shape[0]
    LLRSdeg = LLRS.shape[0]
    wk = array32(json_params["wk"])
    K = wk.shape[0]
    LDtD = np.moveaxis(
        array32(json_params["LDtD"]).reshape(2 * nfeatures, K, 2 * nfeatures), 1, 2
    )
    muDtD = array32(json_params["mu_DtD"])

    return CellParams(
        Umax,
        U0,
        eta,
        nfeatures,
        p,
        K,
        gamma_deg,
        G_HHRS,
        G_LLRS,
        HHRSdeg,
        LLRSdeg,
        HHRS,
        LLRS,
        gamma,
        wk,
        muDtD,
        LDtD,
        VAR,
    )


default_params = load_params(default_param_fp)


def CellArrayCPU(M):
    return CellArray(M, default_params)


def Imix(r, U, HHRS, LLRS):
    return (1 - r) * polyval(LLRS, U) + r * polyval(HHRS, U)


def I(c: CellArray, U):
    return Imix(c.r, U, c.params.HHRS, c.params.LLRS)


def Iread(c: CellArray, U=Uread, BW=1e8):
    """
    Return the current at Ureadout for the current cell state
    """
    # Don't read out at exactly zero, because then we can't calculate Johnson noise
    if type(U) is np.ndarray:
        U = U.astype(float32, copy=False)
        U[U == 0] = float32(1e-12)
    else:
        U = float32(U)
        if U == 0:
            U = float32(1e-12)

    BW = float32(BW)
    Inoiseless = I(c, U)
    johnson = 4 * kBT * BW * np.abs(Inoiseless / U)
    shot = 2 * e * np.abs(Inoiseless) * BW
    sigma_total = np.sqrt(johnson + shot)
    randn(c.M, out=c.Iread)
    c.Iread = Inoiseless + c.Iread * sigma_total
    return c.Iread


def test(M=2**5, N=2**6):
    """
    Continuous IV sweeping of M devices for N cycles.
    Gives visual indication whether things are working.
    Every device gets the same voltage, though this is not necessary
    """
    pts = 200  # per cycle, make it divisible by 4
    Umin = -1.5
    Umax = 1.5
    linspace32 = partial(np.linspace, dtype=float32)
    Usweep = np.concatenate(
        (
            linspace32(0, Umin, pts // 4),
            linspace32(Umin, Umax, pts // 2),
            linspace32(Umax, 0, pts // 4),
        )
    )

    cells = CellArrayCPU(M)

    nfeatures = cells.params.nfeatures

    Umat = np.tile(Usweep[:, np.newaxis, np.newaxis], (M, N))
    Imat = np.empty_like(Umat)
    ymat = empty32((nfeatures, M, N))

    for n in range(N):
        ymat[:, :, n] = cells.y
        for i in range(pts):
            cells.applyVoltage(Umat[i, :, n])
            # no noise (otherwise use Iread)
            Imat[i, :, n] = I(cells, Umat[i, :, n])
            # I[i, :, n] = Iread(c, Umat[i,1], 8, -200f-6, 200f-6, 1f9)

    from matplotlib import pyplot as plt

    sqM = int(np.floor(sqrt(M)))
    sqM = min(sqM, 10)
    fig, axs = plt.subplots(
        sqM,
        sqM,
        sharex=True,
        sharey=True,
        layout="tight",
        gridspec_kw=dict(wspace=0, hspace=0),
    )
    colors = plt.cm.jet(np.linspace(0, 1, max(N, 2)))
    lw = 0.5
    alpha = 0.7
    for n in range(N):
        for i in range(sqM):
            for j in range(sqM):
                m = sqM * i + j
                Iplot = 1e6 * Imat[:, m, n]
                Uplot = Umat[:, m, n]
                axs[i, j].plot(Uplot, Iplot, lw=lw, alpha=alpha, color=colors[n])
                if j == 0:
                    axs[i, j].set_ylabel("I [muA]")
                if i == sqM - 1:
                    axs[i, j].set_xlabel("U [V]")

    axs[0, 0].set_xlim(-1.5, 1.5)
    axs[0, 0].set_ylim(-195, 195)

    # Scatterplot of the generated features
    fig, axs = plt.subplots(nrows=nfeatures, sharex=True, figsize=(12, 4))
    colors = [f"C{i}" for i in range(nfeatures)]
    for i in range(nfeatures):
        axs[i].scatter(
            np.arange(N * M), ymat[i, :, :].T, c=colors[i], s=3, edgecolor="none"
        )
        axs[i].set_ylabel(f"Feature {i}")
        for m in range(M):
            axs[i].axvline(m * N, color="black")
    axs[0].set_yscale("log")
    axs[1].set_xlim(0, N * M)
    axs[-1].set_xlabel("Cycle/Device")
    fig.align_ylabels()

    return Umat, Imat, ymat
