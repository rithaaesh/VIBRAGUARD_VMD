import numpy as np
import vmdpy
from typing import Dict, Any, List, Tuple
from ml.features.signal_stats import SignalStatsCalculator


class VMDEngine:
    """
    Variational Mode Decomposition (VMD) engine wrapper.
    Decomposes 1D vibration signal into K Intrinsic Mode Functions (IMFs).
    """

    def __init__(
        self,
        K: int = 5,
        alpha: float = 2000.0,
        tau: float = 0.0,
        DC: int = 0,
        init: int = 1,
        tol: float = 1e-7,
        max_iter: int = 500,
    ):
        self.K = int(K)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.DC = int(DC)
        self.init = int(init)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def decompose(
        self, signal: np.ndarray, fs: float = 12000.0
    ) -> Dict[str, Any]:
        """
        Decomposes signal into K IMFs and computes decomposition quality metrics.
        """
        if signal.ndim != 1:
            signal = signal.flatten()

        N = len(signal)
        if N < 128:
            raise ValueError(f"Signal length ({N}) is too short for VMD decomposition (min 128).")

        # Execute VMD via vmdpy
        # u: (K, N) IMFs
        # u_hat: (N, K) spectra
        # omega: (max_iter, K) center frequency history
        try:
            u, u_hat, omega = vmdpy.VMD(
                f=signal,
                alpha=self.alpha,
                tau=self.tau,
                K=self.K,
                DC=self.DC,
                init=self.init,
                tol=self.tol,
            )
        except Exception as e:
            raise RuntimeError(f"VMD decomposition failed: {str(e)}")

        # Signal Reconstruction
        reconstructed = np.sum(u, axis=0)
        orig_norm = np.linalg.norm(signal)
        if orig_norm > 1e-12:
            reconstruction_error = float(np.linalg.norm(signal - reconstructed) / orig_norm)
        else:
            reconstruction_error = 0.0

        # Per-IMF Statistics & Center Frequencies
        imf_stats: List[Dict[str, Any]] = []
        imf_waveforms: List[List[float]] = []

        # Extract final center frequencies (normalized 0 to 0.5 cycles/sample -> convert to Hz)
        if omega.ndim == 2 and omega.shape[0] > 0:
            final_omegas = omega[-1, :] * fs  # Convert normalized rad/sample to Hz
        else:
            final_omegas = np.zeros(self.K)

        # Downsampling for preview (max 1024 points per IMF)
        max_pts = 1024
        step = max(1, N // max_pts)

        for k in range(self.K):
            imf_k = u[k, :]
            rms = float(np.sqrt(np.mean(imf_k**2)))
            energy = float(np.sum(imf_k**2))
            kurt = float(SignalStatsCalculator.compute_time_features(imf_k)["kurtosis"])
            freq_feats = SignalStatsCalculator.compute_frequency_features(imf_k, fs)
            peak = float(np.max(np.abs(imf_k)))
            var = float(np.var(imf_k))
            center_freq_hz = float(abs(final_omegas[k])) if k < len(final_omegas) else float(freq_feats["dominant_freq"])

            imf_stats.append(
                {
                    "imf_index": k + 1,
                    "rms": rms,
                    "energy": energy,
                    "kurtosis": kurt,
                    "dominant_freq": freq_feats["dominant_freq"],
                    "spectral_entropy": freq_feats["spectral_entropy"],
                    "peak": peak,
                    "variance": var,
                    "center_freq_hz": center_freq_hz,
                }
            )

            # Store downsampled waveform preview
            imf_waveforms.append(imf_k[::step][:max_pts].tolist())

        return {
            "K": self.K,
            "alpha": self.alpha,
            "tau": self.tau,
            "reconstruction_error": reconstruction_error,
            "imf_stats": imf_stats,
            "imf_waveforms": imf_waveforms,
            "reconstructed_preview": reconstructed[::step][:max_pts].tolist(),
            "raw_imfs": u,  # Full raw numpy matrix (K, N)
        }
