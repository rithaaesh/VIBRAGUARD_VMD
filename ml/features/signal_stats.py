import numpy as np
from scipy import stats as scipy_stats
from typing import Dict, Any, Tuple


class SignalStatsCalculator:
    """Computes time-domain and frequency-domain statistical metrics for vibration signals."""

    @staticmethod
    def compute_time_features(signal: np.ndarray) -> Dict[str, float]:
        """
        Computes 11 time-domain condition indicators.
        """
        if len(signal) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "variance": 0.0,
                "rms": 0.0,
                "peak": 0.0,
                "p2p": 0.0,
                "kurtosis": 0.0,
                "skewness": 0.0,
                "crest_factor": 0.0,
                "shape_factor": 0.0,
                "impulse_factor": 0.0,
            }

        mean_val = float(np.mean(signal))
        std_val = float(np.std(signal))
        var_val = float(np.var(signal))
        rms_val = float(np.sqrt(np.mean(signal**2)))
        peak_val = float(np.max(np.abs(signal)))
        p2p_val = float(np.ptp(signal))

        # Kurtosis & Skewness
        kurtosis_val = float(scipy_stats.kurtosis(signal, fisher=True))
        skewness_val = float(scipy_stats.skew(signal))

        # Dimensionless factors (safeguard division by zero)
        crest_factor = float(peak_val / rms_val) if rms_val > 1e-12 else 0.0
        abs_mean = float(np.mean(np.abs(signal)))
        shape_factor = float(rms_val / abs_mean) if abs_mean > 1e-12 else 0.0
        impulse_factor = float(peak_val / abs_mean) if abs_mean > 1e-12 else 0.0

        return {
            "mean": mean_val,
            "std": std_val,
            "variance": var_val,
            "rms": rms_val,
            "peak": peak_val,
            "p2p": p2p_val,
            "kurtosis": kurtosis_val,
            "skewness": skewness_val,
            "crest_factor": crest_factor,
            "shape_factor": shape_factor,
            "impulse_factor": impulse_factor,
        }

    @staticmethod
    def compute_fft(
        signal: np.ndarray, fs: float = 12000.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes Single-Sided Real Fast Fourier Transform (FFT).
        Returns (frequencies_hz, magnitude_spectrum).
        """
        N = len(signal)
        if N == 0:
            return np.array([]), np.array([])

        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)
        magnitudes = np.abs(fft_vals) / N
        magnitudes[1:-1] *= 2.0  # Single-sided amplitude compensation

        return freqs, magnitudes

    @staticmethod
    def compute_frequency_features(
        signal: np.ndarray, fs: float = 12000.0
    ) -> Dict[str, float]:
        """
        Computes frequency-domain condition indicators from FFT spectrum.
        """
        freqs, magnitudes = SignalStatsCalculator.compute_fft(signal, fs)
        if len(magnitudes) == 0 or np.sum(magnitudes) == 0:
            return {
                "dominant_freq": 0.0,
                "spectral_centroid": 0.0,
                "spectral_bandwidth": 0.0,
                "spectral_entropy": 0.0,
                "band_energy": 0.0,
            }

        # 1. Dominant Frequency
        dominant_idx = np.argmax(magnitudes)
        dominant_freq = float(freqs[dominant_idx])

        # 2. Spectral Centroid
        total_mag = np.sum(magnitudes)
        spectral_centroid = float(np.sum(freqs * magnitudes) / total_mag) if total_mag > 1e-12 else 0.0

        # 3. Spectral Bandwidth
        spectral_bandwidth = float(
            np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitudes) / total_mag)
        ) if total_mag > 1e-12 else 0.0

        # 4. Spectral Entropy (Normalized Shannon entropy)
        prob_dist = magnitudes / total_mag
        prob_dist = prob_dist[prob_dist > 0]  # Filter zero components
        spectral_entropy = float(-np.sum(prob_dist * np.log2(prob_dist)) / np.log2(len(magnitudes))) if len(magnitudes) > 1 else 0.0

        # 5. Band Energy
        band_energy = float(np.sum(magnitudes**2))

        return {
            "dominant_freq": dominant_freq,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_entropy": spectral_entropy,
            "band_energy": band_energy,
        }

    @classmethod
    def compute_all_features(
        cls, signal: np.ndarray, fs: float = 12000.0
    ) -> Dict[str, float]:
        """Combines time and frequency domain statistical features."""
        time_feats = cls.compute_time_features(signal)
        freq_feats = cls.compute_frequency_features(signal, fs)
        return {**time_feats, **freq_feats}
