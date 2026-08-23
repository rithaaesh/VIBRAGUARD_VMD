import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, Any, Tuple, Optional


class SignalPreprocessor:
    """Handles signal validation, detrending, filtering, normalization, and windowing."""

    @staticmethod
    def validate_signal(raw_signal: np.ndarray, min_length: int = 128) -> np.ndarray:
        """
        Validates raw vibration signal array.
        - Checks array type and dimensions
        - Rejects NaN and Infinity values
        - Ensures minimum length constraint
        """
        if not isinstance(raw_signal, np.ndarray):
            try:
                raw_signal = np.array(raw_signal, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Signal must contain only numeric samples.") from exc

        if raw_signal.ndim != 1:
            raw_signal = raw_signal.flatten()

        if np.isnan(raw_signal).any() or np.isinf(raw_signal).any():
            raise ValueError("Signal contains NaN or infinite samples; remove invalid values and retry.")

        if len(raw_signal) < min_length:
            raise ValueError(
                f"Signal length ({len(raw_signal)}) is below minimum required threshold ({min_length} samples)."
            )

        return raw_signal

    @staticmethod
    def detrend_signal(signal: np.ndarray, type: str = "linear") -> np.ndarray:
        """Removes linear or constant trend from the signal."""
        return scipy_signal.detrend(signal, type=type)

    @staticmethod
    def bandpass_filter(
        signal: np.ndarray,
        fs: float,
        lowcut: Optional[float] = 10.0,
        highcut: Optional[float] = 5000.0,
        order: int = 4,
    ) -> np.ndarray:
        """
        Applies a Butterworth bandpass filter.
        Ensures cutoff frequencies are within valid Nyquist limits (0 < lowcut < highcut < fs/2).
        """
        nyquist = 0.5 * fs
        if nyquist <= 0:
            return signal

        # Clamp cutoffs to Nyquist limits
        low = max(lowcut if lowcut else 10.0, 1.0)
        high = min(highcut if highcut else nyquist * 0.95, nyquist * 0.95)

        if low >= high:
            return signal

        sos = scipy_signal.butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
        return scipy_signal.sosfilt(sos, signal)

    @staticmethod
    def normalize_signal(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Normalizes signal via Z-score standard scaling or Min-Max scaling."""
        if method == "zscore":
            std = np.std(signal)
            if std == 0:
                return signal - np.mean(signal)
            return (signal - np.mean(signal)) / std
        elif method == "minmax":
            min_val = np.min(signal)
            max_val = np.max(signal)
            if max_val == min_val:
                return np.zeros_like(signal)
            return (signal - min_val) / (max_val - min_val)
        return signal

    def process(
        self,
        raw_signal: np.ndarray,
        fs: float = 12000.0,
        detrend: bool = True,
        filter_signal: bool = False,
        lowcut: Optional[float] = 10.0,
        highcut: Optional[float] = 5000.0,
        normalization: Optional[str] = "zscore",
    ) -> Dict[str, Any]:
        """
        Executes full preprocessing pipeline and returns clean signal with metadata.
        """
        valid_signal = self.validate_signal(raw_signal)

        processed = valid_signal.copy()
        if detrend:
            processed = self.detrend_signal(processed)

        if filter_signal:
            processed = self.bandpass_filter(processed, fs, lowcut, highcut)

        if normalization:
            processed = self.normalize_signal(processed, method=normalization)

        return {
            "raw_signal": valid_signal,
            "processed_signal": processed,
            "sample_count": len(processed),
            "sampling_rate": fs,
            "duration_sec": len(processed) / fs if fs > 0 else 0.0,
        }

    @staticmethod
    def create_windows(
        signal: np.ndarray, window_size: int = 1024, overlap_pct: float = 0.5
    ) -> np.ndarray:
        """
        Segments 1D signal into sliding windows.
        Returns 2D array of shape (num_windows, window_size).
        """
        if len(signal) < window_size:
            # Zero-pad if shorter than window size
            padded = np.zeros(window_size)
            padded[: len(signal)] = signal
            return np.array([padded])

        step = int(window_size * (1.0 - overlap_pct))
        step = max(step, 1)

        windows = []
        for i in range(0, len(signal) - window_size + 1, step):
            windows.append(signal[i : i + window_size])

        return np.array(windows)
