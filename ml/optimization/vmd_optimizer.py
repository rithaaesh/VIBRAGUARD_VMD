import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ml.vmd.vmd_engine import VMDEngine
from ml.features.signal_stats import SignalStatsCalculator


class VMDOptimizer:
    """
    Adaptive VMD parameter selection optimizer.
    Executes a deterministic grid search over K in [3, 10] and log-scaled alpha in [500, 10000]
    to minimize a composite fitness function.
    """

    def __init__(
        self,
        w_recon: float = 0.4,
        w_entropy: float = 0.3,
        w_kurtosis: float = 0.2,
        w_overlap: float = 0.1,
    ):
        self.w_recon = w_recon
        self.w_entropy = w_entropy
        self.w_kurtosis = w_kurtosis
        self.w_overlap = w_overlap

    def calculate_fitness(
        self, vmd_res: Dict[str, Any], fs: float = 12000.0
    ) -> float:
        """
        Calculates scalar fitness metric (lower is better):
        Fitness = w1 * reconstruction_error
                + w2 * mean_spectral_entropy
                - w3 * normalized_mean_kurtosis
                + w4 * mode_overlap_penalty
        """
        recon_err = vmd_res["reconstruction_error"]
        imf_stats = vmd_res["imf_stats"]

        if not imf_stats:
            return 1e6

        # Mean spectral entropy across IMFs
        entropies = [stat["spectral_entropy"] for stat in imf_stats]
        mean_entropy = float(np.mean(entropies))

        # Normalized kurtosis across IMFs
        kurtoses = [stat["kurtosis"] for stat in imf_stats]
        mean_kurtosis = float(np.mean(kurtoses))
        norm_kurtosis = max(0.0, min(1.0, mean_kurtosis / 10.0))  # Scale kurtosis to ~[0, 1]

        # Mode frequency overlap penalty (difference between center frequencies)
        center_freqs = sorted([stat["center_freq_hz"] for stat in imf_stats])
        if len(center_freqs) > 1:
            freq_diffs = np.diff(center_freqs)
            min_diff = np.min(freq_diffs)
            # Penalty if mode center frequencies are closer than 20 Hz
            overlap_penalty = float(max(0.0, (20.0 - min_diff) / 20.0))
        else:
            overlap_penalty = 0.0

        fitness = (
            self.w_recon * recon_err
            + self.w_entropy * mean_entropy
            - self.w_kurtosis * norm_kurtosis
            + self.w_overlap * overlap_penalty
        )
        return float(fitness)

    def optimize_deterministic(
        self,
        signal: np.ndarray,
        fs: float = 12000.0,
        k_range: Tuple[int, int] = (3, 7),
        alpha_candidates: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Executes deterministic search over candidate K and alpha parameters.
        Returns baseline decomposition vs optimal decomposition and optimization trajectory log.
        """
        if signal.ndim != 1:
            signal = signal.flatten()
        if len(signal) < 128:
            raise ValueError("Signal length must be at least 128 samples for optimization.")
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains NaN or infinite samples; remove invalid values and retry.")

        k_min, k_max = k_range
        if k_min < 3 or k_max > 10 or k_min > k_max:
            raise ValueError("k_range must be an inclusive range within 3 through 10.")

        if len(signal) > 2048:
            search_signal = signal[:2048]  # Downsample window length for fast search
        else:
            search_signal = signal

        if alpha_candidates is None:
            alpha_candidates = np.geomspace(500.0, 10000.0, num=5).tolist()

        # 1. Evaluate Baseline (default K=5, alpha=2000)
        baseline_engine = VMDEngine(K=5, alpha=2000.0)
        baseline_res = baseline_engine.decompose(search_signal, fs=fs)
        baseline_fitness = self.calculate_fitness(baseline_res, fs=fs)

        best_fitness = baseline_fitness
        best_params = {"K": 5, "alpha": 2000.0}
        best_res = baseline_res
        trajectory: List[Dict[str, Any]] = [
            {
                "K": 5,
                "alpha": 2000.0,
                "fitness": baseline_fitness,
                "reconstruction_error": baseline_res["reconstruction_error"],
            }
        ]

        # 2. Grid Search Loop
        for K in range(k_min, k_max + 1):
            for alpha in alpha_candidates:
                try:
                    engine = VMDEngine(K=K, alpha=alpha)
                    res = engine.decompose(search_signal, fs=fs)
                    fit = self.calculate_fitness(res, fs=fs)

                    trajectory.append(
                        {
                            "K": K,
                            "alpha": alpha,
                            "fitness": fit,
                            "reconstruction_error": res["reconstruction_error"],
                        }
                    )

                    if fit < best_fitness:
                        best_fitness = fit
                        best_params = {"K": K, "alpha": alpha}
                        best_res = res
                except Exception:
                    continue

        # 3. Final optimal decomposition on full signal
        final_engine = VMDEngine(K=best_params["K"], alpha=best_params["alpha"])
        final_opt_res = final_engine.decompose(signal, fs=fs)
        final_opt_res["fitness"] = best_fitness

        return {
            "initial_K": 5,
            "initial_alpha": 2000.0,
            "initial_fitness": baseline_fitness,
            "optimized_K": best_params["K"],
            "optimized_alpha": best_params["alpha"],
            "optimized_fitness": best_fitness,
            "fitness_improvement": float(baseline_fitness - best_fitness),
            "trajectory": trajectory,
            "optimized_decomposition": final_opt_res,
        }
