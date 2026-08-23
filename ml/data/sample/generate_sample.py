import os
import numpy as np
import pandas as pd


def generate_sample_signals(output_dir: str = "ml/data/sample"):
    """
    Generates synthetic bearing vibration CSV datasets for testing.
    - healthy_bearing.csv: Low noise baseline signal.
    - faulty_bearing.csv: Signal containing periodic impacts (BPFI simulation).
    """
    os.makedirs(output_dir, exist_ok=True)
    fs = 12000.0  # 12 kHz sampling rate
    N = 4096  # 4096 samples (~0.34 seconds)
    t = np.linspace(0, (N - 1) / fs, N)

    # 1. Healthy Bearing Signal (Fundamental rotational frequency 30 Hz + noise)
    f_rot = 30.0  # 1800 RPM
    healthy_signal = (
        0.15 * np.sin(2 * np.pi * f_rot * t)
        + 0.05 * np.sin(2 * np.pi * 2 * f_rot * t)
        + np.random.normal(0, 0.02, N)
    )

    df_healthy = pd.DataFrame(
        {
            "timestamp": t,
            "vibration": healthy_signal,
        }
    )
    healthy_path = os.path.join(output_dir, "healthy_bearing.csv")
    df_healthy.to_csv(healthy_path, index=False)

    # 2. Faulty Bearing Signal (BPFI ~ 160 Hz + high kurtosis impacts + noise)
    f_fault = 160.0
    impacts = np.zeros(N)
    period_samples = int(fs / f_fault)
    for idx in range(0, N, period_samples):
        impacts[idx : min(idx + 50, N)] += np.exp(-t[: min(50, N - idx)] * 1000) * np.sin(
            2 * np.pi * 2500 * t[: min(50, N - idx)]
        )

    faulty_signal = (
        0.2 * np.sin(2 * np.pi * f_rot * t)
        + 0.8 * impacts
        + np.random.normal(0, 0.05, N)
    )

    df_faulty = pd.DataFrame(
        {
            "timestamp": t,
            "vibration": faulty_signal,
        }
    )
    faulty_path = os.path.join(output_dir, "faulty_bearing.csv")
    df_faulty.to_csv(faulty_path, index=False)

    print(f"Generated sample datasets:\n  - {healthy_path}\n  - {faulty_path}")
    return healthy_path, faulty_path


if __name__ == "__main__":
    generate_sample_signals()
