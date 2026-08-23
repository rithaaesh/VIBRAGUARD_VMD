from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import io


class SignalMetadata:
    """Holds metadata for a vibration signal recording."""

    def __init__(
        self,
        sampling_rate: Optional[float] = None,
        rpm: Optional[float] = None,
        load_hp: Optional[float] = None,
        machine_id: Optional[str] = None,
        dataset_source: str = "Generic CSV",
    ):
        self.sampling_rate = sampling_rate if sampling_rate is not None else 12000.0  # Default 12kHz
        self.rpm = rpm if rpm is not None else "Unknown / Not Provided"
        self.load_hp = load_hp if load_hp is not None else "Unknown / Not Provided"
        self.machine_id = machine_id if machine_id is not None else "Unknown / Not Provided"
        self.dataset_source = dataset_source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sampling_rate": self.sampling_rate,
            "rpm": self.rpm,
            "load_hp": self.load_hp,
            "machine_id": self.machine_id,
            "dataset_source": self.dataset_source,
        }


class DatasetAdapter(ABC):
    """Abstract base class for vibration dataset adapters."""

    @abstractmethod
    def load_signal(
        self, data_input: Any, metadata_override: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, SignalMetadata]:
        """Loads raw vibration signal array and metadata."""
        pass


class GenericCSVAdapter(DatasetAdapter):
    """Adapter for generic CSV files containing vibration signals."""

    def load_signal(
        self, data_input: Any, metadata_override: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, SignalMetadata]:
        """
        Parses CSV input (file path, bytes, or StringIO).
        Expects a column named 'vibration' (or picks the first numeric column).
        """
        if isinstance(data_input, (str, io.StringIO, io.BytesIO)):
            df = pd.read_csv(data_input)
        elif isinstance(data_input, pd.DataFrame):
            df = data_input
        else:
            raise ValueError(f"Unsupported input type for GenericCSVAdapter: {type(data_input)}")

        if df.empty:
            raise ValueError("Uploaded CSV file is empty.")

        # Find vibration column
        target_col = None
        for col in df.columns:
            if "vib" in col.lower() or "acc" in col.lower() or "val" in col.lower() or "sig" in col.lower():
                target_col = col
                break

        if target_col is None:
            # Fallback to first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[0]
            else:
                raise ValueError("No numeric vibration data column found in CSV.")

        # Extract signal array
        signal_array = df[target_col].to_numpy(dtype=np.float64)

        # Parse metadata
        meta_dict = metadata_override or {}
        sampling_rate = meta_dict.get("sampling_rate")
        rpm = meta_dict.get("rpm")
        load_hp = meta_dict.get("load_hp")
        machine_id = meta_dict.get("machine_id")

        metadata = SignalMetadata(
            sampling_rate=float(sampling_rate) if sampling_rate else 12000.0,
            rpm=float(rpm) if rpm else None,
            load_hp=float(load_hp) if load_hp else None,
            machine_id=str(machine_id) if machine_id else None,
            dataset_source="Generic CSV Upload",
        )

        return signal_array, metadata


class CWRUAdapter(DatasetAdapter):
    """Adapter for Case Western Reserve University (CWRU) bearing dataset files."""

    def load_signal(
        self, data_input: Any, metadata_override: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, SignalMetadata]:
        meta_dict = metadata_override or {}
        adapter = GenericCSVAdapter()
        signal, metadata = adapter.load_signal(data_input, metadata_override)
        metadata.dataset_source = "CWRU Bearing Dataset"
        if not meta_dict.get("sampling_rate"):
            metadata.sampling_rate = 12000.0  # CWRU standard
        return signal, metadata
