from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SignalMetadataSchema(BaseModel):
    sampling_rate: float = Field(12000.0, description="Sampling rate in Hz")
    rpm: Any = Field("Unknown / Not Provided", description="Machine rotational speed in RPM")
    load_hp: Any = Field("Unknown / Not Provided", description="Motor load in Horsepower")
    machine_id: Any = Field("Unknown / Not Provided", description="Machine identifier")
    dataset_source: str = Field("Generic CSV Upload", description="Source description")


class TimeDomainStatsSchema(BaseModel):
    mean: float
    std: float
    variance: float
    rms: float
    peak: float
    p2p: float
    kurtosis: float
    skewness: float
    crest_factor: float
    shape_factor: float
    impulse_factor: float


class FrequencyDomainStatsSchema(BaseModel):
    dominant_freq: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_entropy: float
    band_energy: float


class ProcessSignalRequest(BaseModel):
    signal: List[float] = Field(..., description="Array of raw vibration amplitude values")
    sampling_rate: Optional[float] = 12000.0
    detrend: Optional[bool] = True
    filter_signal: Optional[bool] = False
    lowcut: Optional[float] = 10.0
    highcut: Optional[float] = 5000.0
    normalization: Optional[str] = "zscore"
    rpm: Optional[float] = None
    load_hp: Optional[float] = None
    machine_id: Optional[str] = None


class SignalAnalysisResponse(BaseModel):
    metadata: SignalMetadataSchema
    sample_count: int
    duration_sec: float
    time_stats: TimeDomainStatsSchema
    frequency_stats: FrequencyDomainStatsSchema
    time_series_preview: List[float] = Field(..., description="Downsampled/full time waveform for charting (max 1024 points)")
    fft_spectrum: Dict[str, List[float]] = Field(..., description="Contains 'frequencies' and 'magnitudes' arrays (max 1024 points)")
