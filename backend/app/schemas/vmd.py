from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class VMDDecomposeRequest(BaseModel):
    signal: List[float] = Field(..., description="1D vibration signal array")
    sampling_rate: Optional[float] = 12000.0
    K: Optional[int] = Field(5, ge=2, le=12, description="Number of modes (IMFs)")
    alpha: Optional[float] = Field(2000.0, ge=100.0, le=20000.0, description="Balancing parameter")
    tau: Optional[float] = Field(0.0, description="Dual ascent time step")
    DC: Optional[int] = Field(0, description="DC mode inclusion")


class IMFStatSchema(BaseModel):
    imf_index: int
    rms: float
    energy: float
    kurtosis: float
    dominant_freq: float
    spectral_entropy: float
    peak: float
    variance: float
    center_freq_hz: float


class VMDDecomposeResponse(BaseModel):
    K: int
    alpha: float
    tau: float
    reconstruction_error: float
    imf_stats: List[IMFStatSchema]
    imf_waveforms: List[List[float]]
    reconstructed_preview: List[float]


class VMDOptimizeRequest(BaseModel):
    signal: List[float] = Field(..., description="1D vibration signal array")
    sampling_rate: Optional[float] = 12000.0
    k_min: Optional[int] = Field(3, ge=2, le=10)
    k_max: Optional[int] = Field(7, ge=3, le=12)


class TrajectoryPointSchema(BaseModel):
    K: int
    alpha: float
    fitness: float
    reconstruction_error: float


class VMDOptimizeResponse(BaseModel):
    initial_K: int
    initial_alpha: float
    initial_fitness: float
    optimized_K: int
    optimized_alpha: float
    optimized_fitness: float
    fitness_improvement: float
    trajectory: List[TrajectoryPointSchema]
    optimized_decomposition: VMDDecomposeResponse
