from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
import numpy as np
import io
from typing import Optional

from ml.preprocessing.dataset_adapter import GenericCSVAdapter, SignalMetadata
from ml.preprocessing.signal_preprocessor import SignalPreprocessor
from ml.features.signal_stats import SignalStatsCalculator
from backend.app.schemas.signal import (
    SignalAnalysisResponse,
    ProcessSignalRequest,
    SignalMetadataSchema,
    TimeDomainStatsSchema,
    FrequencyDomainStatsSchema,
)

router = APIRouter(prefix="/signal", tags=["Signal Processing"])
preprocessor = SignalPreprocessor()
adapter = GenericCSVAdapter()


def _build_analysis_response(
    raw_signal: np.ndarray,
    metadata: SignalMetadata,
    detrend: bool = True,
    filter_signal: bool = False,
    lowcut: Optional[float] = 10.0,
    highcut: Optional[float] = 5000.0,
    normalization: Optional[str] = "zscore",
) -> SignalAnalysisResponse:
    """Helper to process signal and generate structured Pydantic response."""
    # 1. Preprocess signal
    prep_res = preprocessor.process(
        raw_signal=raw_signal,
        fs=metadata.sampling_rate,
        detrend=detrend,
        filter_signal=filter_signal,
        lowcut=lowcut,
        highcut=highcut,
        normalization=normalization,
    )
    processed_signal = prep_res["processed_signal"]
    fs = metadata.sampling_rate

    # 2. Compute Statistics
    time_stats = SignalStatsCalculator.compute_time_features(processed_signal)
    freq_stats = SignalStatsCalculator.compute_frequency_features(processed_signal, fs)

    # 3. Compute FFT for visualization
    freqs, magnitudes = SignalStatsCalculator.compute_fft(processed_signal, fs)

    # Downsample time & FFT vectors for fast frontend rendering (max 1024 points)
    max_pts = 1024
    if len(processed_signal) > max_pts:
        step = len(processed_signal) // max_pts
        time_preview = processed_signal[::step][:max_pts].tolist()
    else:
        time_preview = processed_signal.tolist()

    if len(freqs) > max_pts:
        step_fft = len(freqs) // max_pts
        freqs_preview = freqs[::step_fft][:max_pts].tolist()
        mags_preview = magnitudes[::step_fft][:max_pts].tolist()
    else:
        freqs_preview = freqs.tolist()
        mags_preview = magnitudes.tolist()

    return SignalAnalysisResponse(
        metadata=SignalMetadataSchema(**metadata.to_dict()),
        sample_count=len(processed_signal),
        duration_sec=float(len(processed_signal) / fs) if fs > 0 else 0.0,
        time_stats=TimeDomainStatsSchema(**time_stats),
        frequency_stats=FrequencyDomainStatsSchema(**freq_stats),
        time_series_preview=time_preview,
        fft_spectrum={"frequencies": freqs_preview, "magnitudes": mags_preview},
    )


@router.post("/upload", response_model=SignalAnalysisResponse)
async def upload_signal_csv(
    file: UploadFile = File(...),
    sampling_rate: Optional[float] = Form(12000.0),
    rpm: Optional[float] = Form(None),
    load_hp: Optional[float] = Form(None),
    machine_id: Optional[str] = Form(None),
    detrend: bool = Form(True),
    filter_signal: bool = Form(False),
    lowcut: Optional[float] = Form(10.0),
    highcut: Optional[float] = Form(5000.0),
    normalization: Optional[str] = Form("zscore"),
):
    """
    Accepts CSV upload with vibration signal data and optional metadata.
    Returns processed signal preview, FFT spectrum, and time/frequency statistics.
    """
    if not file.filename.endswith(".csv") and not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only CSV or TXT files are accepted.",
        )

    try:
        content = await file.read()
        csv_io = io.BytesIO(content)

        meta_override = {
            "sampling_rate": sampling_rate,
            "rpm": rpm,
            "load_hp": load_hp,
            "machine_id": machine_id,
        }

        signal_array, metadata = adapter.load_signal(csv_io, meta_override)

        return _build_analysis_response(
            raw_signal=signal_array,
            metadata=metadata,
            detrend=detrend,
            filter_signal=filter_signal,
            lowcut=lowcut,
            highcut=highcut,
            normalization=normalization,
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signal processing error: {str(e)}",
        )


@router.post("/process", response_model=SignalAnalysisResponse)
async def process_signal_payload(req: ProcessSignalRequest):
    """
    Accepts raw signal array in JSON payload for inline processing.
    """
    if not req.signal or len(req.signal) < 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Signal array must contain at least 128 numeric samples.",
        )

    try:
        raw_signal = np.array(req.signal, dtype=np.float64)
        metadata = SignalMetadata(
            sampling_rate=req.sampling_rate or 12000.0,
            rpm=req.rpm,
            load_hp=req.load_hp,
            machine_id=req.machine_id,
            dataset_source="JSON Payload",
        )

        return _build_analysis_response(
            raw_signal=raw_signal,
            metadata=metadata,
            detrend=req.detrend if req.detrend is not None else True,
            filter_signal=req.filter_signal if req.filter_signal is not None else False,
            lowcut=req.lowcut,
            highcut=req.highcut,
            normalization=req.normalization,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signal processing error: {str(e)}",
        )
