from fastapi import APIRouter, HTTPException, status
import numpy as np
from ml.vmd.vmd_engine import VMDEngine
from ml.optimization.vmd_optimizer import VMDOptimizer
from backend.app.schemas.vmd import (
    VMDDecomposeRequest,
    VMDDecomposeResponse,
    IMFStatSchema,
    VMDOptimizeRequest,
    VMDOptimizeResponse,
    TrajectoryPointSchema,
)

router = APIRouter(prefix="/vmd", tags=["VMD Analysis"])


@router.post("/decompose", response_model=VMDDecomposeResponse)
async def decompose_signal_vmd(req: VMDDecomposeRequest):
    """
    Performs fixed-parameter Variational Mode Decomposition (VMD).
    Returns K IMFs with per-IMF statistics, center frequencies, and reconstruction error.
    """
    if not req.signal or len(req.signal) < 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Signal array must contain at least 128 samples for VMD decomposition.",
        )

    try:
        signal_array = np.array(req.signal, dtype=np.float64)
        engine = VMDEngine(
            K=req.K or 5,
            alpha=req.alpha or 2000.0,
            tau=req.tau or 0.0,
            DC=req.DC or 0,
        )

        res = engine.decompose(signal_array, fs=req.sampling_rate or 12000.0)
        imf_schemas = [IMFStatSchema(**stat) for stat in res["imf_stats"]]

        return VMDDecomposeResponse(
            K=res["K"],
            alpha=res["alpha"],
            tau=res["tau"],
            reconstruction_error=res["reconstruction_error"],
            imf_stats=imf_schemas,
            imf_waveforms=res["imf_waveforms"],
            reconstructed_preview=res["reconstructed_preview"],
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"VMD engine error: {str(e)}",
        )


@router.post("/optimize", response_model=VMDOptimizeResponse)
async def optimize_vmd_parameters(req: VMDOptimizeRequest):
    """
    Performs deterministic search over K and alpha parameters to find optimal VMD parameters.
    Returns baseline vs optimized K/alpha, fitness metrics, parameter trajectory, and optimal decomposition.
    """
    if not req.signal or len(req.signal) < 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Signal array must contain at least 128 samples for adaptive VMD optimization.",
        )

    try:
        signal_array = np.array(req.signal, dtype=np.float64)
        optimizer = VMDOptimizer()
        opt_res = optimizer.optimize_deterministic(
            signal=signal_array,
            fs=req.sampling_rate or 12000.0,
            k_range=(req.k_min or 3, req.k_max or 7),
        )

        opt_decomp = opt_res["optimized_decomposition"]
        imf_schemas = [IMFStatSchema(**stat) for stat in opt_decomp["imf_stats"]]

        decomp_response = VMDDecomposeResponse(
            K=opt_decomp["K"],
            alpha=opt_decomp["alpha"],
            tau=opt_decomp["tau"],
            reconstruction_error=opt_decomp["reconstruction_error"],
            imf_stats=imf_schemas,
            imf_waveforms=opt_decomp["imf_waveforms"],
            reconstructed_preview=opt_decomp["reconstructed_preview"],
        )

        trajectory_schemas = [
            TrajectoryPointSchema(**point) for point in opt_res["trajectory"]
        ]

        return VMDOptimizeResponse(
            initial_K=opt_res["initial_K"],
            initial_alpha=opt_res["initial_alpha"],
            initial_fitness=opt_res["initial_fitness"],
            optimized_K=opt_res["optimized_K"],
            optimized_alpha=opt_res["optimized_alpha"],
            optimized_fitness=opt_res["optimized_fitness"],
            fitness_improvement=opt_res["fitness_improvement"],
            trajectory=trajectory_schemas,
            optimized_decomposition=decomp_response,
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Adaptive VMD optimization error: {str(e)}",
        )
