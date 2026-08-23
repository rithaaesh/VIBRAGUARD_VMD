from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.api.v1.signal import router as signal_router
from backend.app.api.v1.vmd import router as vmd_router
from backend.app.api.v1.predict import router as predict_router
from backend.app.api.v1.system import router as system_router

app = FastAPI(
    title=settings.APP_NAME,
    description="Adaptive VMD-Based Predictive Maintenance and Explainable Fault Diagnosis System API",
    version="1.0.0",
    debug=settings.DEBUG,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register v1 Routers
app.include_router(signal_router, prefix="/api/v1")
app.include_router(vmd_router, prefix="/api/v1")
app.include_router(predict_router, prefix="/api/v1")
app.include_router(system_router, prefix="/api/v1")


@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint returning system status."""
    return {
        "status": "ok",
        "system": settings.APP_NAME,
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
