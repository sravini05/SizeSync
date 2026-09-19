import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.routers import profile

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SizeSync API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok", "reconstruction_backend": settings.RECONSTRUCTION_BACKEND}


app.include_router(profile.router)

# Serves generated .glb meshes at /media/meshes/<file>.glb -- this is what
# BodyProfile.mesh_url points at, and what the frontend's Three.js viewer
# loads directly.
app.mount("/media/meshes", StaticFiles(directory=str(settings.MESHES_DIR)), name="meshes")
