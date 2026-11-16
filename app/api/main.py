from fastapi import FastAPI
from app.api.core.config import settings
from app.api.routers import resume, job, matching, explanations

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

app.include_router(resume.router, prefix=settings.API_V1_PREFIX)
app.include_router(job.router, prefix=settings.API_V1_PREFIX)
app.include_router(matching.router, prefix=settings.API_V1_PREFIX)
app.include_router(explanations.router, prefix=settings.API_V1_PREFIX)

@app.get("/")
def health_check():
    return {"status": "ok", "app": settings.APP_NAME}
