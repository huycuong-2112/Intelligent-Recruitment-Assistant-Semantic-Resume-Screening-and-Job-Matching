from fastapi import APIRouter

router = APIRouter(prefix="/resume", tags=["Resume"])


@router.get("/ping")
def ping():
    return {"message": "Resume router is working"}


# TODO: thêm endpoint upload/parse CV thật ở đây (sprint sau)