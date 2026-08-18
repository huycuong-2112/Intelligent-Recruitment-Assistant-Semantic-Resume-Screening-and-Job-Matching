from fastapi import APIRouter

router = APIRouter(prefix="/job", tags=["Job"])


@router.get("/ping")
def ping():
    return {"message": "Job router is working"}


# TODO: thêm endpoint upload/parse Job Description thật ở đây (sprint sau)