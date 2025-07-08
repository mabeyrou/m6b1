from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def hello_world():
    return {"message": "The server is up and running!"}


@router.get("/health")
async def heath():
    return {"status": "ok"}
