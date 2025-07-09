from pydantic import BaseModel


class DigitRequest(BaseModel):
    image: str
