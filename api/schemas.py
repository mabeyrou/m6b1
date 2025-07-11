from pydantic import BaseModel


class PredictRequest(BaseModel):
    image: str


class PredictResponse(BaseModel):
    predicted_digit: int
    confidence: float
    digit_uuid: str


class FeedbackRequest(BaseModel):
    true_digit: int
    digit_uuid: str
    is_correct: bool
