from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, func
import uuid as uuid_lib

from database import Base


class Digit(Base):
    __tablename__ = "digits"
    uuid = Column(
        String(36), primary_key=True, index=True, default=lambda: str(uuid_lib.uuid4())
    )
    img_path = Column(String, nullable=False)
    predicted_label = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    true_label = Column(Integer)
    has_feedback = Column(Boolean, default=False)
    was_used_for_training = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
