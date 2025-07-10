from dotenv import load_dotenv
from os import getenv

load_dotenv()

APP_ENV = getenv("APP_ENV", "development")

DATABASE_URL = getenv("DATABASE_URL", "sqlite:///./data/app.db")

LOG_LEVEL = getenv("LOG_LEVEL", "DEBUG")
