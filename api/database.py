from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

from config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Generating database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_db_tables():
    """Creating tables"""
    Base.metadata.create_all(bind=engine)


def clear_table(table_name):
    """Clearing a table"""
    db = SessionLocal()
    try:
        db.execute(text(f"DELETE FROM {table_name}"))
        db.commit()
        print(f"🗑️  Table {table_name} cleared")
    except Exception as e:
        db.rollback()
        print(f"❌ Error while clearing: {e}")
        raise
    finally:
        db.close()
