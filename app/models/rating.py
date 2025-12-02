# app/models/rating.py

from sqlalchemy import Column, Integer, Numeric, TIMESTAMP, ForeignKey, CheckConstraint, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Rating(Base):
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, index=True)

    # Klucze Obce
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)

    # Ocena ogólna (NUMERIC(2, 1))
    rating = Column(Numeric(2, 1), nullable=False)

    # Ocena aspektowa (do modelu ANFIS)
    story = Column(Integer, CheckConstraint('story BETWEEN 1 AND 5'))
    acting = Column(Integer, CheckConstraint('acting BETWEEN 1 AND 5'))
    visuals = Column(Integer, CheckConstraint('visuals BETWEEN 1 AND 5'))
    sound = Column(Integer, CheckConstraint('sound BETWEEN 1 AND 5'))
    direction = Column(Integer, CheckConstraint('direction BETWEEN 1 AND 5'))

    rated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # --- Relacje ---
    # Relacja Jeden-do-Wielu (Rating -> User, Rating -> Movie)
    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")

    # Ograniczenie unikalności (UNIQUE(user_id, movie_id))
    __table_args__ = (
        UniqueConstraint('user_id', 'movie_id', name='uix_user_movie_rating'),
    )

