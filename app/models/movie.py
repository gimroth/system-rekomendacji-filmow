from sqlalchemy import Column, Integer, Text, TIMESTAMP, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    year = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    tmdb_id = Column(Integer, unique=True, nullable=True)
    description = Column(Text, nullable=True)
    poster_url = Column(Text, nullable=True)
    rating = Column(Float, default=0.0)

    # Relacje
    genres = relationship("Genre", secondary="movie_genres", back_populates="movies", overlaps="movie,genre")
    movie_genres = relationship("MovieGenre", back_populates="movie", cascade="all, delete-orphan", overlaps="genres")
    ratings = relationship("Rating", back_populates="movie", cascade="all, delete-orphan")
    
    # Relacja do komentarzy (musi pasować do back_populates="movie" w Comment)
    comments = relationship("Comment", back_populates="movie", cascade="all, delete-orphan")