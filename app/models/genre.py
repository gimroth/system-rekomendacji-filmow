from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.database import Base

# Tabela asocjacyjna
class MovieGenre(Base):
    __tablename__ = "movie_genres"

    movie_id = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    genre_id = Column(Integer, ForeignKey("genres.id", ondelete="CASCADE"), primary_key=True)

    # Relacje
    movie = relationship("Movie", back_populates="movie_genres", overlaps="genres")
    genre = relationship("Genre", back_populates="movie_genres", overlaps="movies")


class Genre(Base):
    __tablename__ = "genres"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)

    # Wiele-do-Wielu z Movie
    movies = relationship("Movie", secondary="movie_genres", back_populates="genres", overlaps="movie,genre")

    # Relacja do MovieGenre
    movie_genres = relationship("MovieGenre", back_populates="genre", cascade="all, delete-orphan")
