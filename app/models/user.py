from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relacja Jeden-do-Wielu (User -> Ratings)
    # Zakładam, że model Rating istnieje w innym pliku
    ratings = relationship("Rating", back_populates="user", cascade="all, delete-orphan")

    # --- NOWA LINIA: Relacja do Komentarzy ---
    # Musi tu być, aby back_populates="comments" w drugim pliku działało
    comments = relationship("Comment", back_populates="user", cascade="all, delete-orphan")


# Definicja tabeli preferencji (pozostaje bez zmian, jest poprawna)
class UserPreference(Base):
    __tablename__ = "user_preferences"

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        index=True
    )

    preferred_genres = Column(String(500), nullable=False)
    weight_story = Column(Integer, nullable=False)
    weight_acting = Column(Integer, nullable=False)
    weight_visuals = Column(Integer, nullable=False)
    weight_sound = Column(Integer, nullable=False)
    weight_direction = Column(Integer, nullable=False)

    # Tutaj używasz backref="preferences", co tworzy pole user.preferences
    user = relationship("User", backref="preferences")