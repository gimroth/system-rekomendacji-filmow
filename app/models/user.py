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
    ratings = relationship("Rating", back_populates="user", cascade="all, delete-orphan")


# Definicja nowej tabeli na preferencje początkowe
class UserPreference(Base):
    __tablename__ = "user_preferences"

    # KLUCZ GŁÓWNY I OBCY: user_id
    # Używamy user_id jako klucza głównego, zapewniając jedną preferencję na użytkownika.
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),  # Powiązanie z tabelą "users"
        primary_key=True,
        index=True
    )

    # POLA Z FORMULARZA (zgodne ze schematem Pydantic)
    preferred_genres = Column(String(500), nullable=False)

    # Wagi aspektów (skala 1-5, zgodne z polami w tabeli Ratings)
    weight_story = Column(Integer, nullable=False)
    weight_acting = Column(Integer, nullable=False)
    weight_visuals = Column(Integer, nullable=False)
    weight_sound = Column(Integer, nullable=False)
    weight_direction = Column(Integer, nullable=False)

    # Relacja do użytkownika (Relacja Jeden-do-Jednego)
    # Odwołuje się do klasy User, którą masz zdefiniowaną wyżej
    user = relationship("User", backref="preferences")