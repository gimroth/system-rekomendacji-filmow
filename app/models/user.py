from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, CheckConstraint, Text
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

    # --- RELACJE ---
    ratings = relationship("Rating", back_populates="user", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan")


class UserPreference(Base):
    __tablename__ = "user_preferences"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)

    # Wagi (Constrainty 1-5)
    weight_story = Column(Integer, CheckConstraint('weight_story BETWEEN 1 AND 5'), nullable=False)
    weight_acting = Column(Integer, CheckConstraint('weight_acting BETWEEN 1 AND 5'), nullable=False)
    weight_visuals = Column(Integer, CheckConstraint('weight_visuals BETWEEN 1 AND 5'), nullable=False)
    weight_sound = Column(Integer, CheckConstraint('weight_sound BETWEEN 1 AND 5'), nullable=False)
    weight_direction = Column(Integer, CheckConstraint('weight_direction BETWEEN 1 AND 5'), nullable=False)

    onboarding_completed = Column(Boolean, default=True)
    # Backwards-compatible column: some databases (older installs) may have
    # stored preferred genres directly on the preferences row. Keep a
    # nullable-safe text column so inserts won't fail if the DB expects it.
    preferred_genres = Column(Text, nullable=False, default='')
    
    user = relationship("User", back_populates="preferences")