from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    # Treść komentarza
    content = Column(Text, nullable=False)
    
    # Klucz obcy do UŻYTKOWNIKA
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    
    # --- TO JEST BRAKUJĄCY ELEMENT, KTÓRY WYWOŁUJE BŁĄD ---
    # Klucz obcy do FILMU
    movie_id = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relacje zwrotne
    user = relationship("User", back_populates="comments")
    movie = relationship("Movie", back_populates="comments")