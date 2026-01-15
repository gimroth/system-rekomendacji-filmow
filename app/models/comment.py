from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from ..database import Base

class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Klucz obcy do użytkownika
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # ID filmu (z pliku CSV)
    movie_id = Column(Integer, nullable=False) 

    # Relacja zwrotna
    # "comments" tutaj musi odpowiadać nazwie zmiennej w klasie User (którą dodaliśmy wyżej)
    user = relationship("User", back_populates="comments")