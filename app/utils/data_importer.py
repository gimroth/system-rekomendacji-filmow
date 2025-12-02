import pandas as pd
import numpy as np # Do obsługi wartości NaN
from sqlalchemy.exc import IntegrityError
from app.database import SessionLocal, engine, Base
# Importuj wszystkie swoje modele, aby SQLAlchemy wiedział, jak mapować tabele
from app.models.movie import Movie
from app.models.genre import Genre, MovieGenre
from app.models.rating import Rating
# Pamiętaj: Użytkownicy (User) powinni być dodani ręcznie lub zaimportowani z osobnego źródła,
# bo plik ratings.csv zawiera tylko ID użytkowników, a nie dane konta.