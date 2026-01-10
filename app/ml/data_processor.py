import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.models.movie import Movie
from app.models.rating import Rating
from app.models.user import UserPreference


class DataProcessor:
    def __init__(self, db: Session):
        self.db = db

    def _get_top_3_aspect_names(self, weights_dict: dict):
        """
        Pomocnicza metoda wybierająca 3 aspekty z najwyższymi wagami.
        """
        # Sortowanie aspektów według wag malejąco i wybór 3 pierwszych
        top_3 = sorted(weights_dict, key=weights_dict.get, reverse=True)[:3]
        return top_3

    def get_training_data(self):
        """
        Pobiera dane do treningu. Łączy oceny użytkowników z cechami filmów.
        Używane przez Emilię do budowy bazy reguł ANFIS.
        """
        query = (
            select(
                Rating.user_id,
                Rating.movie_id,
                Rating.story.label('m_story'),
                Rating.acting.label('m_acting'),
                Rating.visuals.label('m_visuals'),
                Rating.sound.label('m_sound'),
                Rating.direction.label('m_direction'),
                Rating.rating.label('target')
            )
        )
        results = self.db.execute(query).all()

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Skalowanie ocen 1-5 na zakres 0-1 (wymagane dla funkcji przynależności)
        cols_to_scale = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction', 'target']
        for col in cols_to_scale:
            df[col] = (df[col].astype(float) - 1) / 4

        return df

    def get_user_input_for_anfis(self, user_id: int):
        """
        Przygotowuje spersonalizowany wektor 3 najważniejszych cech użytkownika.
        Rozwiązuje problem Cold Start wykorzystując wagi z nowego formularza.
        """
        # Sprawdzenie liczby ocen użytkownika (historia vs onboarding)
        ratings_count = self.db.query(Rating).filter(Rating.user_id == user_id).count()

        if ratings_count < 5:
            # SCENARIUSZ 1: Cold Start - Pobieramy wagi z Twojego modelu UserPreference
            pref = self.db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
            if not pref:
                return None

            weights = {
                'story': pref.weight_story,
                'acting': pref.weight_acting,
                'visuals': pref.weight_visuals,
                'sound': pref.weight_sound,
                'direction': pref.weight_direction
            }
        else:
            # SCENARIUSZ 2: Normalny - Obliczamy średnie z prawdziwych ocen użytkownika
            avg = self.db.query(
                func.avg(Rating.story),
                func.avg(Rating.acting),
                func.avg(Rating.visuals),
                func.avg(Rating.sound),
                func.avg(Rating.direction)
            ).filter(Rating.user_id == user_id).first()

            weights = {
                'story': float(avg[0] or 0),
                'acting': float(avg[1] or 0),
                'visuals': float(avg[2] or 0),
                'sound': float(avg[3] or 0),
                'direction': float(avg[4] or 0)
            }

        # Dynamiczny wybór 3 najważniejszych cech dla TEGO użytkownika
        top_3_keys = self._get_top_3_aspect_names(weights)

        # Tworzenie spersonalizowanego wektora wejściowego (znormalizowanego 0-1)
        personalized_input = {}
        for key in top_3_keys:
            # Normalizacja: waga / 5.0 (ponieważ min to 1, a max to 5)
            personalized_input[f'u_{key}'] = weights[key] / 5.0

        return personalized_input