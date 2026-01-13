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
        """Wybiera 3 aspekty z najwyższymi wagami."""
        top_3 = sorted(weights_dict, key=weights_dict.get, reverse=True)[:3]
        return top_3

    def get_training_data(self):
        """Pobiera dane do treningu ANFIS."""
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

        # Skalowanie 1-5 na 0-1
        cols_to_scale = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction', 'target']
        for col in cols_to_scale:
            df[col] = (df[col].astype(float) - 1) / 4
            
        return df

    def get_user_input_for_anfis(self, user_id: int):
        """Przygotowuje spersonalizowany wektor 3 najważniejszych cech."""
        ratings_count = self.db.query(Rating).filter(Rating.user_id == user_id).count()

        if ratings_count < 5:
            # Cold Start - z preferencji
            pref = self.db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
            if not pref:
                return None, []

            weights = {
                'story': pref.weight_story,
                'acting': pref.weight_acting,
                'visuals': pref.weight_visuals,
                'sound': pref.weight_sound,
                'direction': pref.weight_direction
            }
        else:
            # Z historii ocen
            avg = self.db.query(
                func.avg(Rating.story),
                func.avg(Rating.acting),
                func.avg(Rating.visuals),
                func.avg(Rating.sound),
                func.avg(Rating.direction)
            ).filter(Rating.user_id == user_id).first()

            weights = {
                'story': float(avg[0] or 3),
                'acting': float(avg[1] or 3),
                'visuals': float(avg[2] or 3),
                'sound': float(avg[3] or 3),
                'direction': float(avg[4] or 3)
            }

        top_3_keys = self._get_top_3_aspect_names(weights)

        personalized_input = {}
        for key in top_3_keys:
            # Normalizacja 0-1
            personalized_input[f'u_{key}'] = (float(weights[key]) - 1) / 4.0

        return personalized_input, top_3_keys

    def get_movie_input_for_anfis(self, movie_id: int, top_3_keys: list):
        """Pobiera dane filmu dla cech wybranych dla użytkownika."""
        avg = self.db.query(
            func.avg(Rating.story), func.avg(Rating.acting),
            func.avg(Rating.visuals), func.avg(Rating.sound),
            func.avg(Rating.direction)
        ).filter(Rating.movie_id == movie_id).first()

        all_aspects = {
            'story': float(avg[0] or 3.0), 'acting': float(avg[1] or 3.0),
            'visuals': float(avg[2] or 3.0), 'sound': float(avg[3] or 3.0),
            'direction': float(avg[4] or 3.0)
        }

        movie_input = {}
        for key in top_3_keys:
            movie_input[f'm_{key}'] = (all_aspects[key] - 1) / 4.0
            
        return movie_input