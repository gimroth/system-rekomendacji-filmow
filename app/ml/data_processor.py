import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.models.rating import Rating
from app.models.user import UserPreference

class DataProcessor:
    def __init__(self, db: Session):
        self.db = db

    def _get_top_3_aspect_names(self, weights_dict: dict):
        return sorted(weights_dict, key=weights_dict.get, reverse=True)[:3]

    def get_training_data(self):
        """Bezpieczne, liniowe skalowanie bez potęgowania."""
        query = select(
            Rating.user_id, Rating.movie_id,
            Rating.story.label('m_story'), Rating.acting.label('m_acting'),
            Rating.visuals.label('m_visuals'), Rating.sound.label('m_sound'),
            Rating.direction.label('m_direction'), Rating.rating.label('target')
        )
        results = self.db.execute(query).all()
        if not results: return pd.DataFrame()

        df = pd.DataFrame(results)
        
        # Skalujemy wszystko liniowo 0-1
        cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction', 'target']
        for col in cols:
            df[col] = (df[col].astype(float) - 1.0) / 4.0
            # Dodajemy mały margines, aby uniknąć czystego 0 i 1 (krytyczne dla ANFIS)
            df[col] = df[col].clip(0.001, 0.999) 
        
        return df

    def get_user_input_for_anfis(self, user_id: int):
        """Przygotowuje profil użytkownika na podstawie ocen lub preferencji."""
        ratings_count = self.db.query(Rating).filter(Rating.user_id == user_id).count()

        if ratings_count < 5:
            pref = self.db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
            if not pref: return None, []
            weights = {'story': pref.weight_story, 'acting': pref.weight_acting, 'visuals': pref.weight_visuals, 'sound': pref.weight_sound, 'direction': pref.weight_direction}
        else:
            avg = self.db.query(func.avg(Rating.story), func.avg(Rating.acting), func.avg(Rating.visuals), func.avg(Rating.sound), func.avg(Rating.direction)).filter(Rating.user_id == user_id).first()
            weights = {'story': float(avg[0] or 3), 'acting': float(avg[1] or 3), 'visuals': float(avg[2] or 3), 'sound': float(avg[3] or 3), 'direction': float(avg[4] or 3)}

        top_3_keys = self._get_top_3_aspect_names(weights)
        personalized_input = {f'u_{key}': (float(weights[key]) - 1) / 4 for key in top_3_keys}
        return personalized_input, top_3_keys

    def get_movie_input_for_anfis(self, movie_id: int, top_3_keys: list):
        avg = self.db.query(func.avg(Rating.story), func.avg(Rating.acting), func.avg(Rating.visuals), func.avg(Rating.sound), func.avg(Rating.direction)).filter(Rating.movie_id == movie_id).first()
        all_aspects = {'story': float(avg[0] or 3.0), 'acting': float(avg[1] or 3.0), 'visuals': float(avg[2] or 3.0), 'sound': float(avg[3] or 3.0), 'direction': float(avg[4] or 3.0)}
        return {f'm_{key}': (all_aspects[key] - 1) / 4 for key in top_3_keys}

    def get_smart_user_features(self, user_id: int):
        pref = self.db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        ratings_count = self.db.query(func.count(Rating.id)).filter(Rating.user_id == user_id).scalar()

        # PRIORYTET 1: Formularz (dodajemy BOOSTER)
        if pref and self._has_valid_preferences(pref):
            weights = {
                'story': pref.weight_story,
                'acting': pref.weight_acting,
                'visuals': pref.weight_visuals,
                'sound': pref.weight_sound,
                'direction': pref.weight_direction
            }
            top_3 = self._get_top_3_aspect_names(weights)

            user_features = {}
            for key in top_3:
                # BOOSTER: Przesuwamy wagi z formularza w górę o 10%, 
                # aby łatwiej wpadały w zakres "Bardzo Polecam" (max 1.0)
                val_norm = (weights[key] - 1) / 4.0
                user_features[f'u_{key}'] = min(1.0, val_norm * 1.1)

            return user_features, top_3, 'form'
        elif ratings_count >= 5:
            ratings = self.db.query(Rating).filter(Rating.user_id == user_id).all()
            weights = {}
            for aspect in ['story', 'acting', 'visuals', 'sound', 'direction']:
                values = [getattr(r, aspect) for r in ratings if getattr(r, aspect)]
                if values:
                    # Zamiast czystego np.mean, bierzemy 75-ty percentyl lub ważoną 
                    # (daje to wagę bliższą Twoim najwyższym ocenom)
                    weights[aspect] = float(np.percentile(values, 75)) 
                else:
                    weights[aspect] = 3.0
        return {f'u_{k}': 0.6 for k in ['story', 'acting', 'direction']}, ['story', 'acting', 'direction'], 'default'

    def get_smart_user_genres(self, user_id: int):
        """
        Inteligentne pobieranie gatunków użytkownika.

        Logika:
        - Priorytet 1: Z formularza (jeśli wypełniony)
        - Priorytet 2: Z ocenionych filmów (2-10 najczęstszych)
        - Łączenie: Jeśli ma formularz I oceny → merge inteligentnie

        Returns:
            list: Lista preferowanych gatunków (2-10)
        """
        from app.models import Genre
        from app.models.genre import MovieGenre

        # Sprawdź formularz
        pref = self.db.query(UserPreference).filter(
            UserPreference.user_id == user_id
        ).first()

        # Sprawdź oceny
        ratings_count = self.db.query(func.count(Rating.id)).filter(
            Rating.user_id == user_id
        ).scalar()

        genres_from_form = []
        genres_from_ratings = []

        # Pobierz gatunki z formularza
        if pref:
            # DOSTOSUJ DO TWOJEGO MODELU!
            # Jeśli masz pole preferred_genre_ids w UserPreference:
            form_genre_ids = pref.preferred_genre_ids if hasattr(pref, 'preferred_genre_ids') else []

            # Albo jeśli masz tabelę user_preferred_genres:
            if not form_genre_ids:
                try:
                    from app.models.user_preferred_genre import UserPreferredGenre
                    form_genres = self.db.query(UserPreferredGenre).filter(
                        UserPreferredGenre.user_id == user_id
                    ).all()
                    form_genre_ids = [fg.genre_id for fg in form_genres]
                except:
                    pass

            if form_genre_ids:
                genres_from_form = [
                    g.name for g in self.db.query(Genre).filter(
                        Genre.id.in_(form_genre_ids)
                    ).all()
                ]

        # Pobierz gatunki z ocenionych filmów
        if ratings_count >= 2:
            # Znajdź najczęściej oceniane gatunki
            genre_counts = self.db.query(
                Genre.name,
                func.count(Rating.id).label('count')
            ).join(MovieGenre, MovieGenre.genre_id == Genre.id) \
                .join(Rating, Rating.movie_id == MovieGenre.movie_id) \
                .filter(Rating.user_id == user_id) \
                .group_by(Genre.name) \
                .order_by(func.count(Rating.id).desc()) \
                .limit(10) \
                .all()

            genres_from_ratings = [g[0] for g in genre_counts if g[1] >= 2]  # Min 2 oceny

        # MERGE LOGIC
        if genres_from_form and genres_from_ratings:
            # Ma formularz I oceny
            # Priorytet: Najpierw z formularza, potem z ocen
            result = []
            for g in genres_from_form:
                if g not in result:
                    result.append(g)
            for g in genres_from_ratings:
                if g not in result and len(result) < 10:
                    result.append(g)

            return result[:10]

        elif genres_from_form:
            # Tylko formularz
            return genres_from_form[:10]

        elif genres_from_ratings:
            # Tylko oceny (min 2, max 10)
            return genres_from_ratings[:10]

        else:
            # Brak danych
            return []

    def _has_valid_preferences(self, pref):
        """Sprawdza czy formularz jest wypełniony."""
        if not pref:
            return False

        weights = [
            pref.weight_story,
            pref.weight_acting,
            pref.weight_visuals,
            pref.weight_sound,
            pref.weight_direction
        ]

        # Sprawdź czy są wypełnione (nie None, nie 0)
        return all(w is not None and w > 0 for w in weights)