"""
DataProcessor - wersja rozszerzona
Łączy istniejące metody z inteligentnymi funkcjami dla cech i gatunków

Autorzy: Nadia (oryginał), Emilia (rozszerzenia)
Data: 2026-01-13
"""

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.models.rating import Rating
from app.models.user import UserPreference


class DataProcessor:
    def __init__(self, db: Session):
        self.db = db

    # =========================================================================
    # ISTNIEJĄCE METODY (od Nadii)
    # =========================================================================

    def _get_top_3_aspect_names(self, weights_dict: dict):
        """Wybiera 3 aspekty z najwyższymi wagami."""
        top_3 = sorted(weights_dict, key=weights_dict.get, reverse=True)[:3]
        return top_3

    def get_training_data(self):
        """Pobiera i skaluje (0-1) dane do treningu modelu."""
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

        cols_to_scale = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction', 'target']
        for col in cols_to_scale:
            # Skalowanie 1-5 na 0-1
            df[col] = (df[col].astype(float) - 1) / 4
            # Zabezpieczenie (clip) - to jest to co dodały dziewczyny, bardzo ważne!
            df[col] = df[col].clip(lower=0, upper=1)

        return df

    def get_user_input_for_anfis(self, user_id: int):
        """Przygotowuje wektor użytkownika i zwraca nazwy wybranych cech."""
        ratings_count = self.db.query(Rating).filter(Rating.user_id == user_id).count()

        if ratings_count < 5:
            pref = self.db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
            if not pref:
                return None, []
            weights = {
                'story': pref.weight_story, 'acting': pref.weight_acting,
                'visuals': pref.weight_visuals, 'sound': pref.weight_sound,
                'direction': pref.weight_direction
            }
        else:
            avg = self.db.query(
                func.avg(Rating.story), func.avg(Rating.acting),
                func.avg(Rating.visuals), func.avg(Rating.sound),
                func.avg(Rating.direction)
            ).filter(Rating.user_id == user_id).first()
            weights = {
                'story': float(avg[0] or 3), 'acting': float(avg[1] or 3),
                'visuals': float(avg[2] or 3), 'sound': float(avg[3] or 3),
                'direction': float(avg[4] or 3)
            }

        top_3_keys = self._get_top_3_aspect_names(weights)

        personalized_input = {}
        for key in top_3_keys:
            # Skalowanie 0-1
            personalized_input[f'u_{key}'] = (float(weights[key]) - 1) / 4

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
            # Mapowanie na te same klucze co użytkownik, skala 0-1
            movie_input[f'm_{key}'] = (all_aspects[key] - 1) / 4

        return movie_input

    # =========================================================================
    # NOWE METODY: Inteligentne cechy + gatunki (Emilia)
    # =========================================================================

    def get_smart_user_features(self, user_id: int):
        """
        Inteligentne pobieranie cech użytkownika.

        Logika:
        - Priorytet 1: Formularz (jeśli wypełniony)
        - Priorytet 2: Średnie z ocen (jeśli ≥5 ocen)
        - Fallback: Domyślne wartości

        Returns:
            tuple: (user_features_dict, top_3_keys, source)
                   source = 'form' | 'ratings' | 'default'
        """
        # Sprawdź formularz
        pref = self.db.query(UserPreference).filter(
            UserPreference.user_id == user_id
        ).first()

        ratings_count = self.db.query(func.count(Rating.id)).filter(
            Rating.user_id == user_id
        ).scalar()

        # PRIORYTET 1: Formularz (jeśli wypełniony)
        if pref and self._has_valid_preferences(pref):
            weights = {
                'story': pref.weight_story,
                'acting': pref.weight_acting,
                'visuals': pref.weight_visuals,
                'sound': pref.weight_sound,
                'direction': pref.weight_direction
            }
            top_3_keys = self._get_top_3_aspect_names(weights)

            user_features = {}
            for key in top_3_keys:
                user_features[f'u_{key}'] = (weights[key] - 1) / 4.0

            return user_features, top_3_keys, 'form'

        # PRIORYTET 2: Średnie z ocen (jeśli ≥5)
        elif ratings_count >= 5:
            ratings = self.db.query(Rating).filter(
                Rating.user_id == user_id
            ).all()

            weights = {}
            for aspect in ['story', 'acting', 'visuals', 'sound', 'direction']:
                values = [getattr(r, aspect) for r in ratings if getattr(r, aspect)]
                weights[aspect] = float(np.mean(values)) if values else 3.0

            top_3_keys = self._get_top_3_aspect_names(weights)

            user_features = {}
            for key in top_3_keys:
                user_features[f'u_{key}'] = (weights[key] - 1) / 4.0

            return user_features, top_3_keys, 'ratings'

        # FALLBACK: Domyślne
        else:
            default_keys = ['story', 'acting', 'direction']
            user_features = {f'u_{k}': 0.6 for k in default_keys}
            return user_features, default_keys, 'default'

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