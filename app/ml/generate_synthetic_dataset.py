import os
import numpy as np
import pandas as pd

"""
Generates a synthetic dataset compatible with DataProcessor.get_training_data()
Columns: user_id, movie_id, m_story, m_acting, m_visuals, m_sound, m_direction, target
"""

def generate(path=None, n_samples=3000, random_state=42):
    rng = np.random.default_rng(random_state)
    user_ids = rng.integers(1, 500, size=n_samples)
    movie_ids = rng.integers(1, 1000, size=n_samples)

    # generate 5 aspect scores in 0-1
    m_story = rng.random(n_samples)
    m_acting = rng.random(n_samples)
    m_visuals = rng.random(n_samples)
    m_sound = rng.random(n_samples)
    m_direction = rng.random(n_samples)

    # create target as weighted combination + noise
    # weights chosen to create meaningful signal
    weights = np.array([0.30, 0.25, 0.20, 0.10, 0.15])
    feats = np.vstack([m_story, m_acting, m_visuals, m_sound, m_direction]).T
    base = feats @ weights
    noise = rng.normal(0, 0.05, size=n_samples)
    target = np.clip(base + noise, 0.0, 1.0)

    df = pd.DataFrame({
        'user_id': user_ids,
        'movie_id': movie_ids,
        'm_story': m_story,
        'm_acting': m_acting,
        'm_visuals': m_visuals,
        'm_sound': m_sound,
        'm_direction': m_direction,
        'target': target
    })

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Synthetic dataset saved to {path}")
    return df


if __name__ == '__main__':
    generate(path=os.path.join(os.path.dirname(__file__), 'datasets', 'anfis_synthetic.csv'))
