-- USERS
CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(50) NOT NULL UNIQUE,
    email           VARCHAR(100) NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    role            VARCHAR(20) NOT NULL DEFAULT 'user',  -- 'user' | 'admin'
    created_at      TIMESTAMP DEFAULT NOW()
);

-- MOVIES
CREATE TABLE movies (
    id          INTEGER PRIMARY KEY,     -- movieId z CSV
    title       TEXT NOT NULL,
    year        INTEGER,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- GENRES
CREATE TABLE genres (
    id      SERIAL PRIMARY KEY,
    name    VARCHAR(50) UNIQUE NOT NULL
);

-- MOVIE_GENRES (tabela relacyjna many-to-many)
CREATE TABLE movie_genres (
    movie_id    INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    genre_id    INTEGER REFERENCES genres(id) ON DELETE CASCADE,
    PRIMARY KEY (movie_id, genre_id)
);

-- RATINGS
CREATE TABLE ratings (
    id          SERIAL PRIMARY KEY,

    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    movie_id    INTEGER REFERENCES movies(id) ON DELETE CASCADE,

    rating      NUMERIC(2, 1) NOT NULL CHECK (rating >= 0 AND rating <= 5),

    story       INTEGER CHECK (story BETWEEN 1 AND 5),
    acting      INTEGER CHECK (acting BETWEEN 1 AND 5),
    visuals     INTEGER CHECK (visuals BETWEEN 1 AND 5),
    sound       INTEGER CHECK (sound BETWEEN 1 AND 5),
    direction   INTEGER CHECK (direction BETWEEN 1 AND 5),

    rated_at    TIMESTAMP DEFAULT NOW(),

    UNIQUE(user_id, movie_id)
);
