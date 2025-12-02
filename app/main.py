from fastapi import FastAPI
from app.database import Base, engine
from app.routers import auth
from app.routers import admin  # Dodano import admin routera

# Tworzy tabele
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Movie Recommendation API") # Dodano title

app.include_router(auth.router)
app.include_router(admin.router) # Dodano admin router

@app.get("/")
def root():
    return {"message": "API działa!"}