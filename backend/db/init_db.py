from backend.db.database import engine
from backend.db import models

models.Base.metadata.create_all(bind=engine)

print("Database initialized successfully.")