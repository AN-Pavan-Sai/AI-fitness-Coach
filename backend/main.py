import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from database.db import connect_db, disconnect_db, db
from model.rag import init_rag, get_fitness_response

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Connecting to DB...")
    # NOTE: db connection will fail if DATABASE_URL is not set and prisma is not generated
    try:
        await connect_db()
    except Exception as e:
        print("Database not connected (make sure Prisma schema is pushed and DATABASE_URL is set):", e)
        
    print("Initializing RAG Model...")
    init_rag()
    yield
    # Shutdown logic
    print("Disconnecting DB...")
    await disconnect_db()

app = FastAPI(title="AI Fitness Coach API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class UserRegister(BaseModel):
    email: str
    password: str
    name: str

class UserLogin(BaseModel):
    email: str
    password: str

class ChatQuery(BaseModel):
    query: str

# --- Endpoints ---
@app.post("/register")
async def register(user: UserRegister):
    existing = await db.user.find_unique(where={"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = await db.user.create(
        data={
            "email": user.email,
            "password": user.password,
            "name": user.name
        }
    )
    return {"message": "User registered successfully", "user_id": new_user.id}

@app.post("/login")
async def login(user: UserLogin):
    db_user = await db.user.find_unique(where={"email": user.email})
    if not db_user or db_user.password != user.password:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"message": "Login successful", "user_id": db_user.id}

@app.post("/chat")
async def chat(chat_query: ChatQuery):
    response = get_fitness_response(chat_query.query)
    return {"response": response}
