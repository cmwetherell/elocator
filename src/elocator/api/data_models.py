from pydantic import BaseModel

class ComplexityRequest(BaseModel):
    fen: str

class GameRequest(BaseModel):
    pgn: str
