from fastapi import FastAPI
from pydantic import BaseModel

from query_expansion import expand_query


app = FastAPI(
    title="Hospital SOP Query Expansion API",
    version="1.0"
)


class QueryRequest(BaseModel):

    query: str

    max_queries: int = 8


@app.get("/")
def home():

    return {
        "message": "Hospital SOP Query Expansion API"
    }


@app.post("/expand_query")
def expand(request: QueryRequest):

    result = expand_query(
        request.query,
        request.max_queries
    )

    return result
