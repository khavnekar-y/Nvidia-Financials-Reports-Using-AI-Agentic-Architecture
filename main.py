# main.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph_pipeline.pipeline import build_graph

app = FastAPI()
graph = build_graph()

class QueryRequest(BaseModel):
    question: str
    year: int
    quarter: int

@app.post("/research_report")
def research_report(request: QueryRequest):
    state = {
        "question": request.question,
        "year": request.year,
        "quarter": request.quarter
    }
    result = graph.invoke(state)
    return {"final_report": result.get("final_report")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
