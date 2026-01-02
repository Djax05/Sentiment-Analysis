from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
async def root():
    return {"meesage": "This site is heathy"}