from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {'Python Backend course project by Anton Prazdnichnykh'}
