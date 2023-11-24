from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routers.routerPredict import routerPredict as predit_router

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "http://192.168.0.7:5500",
    "http:52.188.149.227:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def message():
    return "Se debe escribir /predict"

app.include_router(predit_router)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

