from typing import List
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.templating import Jinja2Templates
from starlette.templating import Jinja2Templates
from app.core import calculate_numbers as cn, image_to_number as itn

app = FastAPI(
    title="Calculator from ML Image-Reconizer",
    description="App para calcular números con imágenes.",
    version="0.1.0",
    contact={
        "name": "Crhistian David Vergara Gomez",
        "url": "https://www.linkedin.com/in/cristian-david-vergara-gomez/",
        "email": "krisskira@gmail.com"
    }
)

app.mount("/public", StaticFiles(directory="public"), name="public")
templates = Jinja2Templates(directory="templates", auto_reload=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={}
    )


@app.post("/api/v1/calculate", response_class=JSONResponse)
async def calculate(images: List[UploadFile] = File(...), operation: str = Form(...)):
    if not images:
        raise HTTPException(
            status_code=400, detail="No se proporcionaron imágenes.")
    if operation not in ["sum", "subtract", "multiply", "divide"]:
        raise HTTPException(status_code=400, detail="Operación no válida.")

    numbers = []
    for image in images:
        number = await itn.image_to_number(image)
        numbers.append(number)

    signature = {
        "sum": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/"
    }

    result = await cn.calculate_numbers(numbers, signature[operation])
    response = {"result": result,
                "signature": signature[operation], "numbers": numbers}
    print(">>> Response:", response)
    return response
