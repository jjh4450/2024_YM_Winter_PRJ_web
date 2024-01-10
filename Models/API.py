from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile
from CNN_MNIST_Server import predict_result
from typing import IO
import os

app = FastAPI()

@app.get("/")
def read_root():
    """ 루트 경로에 대한 GET 요청을 처리합니다. 
    단순히 "Hello: World"라는 메시지를 반환합니다. """
    return {"Hello": "World"}

async def save_file(file: UploadFile = File(...)):
    """ 업로드된 파일을 서버에 저장합니다.
    파일을 서버의 지정된 경로(UPLOAD_DIR)에 저장합니다. """
    try:
        UPLOAD_DIR = "."  # 이미지를 저장할 서버 경로
        content = await file.read()
        filename = f"test.png"
        with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
            fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)
        return True
    except:
        return False

@app.post("/file/store")
async def store_file(file: UploadFile = File(...)):
    """ 파일을 서버에 저장하는 엔드포인트입니다.
    업로드된 파일을 save_file 함수를 통해 저장하고 성공 여부를 반환합니다. """
    save_status = await save_file(file)
    return {"isSave": save_status}

@app.post("/file/predict")
async def predict_file(file: UploadFile = File(...)):
    """ 파일에 대해 예측을 수행하는 엔드포인트입니다.
    파일을 먼저 저장하고, 저장 성공 시 예측 결과를 반환합니다.
    저장에 실패하면 예측 레이블로 -1을 반환합니다. """
    if await save_file(file):
        predicted_label = await predict_result()
        return {"predicted_label": predicted_label}
    else:
        return {"predicted_label": -1}
