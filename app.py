# from fastapi import FastAPI, UploadFile, File
# from faster_whisper import WhisperModel
# import tempfile
# import shutil
# import os

# app = FastAPI()

# # High accuracy configuration
# model = WhisperModel(
#     "small",
#     compute_type="int8",
# )

# @app.post("/transcribe")
# async def transcribe(file: UploadFile = File(...)):

#     with tempfile.NamedTemporaryFile(delete=False) as temp:
#         shutil.copyfileobj(file.file, temp)
#         temp_path = temp.name

#     segments, info = model.transcribe(
#         temp_path,
#         beam_size=5,
#         best_of=5,
#         temperature=0,
#         vad_filter=True,
#         language="en"
#     )

#     text = ""
#     for segment in segments:
#         text += segment.text + " "

#     os.remove(temp_path)

#     return {"text": text.strip()}

from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import shutil
import os

app = FastAPI()

model = WhisperModel("small", compute_type="int8")

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
        shutil.copyfileobj(audio.file, temp)
        temp_path = temp.name

    segments, info = model.transcribe(
        temp_path,
        beam_size=5,
        best_of=5,
        temperature=0,
        vad_filter=True,
        language="en"
    )

    text = " ".join([segment.text for segment in segments])

    os.remove(temp_path)

    return {"text": text.strip()}