from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import shutil
import tempfile
from utils import extract_accent_features, add_vector_to_mongodb, match_accent
import os
import uvicorn

app = FastAPI()

@app.post("/upload-accent")
async def upload_accent(accent_name: str = Form(...), file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        accent_features = extract_accent_features(temp_audio_path)
        add_vector_to_mongodb(accent_name, accent_features)
        os.unlink(temp_audio_path)
        return JSONResponse(content={"message": f"{accent_name} accent uploaded successfully."}, status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading accent: {str(e)}")

@app.post("/match-accent")
async def match_accent_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are allowed")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        matched_accent, similarity_score = match_accent(temp_file_path)
        return JSONResponse(content={
            "matched_accent": matched_accent,
            "similarity_score": similarity_score
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(temp_file_path)

if __name__ == "__main__":
    uvicorn.run("server:app", port=8080, reload=True)