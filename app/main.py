from pathlib import Path
import mimetypes
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from app.utility import ApiUtility
from PotholeDetection.logging.logger import logger


app = FastAPI(title = "Pothole Detection API",
              description = "API for detecting potholes in images using a YOLOv8 model.",
              debug=True)

# Initialize utility and download model
app_utils = ApiUtility()
app_utils.download_model()
model = app_utils.load_model()
pred_path = app_utils.artifacts_path()


@app.get('/')
async def home():
    return """Welcome to the Pothole Detection API. Visit /docs for API documentation.

    Available Endpoints:
    - POST /detect: Upload an image to detect potholes.

    """

@app.post('/detect')
async def detect_potholes(file: UploadFile = File(...)):
    logger.info("Pothole detection request received.\nProcessing.....")
    try:
        file_content = await file.read()
        file_path = temp_file(file, file_content)

        # Run inference
        results = model.predict(str(file_path), conf = 0.3, save = True, project = str(pred_path))

        # Process results
        detections = app_utils.get_detections(results)

        #Get prediction image
        pred_img = Path(results[0].save_dir)/file.filename
        logger.info(f"YOLO save_dir: {results[0].save_dir}")
        logger.info(f"Predicted image expected at: {pred_img}")
        logger.info(f"File exists? {pred_img.exists()}")

        # Clean up temporary file
        temp_file(file_obj = file, delete = True)

        logger.info("Pothole detection completed successfully.")
        
        # return JSONResponse(status_code = 200,
        #                     content = {
        #                         'image_name': file.filename,
        #                         'number_of_detections': len(detections),
        #                         'detections': detections
        #                     })

        mime_type, _ = mimetypes.guess_type(pred_img)
        return FileResponse(str(pred_img), status_code=200, media_type=mime_type)

    except Exception as e:
        logger.error(f"Error during pothole detection: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal Server Error"})
    

def temp_file(file_obj = None, file_content = None, delete = False):

    if file_obj is not None:
        temp_dir = Path('temp_upload')
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)
        uploaded_file_path = temp_dir/file_obj.filename

        if not delete:
            #Save the file temporarily
            with open(uploaded_file_path, 'wb') as f:
                f.write(file_content)
            logger.info(f'Uploaded image saved temporarily at {uploaded_file_path}')
            return uploaded_file_path
        
        else:
            #Delete the file
            if uploaded_file_path.exists():
                uploaded_file_path.unlink()                
            logger.info(f'Temporary saved file at {uploaded_file_path} deleted successfully.')

            #Remove temp directory if empty
            if not any(temp_dir.iterdir()):
                temp_dir.rmdir()
                logger.info(f'Temporary directory {temp_dir} deleted as it was empty.')

if __name__ == '__main__':
    uvicorn.run("main:app", host = '0.0.0.0', port = 8000)