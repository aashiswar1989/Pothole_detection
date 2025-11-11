from fastapi import FastAPI
from pathlib import Path
import gradio as gr
from PotholeDetection.logging.logger import logger
import json
import mimetypes
import os
import requests

# FASTAPI_URL = 'http://127.0.0.1:8000/detect'
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000/detect")


def detect_potholes(image):
    if image is None:
        logger.error('No image provided. Please upload an image')
        return 'No image provided. Please upload an image'
    
    try:
        with open(image, 'rb') as f:
            response = requests.post(FASTAPI_URL, files = {'file': f})
        
        if response.status_code != 200:
            return f'Internal Server Error: {response.status_code}: {response.text}'
        
        # result = json.dumps(response.json(), indent=2)
        
        # Determine the image extension from Content-Type header
        content_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type) or ".jpg"

        # Create an output file with correct extension
        pred_img = f"annotated_output{extension}"

        with open(pred_img, 'wb') as f:
            f.write(response.content)
        
        return pred_img


    except Exception as e:
        logger.error('Request failed')
        raise e


def gradio_interface():
    with gr.Blocks(theme = gr.themes.Soft(), title = 'Pothole Detection') as demo:

        gr.HTML("<h1 style='text-align:center; color:#023047;'>🛣️ Pothole Detection Dashboard</h1>")

        with gr.Row():
            image_in = gr.Image(type = 'filepath', label = 'Upload image')
            # json_out = gr.Code(label = 'Detections', language= 'json')
            image_out = gr.Image(label = 'Detections')

        with gr.Row():
            detect_btn = gr.Button('Detect', variant='primary')
            reset_btn = gr.Button('Reset', variant = 'secondary')

        detect_btn.click(detect_potholes,
                         inputs = image_in,
                         outputs = image_out)
        
        reset_btn.click(fn = lambda : (None,None),
                        inputs = None,
                        outputs = [image_in, image_out])
        
        demo.launch(server_name = '0.0.0.0', server_port = 7860, inbrowser=False, share=False)

if __name__ == '__main__':
    gradio_interface()