from fastapi import FastAPI
import gradio as gr
from PotholeDetection.logging.logger import logger
import json
import requests

FASTAPI_URL = 'http://127.0.0.1:8000/detect'


def detect_potholes(image):
    if image is None:
        logger.error('No image provided. Please upload an image')
        return 'No image provided. Please upload an image'
    
    try:
        with open(image, 'rb') as f:
            response = requests.post(FASTAPI_URL, files = {'file': f})
        
        if response.status_code != 200:
            return f'Internal Server Error: {response.status_code}: {response.text}'
        
        result = json.dumps(response.json(), indent=2)
        return result


    except Exception as e:
        logger.error('Request failed')
        raise e


def gradio_interface():
    with gr.Blocks(theme = gr.themes.Soft(), title = 'Pothole Detection') as demo:

        gr.HTML("<h1 style='text-align:center; color:#023047;'>🛣️ Pothole Detection Dashboard</h1>")

        with gr.row():
            image_in = gr.Image(type = 'filepath', label = 'Upload image')
            json_out = gr.Code(label = 'Detections', language= 'json')

        with gr.row():
            detect_btn = gr.Button('Detect', variant='primary')
            reset_btn = gr.Button('Reset', variant = 'secondary')

        detect_btn.click(detect_potholes,
                         inputs = image_in,
                         outputs = json_out)
        
        reset_btn.click(fn = lambda : (None,None),
                        inputs = None,
                        outputs = [image_in, json_out])
        
        demo.launch(server_name = '0.0.0.0', server_port = 7860)

if __name__ == '__main__':
    gradio_interface()