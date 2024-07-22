import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO, solutions
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

css = '''
.gradio-container{max-width: 1000px !important}
h1{text-align:center}
footer {
    visibility: hidden
}
'''

# Load models
yolo_model = YOLO("yolov8n.pt")
yolo_names = yolo_model.model.names
dist_obj = solutions.DistanceCalculation(names=yolo_names, view_img=False)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def process_image_for_detection(image):
    # Convert Gradio image to OpenCV format
    im0 = np.array(image)
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
    
    # Perform object detection
    tracks = yolo_model.track(im0, persist=True, show=False)
    im0 = dist_obj.start_process(im0, tracks)
    
    # Convert processed image back to RGB format
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    
    return im0

def generate_caption(image, min_len=30, max_len=100):
    raw_image = Image.fromarray(image)
    inputs = blip_processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs, min_length=min_len, max_length=max_len)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    
    return caption

def process_image(image, min_len=30, max_len=100):
    # First step: Object Detection
    detected_image = process_image_for_detection(image)
    
    # Second step: Captioning
    caption = generate_caption(detected_image, min_len, max_len)
    
    return detected_image, caption

with gr.Blocks(css=css, theme="allenai/gradio-theme") as demo:
    gr.Markdown("## Object Detection and Captioning")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        min_len_slider = gr.Slider(label="Minimum Length", minimum=1, maximum=1000, value=50)
        max_len_slider = gr.Slider(label="Maximum Length", minimum=1, maximum=1000, value=150)
        submit_button = gr.Button("Submit")
    
    with gr.Row():
        image_output = gr.Image(type="pil", label="Processed Image")
        caption_output = gr.Textbox(label="Caption")
    
    # Function to handle image processing and caption generation in two steps
    def process_and_generate(image, min_len, max_len):
        # Step 1: Object Detection
        detected_image = process_image_for_detection(image)
        # Step 2: Caption Generation
        caption = generate_caption(detected_image, min_len, max_len)
        return detected_image, caption

    # Define the interaction between the button and outputs
    submit_button.click(
        process_and_generate, 
        inputs=[image_input, min_len_slider, max_len_slider], 
        outputs=[image_output, caption_output]
    )

demo.launch(debug=True, quiet=True)