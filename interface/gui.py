import gradio as gr
import cv2
import threading
import numpy as np
import pyttsx3
from vision.llava_wrapper import generate_answer

tts_engine = pyttsx3.init()

def text_to_speech(text: str):
    tts_engine.say(text)
    tts_engine.runAndWait()

def capture_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    # Change color from BGR (opencv) to RGB (gradio)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def ask_question(image, question):
    if image is None:
        return "No image from camera!", None

    answer = generate_answer(image, question)

    # Run text-to-speech in a separate thread to avoid blocking the GUI
    threading.Thread(target=text_to_speech, args=(answer,)).start()

    return answer, image

def live_camera_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame

def run_gui():
    with gr.Blocks() as demo:
        gr.Markdown("# AskCam - Ask your camera what it sees")

        with gr.Row():
            camera_feed = gr.Image(label="Camera feed", sources=["webcam"], streaming=True)
            question_input = gr.Textbox(label="Enter your question", placeholder="What am I holding in my hand?") # For now just a placeholder

        output_text = gr.Textbox(label="Answer", interactive=False)

        ask_btn = gr.Button("Ask")

        ask_btn.click(fn=ask_question, inputs=[camera_feed, question_input], outputs=[output_text, camera_feed])

    demo.launch()

if __name__ == "__main__":
    run_gui()
