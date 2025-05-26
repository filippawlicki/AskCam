import gradio as gr
import cv2
import threading
import numpy as np
import pyttsx3
from audio.listener import listen_hotword_and_get_question
from vision.llava_wrapper import generate_answer

tts_engine = pyttsx3.init()

shared_state = {"question": "", "new_question": False}
state_lock = threading.Lock()
current_frame = None

def text_to_speech(text: str):
    tts_engine.say(text)
    tts_engine.runAndWait()

def capture_camera():
    global current_frame
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            current_frame = frame


def ask_question(image, question):
    print(f"Image: {image.shape if image is not None else 'None'}, Question: {question}")
    if image is None:
        return "No image from camera!", None

    answer = generate_answer(image, question)

    # Run text-to-speech in a separate thread to avoid blocking the GUI
    threading.Thread(target=text_to_speech, args=(answer,)).start()

    return answer, image

def hotword_listener():
    """Function is called in a separate thread to listen for hotword."""
    while True:
        question_text = listen_hotword_and_get_question()
        with state_lock:
            print(f"[Hotword detected] Question: {question_text}")
            shared_state["question"] = question_text
            shared_state["new_question"] = True

def periodic_check():
    global current_frame
    with state_lock:
        if shared_state["new_question"]:
            image = current_frame
            question = shared_state["question"]
            shared_state["new_question"] = False
            answer, _ = ask_question(image, question)
            return question, answer
    return gr.update(), gr.update()


def run_gui():
    with gr.Blocks() as demo:
        gr.Markdown("# AskCam - Ask your camera what it sees")

        with gr.Row():
            camera_feed = gr.Image(label="Camera feed", sources="webcam", type="numpy", streaming=True)
            question_input = gr.Textbox(label="Detected question", interactive=False)
        output_text = gr.Textbox(label="Answer", interactive=False)

        timer = gr.Timer()
        timer.tick(
            periodic_check,
            inputs=[],
            outputs=[question_input, output_text]
        )

        threading.Thread(target=hotword_listener, daemon=True).start()
        threading.Thread(target=capture_camera, daemon=True).start()
    demo.launch()

if __name__ == "__main__":
    run_gui()
