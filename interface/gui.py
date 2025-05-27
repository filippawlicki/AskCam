import gradio as gr
import cv2
import threading
import numpy as np
import pyttsx3
import queue
import time
from audio.listener import AudioListener
from vision.llava_wrapper import generate_answer
from audio.tts import myTTS


shared_state = {"question": "", "new_question": False, "waiting_for_answer": False, "answer": ""}
state_lock = threading.RLock()
current_frame = None

tts_queue = queue.Queue()

def tts_worker():
    """Background worker to process TTS requests from the queue."""
    tts_engine = myTTS()
    while True:
        text = tts_queue.get()
        if text is None:  # Exit signal
            continue
        tts_engine.speak(text)
        print("[TTS] Finished speaking: " + text)
        tts_queue.task_done()
        print("[TTS] Task done, checking for waiting_for_answer state...")
        with state_lock:
            shared_state["waiting_for_answer"] = False
            print("[TTS] Reset waiting_for_answer = False")

def text_to_speech(text: str):
    """Add text to the TTS queue for processing."""
    print("[TTS] Added: " + text)
    tts_queue.put(text)

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

    text_to_speech(answer)

    with state_lock:
        shared_state["answer"] = answer


    return answer, image

def hotword_listener():
    """Function is called in a separate thread to listen for hotword."""
    listener = AudioListener()
    listener.start_audio_stream()
    while True:
        with state_lock:
            should_listen = not shared_state["waiting_for_answer"]
        #print(f"[Hotword listener] should_listen={should_listen}")
        time.sleep(0.5) # Avoid busy waiting

        if should_listen:
            print("[Hotword listener] Waiting for hotword...")
            question_text = listener.listen_hotword_and_get_question()

            print(f"[Hotword detected] Question: {question_text}")
            with state_lock:
                shared_state["question"] = question_text
                shared_state["new_question"] = True
                shared_state["waiting_for_answer"] = True



def check_for_new_question():
    """Function is called in a separate thread to check for new questions."""
    global current_frame
    while True:
        time.sleep(0.5)
        print("[Ask question] Checking for new question...")
        with state_lock:
            if shared_state["new_question"]:
                print("[Ask question] New question detected.")
                shared_state["new_question"] = False
                image = current_frame
                # Transfer from cv2 (BGR) to RGB format
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = None
                question = shared_state["question"]
                ask_question(image, question)

def periodic_check():
    print("[Periodic check] Checking for new question...")
    with state_lock:
        return shared_state["question"], shared_state["answer"]



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

    threading.Thread(target=check_for_new_question, daemon=True).start()
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=hotword_listener, daemon=True).start()
    threading.Thread(target=capture_camera, daemon=True).start()
    demo.launch()

if __name__ == "__main__":
    run_gui()
