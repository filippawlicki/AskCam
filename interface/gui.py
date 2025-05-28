import gradio as gr
import cv2
import threading
import queue
import time
from audio.listener import AudioListener
from vision.llava_wrapper import generate_answer
from audio.tts import myTTS


shared_state = {"question": "", "new_question": False, "waiting_for_answer": False, "answer": "", "info_text": ""}
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
        with state_lock:
            shared_state["info_text"] = "Generating speech..."
        tts_engine.speak(text)
        tts_queue.task_done()
        with state_lock:
            shared_state["waiting_for_answer"] = False

def text_to_speech(text: str):
    """Add text to the TTS queue for processing."""
    tts_queue.put(text)

def capture_camera():
    global current_frame
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            current_frame = frame


def ask_question(image, question):
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
        time.sleep(0.5) # Avoid busy waiting

        if should_listen:
            with state_lock:
                shared_state["info_text"] = "Listening for hotword (hey)..."

            listener.listen_hotword()
            with state_lock:
                shared_state["info_text"] = "Hotword detected. Now listening for question..."

            question_text = listener.listen_question()


            with state_lock:
                shared_state["question"] = question_text
                shared_state["new_question"] = True
                shared_state["waiting_for_answer"] = True
                shared_state["info_text"] = "Question received. Processing..."



def check_for_new_question():
    """Function is called in a separate thread to check for new questions."""
    global current_frame
    while True:
        time.sleep(0.5)
        with state_lock:
            should_check = shared_state["new_question"]

        if should_check:
            with state_lock:
                shared_state["new_question"] = False
                question = shared_state["question"]
            image = current_frame
            # Transfer from cv2 (BGR) to RGB format
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = None
            ask_question(image, question)

def periodic_check():
    with state_lock:
        return shared_state["info_text"], shared_state["answer"]



def run_gui():
    with gr.Blocks() as demo:
        gr.Markdown("# AskCam - Ask Your Camera, Answered by LLM")

        camera_feed = gr.Image(label="Camera feed", sources="webcam", type="numpy", streaming=True, height="70vh")
        info_text = gr.Textbox(label="Info", interactive=False)
        output_text = gr.Textbox(label="Answer", interactive=False)

        timer = gr.Timer()
        timer.tick(
            periodic_check,
            inputs=[],
            outputs=[info_text, output_text]
        )

    threading.Thread(target=check_for_new_question, daemon=True).start()
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=hotword_listener, daemon=True).start()
    threading.Thread(target=capture_camera, daemon=True).start()
    demo.launch()

if __name__ == "__main__":
    run_gui()
