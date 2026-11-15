import base64
import colorsys
import json
import os
import tempfile
import threading
import time

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from PIL import Image
from pythonosc.udp_client import SimpleUDPClient
from websockets.sync.server import serve


# ============================================================
# CONFIGURATION
# ============================================================

OSC_IP = "127.0.0.1"
OSC_PORT = 9000

WEBSOCKET_IP = "127.0.0.1"
WEBSOCKET_PORT = 8765

SCALE = [0, 2, 4, 7, 9]  # C major pentatonic

MAX_IMAGE_WIDTH = 1000
MAX_IMAGE_HEIGHT = 700

NOTE_INTERVAL = 0.075
NOTE_DURATION = 0.16


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sensonic",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Sensonic")
st.caption("Turn images and videos into music through interaction.")


# ============================================================
# OSC CLIENT
# ============================================================

@st.cache_resource
def get_osc_client():

    return SimpleUDPClient(
        OSC_IP,
        OSC_PORT
    )


osc_client = get_osc_client()


# ============================================================
# PIXEL TO MUSIC
# ============================================================

def pixel_to_note(rgb):

    r, g, b = [channel / 255 for channel in rgb]

    hue, saturation, brightness = colorsys.rgb_to_hsv(
        r, g, b
    )

    # Ignore bright white / nearly white pixels
    if saturation <= 0.12 and brightness >= 0.90:
        return None

    scale_index = min(
        int(hue * len(SCALE)),
        len(SCALE) - 1
    )

    octave = min(
        int(brightness * 3),
        2
    )

    midi_note = (
        60
        + SCALE[scale_index]
        + octave * 12
    )

    amplitude = 0.05 + saturation * 0.35

    return midi_note, amplitude


# ============================================================
# WEBSOCKET SERVER
# ============================================================

@st.cache_resource
def start_websocket_server():

    state = {
        "image": None,
        "lock": threading.Lock()
    }

    def handle_connection(websocket):

        last_note_time = 0

        for message in websocket:

            try:

                data = json.loads(message)

                x = float(data["x"])
                y = float(data["y"])

                with state["lock"]:
                    image = state["image"]

                if image is None:
                    continue

                height, width, _ = image.shape

                # Convert normalized coordinates to pixels
                px = int(x * (width - 1))
                py = int(y * (height - 1))

                px = np.clip(px, 0, width - 1)
                py = np.clip(py, 0, height - 1)

                rgb = image[py, px]

                result = pixel_to_note(rgb)

                if result is None:
                    continue

                pitch, amplitude = result

                now = time.monotonic()

                # Limit note frequency
                if now - last_note_time < NOTE_INTERVAL:
                    continue

                osc_client.send_message(
                    "/note",
                    [
                        int(pitch),
                        float(NOTE_DURATION),
                        float(amplitude)
                    ]
                )

                last_note_time = now

            except Exception as error:

                print("WebSocket error:", error)

    def run_server():

        with serve(
            handle_connection,
            WEBSOCKET_IP,
            WEBSOCKET_PORT
        ) as server:

            print(
                f"WebSocket server running on "
                f"{WEBSOCKET_IP}:{WEBSOCKET_PORT}"
            )

            server.serve_forever()

    thread = threading.Thread(
        target=run_server,
        daemon=True
    )

    thread.start()

    return state


server_state = start_websocket_server()


# ============================================================
# IMAGE PROCESSING
# ============================================================

def resize_image(image):

    image.thumbnail(
        (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)
    )

    return image


def load_video_frame(video_bytes, frame_number):

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        ) as temp_file:

            temp_file.write(video_bytes)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = cap.read()

        cap.release()

        if not success:
            return None

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)

        return resize_image(image)

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["png", "jpg", "jpeg", "mp4", "mov", "avi"]
)

if uploaded_file is None:

    st.info("Upload an image or video to start exploring.")

    st.stop()


file_bytes = uploaded_file.getvalue()

file_extension = uploaded_file.name.lower().split(".")[-1]


# ------------------------------------------------------------
# IMAGE
# ------------------------------------------------------------

if file_extension in ["png", "jpg", "jpeg"]:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = resize_image(image)


# ------------------------------------------------------------
# VIDEO
# ------------------------------------------------------------

else:

    with tempfile.NamedTemporaryFile(
        suffix=f".{file_extension}",
        delete=False
    ) as temp_file:

        temp_file.write(file_bytes)
        video_path = temp_file.name

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    os.remove(video_path)

    if total_frames <= 0:
        st.error("Could not read this video.")
        st.stop()

    st.subheader("Video frame selection")

    frame_number = st.slider(
        "Select a frame",
        min_value=0,
        max_value=total_frames - 1,
        value=0
    )

    image = load_video_frame(
        file_bytes,
        frame_number
    )

    if image is None:
        st.error("Could not extract the selected frame.")
        st.stop()

    if fps > 0:
        st.caption(
            f"Frame {frame_number} · "
            f"{frame_number / fps:.2f} seconds"
        )


# ============================================================
# PREPARE IMAGE FOR PYTHON + BROWSER
# ============================================================

image_array = np.array(image)

with server_state["lock"]:
    server_state["image"] = image_array


image_base64 = base64.b64encode(
    image_to_png := (
        lambda img: (
            img.save(
                buffer := __import__("io").BytesIO(),
                format="PNG"
            ),
            buffer.getvalue()
        )[1]
    )(image)
).decode("utf-8")


# ============================================================
# INTERACTIVE CANVAS
# ============================================================

html_code = r"""
<!DOCTYPE html>

<html>

<head>

<style>

body {
    margin: 0;
    padding: 0;
    background: transparent;
    font-family: sans-serif;
}

.container {
    width: 100%;
    text-align: center;
}

.canvas-wrapper {
    display: inline-block;
    position: relative;
    max-width: 100%;
    overflow: hidden;
    border-radius: 12px;
}

canvas {
    display: block;
    width: 100%;
    height: auto;
    cursor: crosshair;
    border-radius: 12px;
}

.status {
    margin-top: 10px;
    font-size: 13px;
    color: #888;
}

</style>

</head>

<body>

<div class="container">

    <div class="canvas-wrapper">

        <canvas id="canvas"></canvas>

    </div>

    <div class="status" id="status">
        Connecting to Sensonic...
    </div>

</div>


<script>

const imageData = "__IMAGE_DATA__";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const statusText = document.getElementById("status");

const img = new Image();

let websocket = null;

let lastSent = 0;

const SEND_INTERVAL = 30;


// --------------------------------------------------
// CONNECT TO PYTHON WEBSOCKET SERVER
// --------------------------------------------------

function connectWebSocket() {

    websocket = new WebSocket(
        "ws://127.0.0.1:8765"
    );

    websocket.onopen = function() {

        statusText.innerText =
            "Connected · Move your cursor over the image";

    };

    websocket.onclose = function() {

        statusText.innerText =
            "Disconnected from Python audio engine";

    };

    websocket.onerror = function() {

        statusText.innerText =
            "Connection error · Check the Python server";

    };

}


// --------------------------------------------------
// LOAD IMAGE
// --------------------------------------------------

img.onload = function() {

    canvas.width = img.width;
    canvas.height = img.height;

    const trailCanvas = document.createElement("canvas");

    trailCanvas.width = img.width;
    trailCanvas.height = img.height;

    const trailCtx = trailCanvas.getContext("2d");

    let mouseActive = false;


    // --------------------------------------------------
    // SEND COORDINATES TO PYTHON
    // --------------------------------------------------

    function sendCoordinates(x, y) {

        const now = performance.now();

        if (now - lastSent < SEND_INTERVAL) {
            return;
        }

        lastSent = now;

        if (
            websocket &&
            websocket.readyState === WebSocket.OPEN
        ) {

            websocket.send(
                JSON.stringify({
                    x: x / canvas.width,
                    y: y / canvas.height
                })
            );

        }

    }


    // --------------------------------------------------
    // TRACK MOUSE MOVEMENT
    // --------------------------------------------------

    canvas.addEventListener(
        "pointermove",
        function(event) {

            const rect = canvas.getBoundingClientRect();

            const x =
                (event.clientX - rect.left)
                * canvas.width / rect.width;

            const y =
                (event.clientY - rect.top)
                * canvas.height / rect.height;

            mouseActive = true;

            sendCoordinates(x, y);


            // Create a soft blurred patch
            // instead of a sharp drawing line

            trailCtx.save();

            trailCtx.beginPath();

            trailCtx.arc(
                x,
                y,
                35,
                0,
                Math.PI * 2
            );

            trailCtx.clip();

            trailCtx.filter = "blur(12px)";

            trailCtx.drawImage(
                img,
                0,
                0
            );

            trailCtx.restore();

        }
    );


    canvas.addEventListener(
        "pointerleave",
        function() {
            mouseActive = false;
        }
    );


    // --------------------------------------------------
    // ANIMATION LOOP
    // --------------------------------------------------

    function animate() {

        // Draw original image
        ctx.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );

        ctx.drawImage(
            img,
            0,
            0
        );


        // Gradually fade the blur trail
        trailCtx.save();

        trailCtx.globalCompositeOperation =
            "destination-out";

        trailCtx.fillStyle =
            "rgba(0, 0, 0, 0.035)";

        trailCtx.fillRect(
            0,
            0,
            canvas.width,
            canvas.height
        );

        trailCtx.restore();


        // Draw the fading trail over the original
        ctx.drawImage(
            trailCanvas,
            0,
            0
        );

        requestAnimationFrame(animate);

    }

    animate();

    connectWebSocket();

};


img.src = "data:image/png;base64," + imageData;

</script>

</body>

</html>
"""


html_code = html_code.replace(
    "__IMAGE_DATA__",
    image_base64
)


# ============================================================
# DISPLAY
# ============================================================

st.subheader("Live interactive instrument")

st.caption(
    "Move your cursor across the image to generate music."
)

components.html(
    html_code,
    height=760,
    scrolling=False
)


st.divider()

st.caption(
    "Sensonic · Real-time image sonification"
)