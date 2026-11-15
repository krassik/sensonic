import cv2
import numpy as np
import pandas as pd
import time
import math
from pythonosc.udp_client import SimpleUDPClient

class VisualInstrument:

    def __init__(
        self,
        file_path,
        effect_radius=35,
        effect_duration=2.0,
        blur_strength=12,
        min_distance=8,
        max_events=150,
        synth = SimpleUDPClient("127.0.0.1", 9000)
    ):

        # ------------------------------------------
        # Synth configuration
        # ------------------------------------------

        self.synth = synth

        # ------------------------------------------
        # Media configuration
        # ------------------------------------------

        self.file_path = file_path
        self.cap = None
        self.is_video = False

        self.image = None
        self.fps = None
        self.total_frames = 1
        self.current_frame = 0
        self.requested_frame = 0

        self.height = None
        self.width = None

        # ------------------------------------------
        # Mouse tracking
        # ------------------------------------------

        self.last_position = None
        self.events = []

        # ------------------------------------------
        # Visual effects
        # ------------------------------------------

        self.blur_events = []

        self.effect_radius = effect_radius
        self.effect_duration = effect_duration
        self.blur_strength = blur_strength
        self.min_distance = min_distance
        self.max_events = max_events

        # ------------------------------------------
        # Application configuration
        # ------------------------------------------

        self.window = "Visual Instrument"
        self.start_time = None
        self.running = False

        # Load the selected media
        self.load_media()

    # ==================================================
    # 1. MEDIA LOADING
    # ==================================================

    def load_media(self):

        # Try loading as an image first
        image = cv2.imread(self.file_path)

        if image is not None:

            self.image = image
            self.is_video = False

        else:

            # Otherwise, try loading as a video
            self.cap = cv2.VideoCapture(self.file_path)

            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                raise ValueError(
                    f"Unable to open file: {self.file_path}"
                )

            self.is_video = True

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            if self.fps <= 0:
                self.fps = 30.0

            self.total_frames = int(
                self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            ret, frame = self.cap.read()

            if not ret:
                self.cap.release()
                self.cap = None
                raise ValueError("Unable to read video")

            self.image = frame

        self.height, self.width = self.image.shape[:2]

        print("Media loaded successfully")
        print(f"Media type: {'Video' if self.is_video else 'Image'}")
        print(f"Resolution: {self.width} x {self.height}")

        if self.is_video:
            print(f"FPS: {self.fps}")
            print(f"Total frames: {self.total_frames}")

    # ==================================================
    # 2. MOUSE CALLBACK
    # ==================================================

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            # print(f"Mouse clicked at ({x}, {y})")
            self.synth.send_message("/melody", ["A#5", "C6", "G#5", "D#6", "D#5"])
            return

        # Mouse scrolling: video navigation
        if self.is_video and event == cv2.EVENT_MOUSEWHEEL:

            delta = (flags >> 16) & 0xFFFF

            # Convert unsigned value to signed value
            if delta >= 32768:
                delta -= 65536

            if delta > 0:
                self.requested_frame += 1

            elif delta < 0:
                self.requested_frame -= 1

            self.requested_frame = max(
                0,
                min(
                    self.total_frames - 1,
                    self.requested_frame
                )
            )

            return

        # Mouse movement: pixel tracking
        if event == cv2.EVENT_MOUSEMOVE:

            if not (
                0 <= x < self.width
                and 0 <= y < self.height
            ):
                return

            self.track_pixels(x, y)

    # ==================================================
    # 3. PIXEL INTERPOLATION
    # ==================================================

    def interpolate_points(self, p1, p2):

        x1, y1 = p1
        x2, y2 = p2

        distance = max(
            abs(x2 - x1),
            abs(y2 - y1)
        )

        if distance == 0:
            return []

        points = []

        for i in range(1, distance + 1):

            x = round(x1 + (x2 - x1) * i / distance)
            y = round(y1 + (y2 - y1) * i / distance)

            points.append((x, y))

        return points

    # ==================================================
    # 4. PIXEL TRACKING
    # ==================================================

    def track_pixels(self, x, y):

        # Determine positions to record
        if self.last_position is None:

            points = [(x, y)]

        else:

            distance = math.hypot(
                x - self.last_position[0],
                y - self.last_position[1]
            )

            if distance < self.min_distance:
                return

            points = self.interpolate_points(
                self.last_position,
                (x, y)
            )

        # Capture timing information
        timestamp = time.perf_counter() - self.start_time

        frame_number = self.current_frame

        if self.is_video:
            video_timestamp_ms = (
                frame_number / self.fps
            ) * 1000.0
        else:
            video_timestamp_ms = None

        # Record pixel information
        for px, py in points:

            b, g, r = self.image[py, px]

            self.events.append({

                "timestamp": timestamp,

                "video_timestamp_ms": video_timestamp_ms,

                "frame_number": frame_number,

                "x": px,
                "y": py,

                "r": int(r),
                "g": int(g),
                "b": int(b)

            })

        # Register visual effect
        self.blur_events.append(
            (x, y, time.perf_counter())
        )

        if len(self.blur_events) > self.max_events:
            del self.blur_events[0]

        self.last_position = (x, y)

    # ==================================================
    # 5. VIDEO NAVIGATION
    # ==================================================

    def navigate_video(self):

        if not self.is_video:
            return

        if self.requested_frame == self.current_frame:
            return

        target_frame = self.requested_frame

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            target_frame
        )

        ret, frame = self.cap.read()

        if ret:

            self.image = frame
            self.current_frame = target_frame

            # Reset mouse interpolation across frames
            self.last_position = None

        else:

            self.requested_frame = self.current_frame

    # ==================================================
    # 6. VISUAL EFFECTS
    # ==================================================

    def render_effects(self):

        now = time.perf_counter()

        display = self.image.copy()

        # Remove expired effects
        self.blur_events[:] = [

            event for event in self.blur_events

            if now - event[2] < self.effect_duration

        ]

        if not self.blur_events:
            return display

        # Create effect mask
        mask = np.zeros(
            (self.height, self.width),
            dtype=np.float32
        )

        for x, y, timestamp in self.blur_events:

            age = now - timestamp

            strength = 1.0 - age / self.effect_duration

            cv2.circle(
                mask,
                (x, y),
                self.effect_radius,
                float(strength),
                thickness=-1,
                lineType=cv2.LINE_AA
            )

        # Smooth the mask
        mask = cv2.GaussianBlur(
            mask,
            (0, 0),
            sigmaX=self.effect_radius / 3
        )

        mask = np.clip(mask, 0, 1)

        # Apply Gaussian blur to current frame
        blurred = cv2.GaussianBlur(
            self.image,
            (0, 0),
            sigmaX=self.blur_strength
        )

        # Blend original and blurred images
        mask = mask[:, :, np.newaxis]

        display = (
            self.image * (1 - mask)
            + blurred * mask
        ).astype(np.uint8)

        return display

    # ==================================================
    # 7. DISPLAY
    # ==================================================

    def display(self):

        display = self.render_effects()

        if self.is_video:

            frame_text = (
                f"Frame: {self.current_frame + 1}"
                f" / {self.total_frames}"
            )

            cv2.putText(
                display,
                frame_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.imshow(self.window, display)

    # ==================================================
    # 8. MAIN APPLICATION LOOP
    # ==================================================

    def run(self):

        self.start_time = time.perf_counter()
        self.running = True

        try:

            cv2.namedWindow(
                self.window,
                cv2.WINDOW_AUTOSIZE
            )

            cv2.setMouseCallback(
                self.window,
                self.mouse_callback
            )

            print("\nVisual Instrument started")
            print("Move your mouse to track pixels")
            print("Scroll to navigate video frames")
            print("Press Q or ESC to exit\n")

            while self.running:

                # Navigate video if necessary
                self.navigate_video()

                # Render and display
                self.display()

                # Process GUI events
                key = cv2.waitKeyEx(15)

                if (key & 0xFF) in (
                    ord("q"),
                    ord("Q"),
                    27
                ):
                    self.running = False

                # Detect window closure
                try:

                    if cv2.getWindowProperty(
                        self.window,
                        cv2.WND_PROP_VISIBLE
                    ) < 1:

                        self.running = False

                except cv2.error:

                    self.running = False

        except KeyboardInterrupt:

            print("Interrupted by user")

        finally:

            self.close()

    # ==================================================
    # 9. EXPORT EVENTS
    # ==================================================

    def get_events(self):

        return pd.DataFrame(self.events)

    # ==================================================
    # 10. CLEANUP
    # ==================================================

    def close(self):

        self.running = False

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()
        cv2.waitKey(1)

        print(
            f"Visual Instrument closed. "
            f"Captured {len(self.events)} pixel events."
        )

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visual Instrument"
    )
    parser.add_argument(
        "file",
        help="Path to an image or video file"
    )
    args = parser.parse_args()
   
    instrument = VisualInstrument(args.file)
    instrument.run()


if __name__ == "__main__":
    main()