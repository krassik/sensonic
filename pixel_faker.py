
#!/usr/bin/env python3

import argparse
import math
import time

from pythonosc.udp_client import SimpleUDPClient


WIDTH = 1920
HEIGHT = 1080


def pixel_generator(client, fps=30, duration=30):

    start = time.monotonic()
    frame = 0
    interval = 1.0 / fps

    print("Sending OSC pixel stream...")
    print("Address: /pixel")

    while True:

        now = time.monotonic()
        elapsed = now - start

        if duration and elapsed >= duration:
            break

        # Simulate a moving pixel
        phase = elapsed * 2 * math.pi * 0.3

        x = int((math.sin(phase) + 1) * 0.5 * (WIDTH - 1))
        y = int((math.cos(phase * 0.7) + 1) * 0.5 * (HEIGHT - 1))

        # Simulate changing RGB values
        r = int((math.sin(phase) + 1) * 127.5)
        g = int((math.sin(phase + 2.1) + 1) * 127.5)
        b = int((math.sin(phase + 4.2) + 1) * 127.5)

        # OSC message
        client.send_message(
            "/pixel",
            [x, y, r, g, b, float(elapsed)]
        )

        if frame % fps == 0:
            print(
                f"t={elapsed:6.2f}s "
                f"pixel=({x:4d},{y:4d}) "
                f"RGB=({r:3d},{g:3d},{b:3d})"
            )

        frame += 1

        # Maintain target frame rate
        target = start + frame * interval
        time.sleep(max(0, target - time.monotonic()))

    print(f"Finished. Sent {frame} messages.")


def main():

    parser = argparse.ArgumentParser(
        description="Generate synthetic OSC pixel messages"
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=30)

    args = parser.parse_args()

    client = SimpleUDPClient(args.host, args.port)

    pixel_generator(
        client,
        fps=args.fps,
        duration=args.duration
    )


if __name__ == "__main__":
    main()
