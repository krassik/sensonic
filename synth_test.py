from pythonosc.udp_client import SimpleUDPClient
synth = SimpleUDPClient("127.0.0.1", 9000)
synth.send_message("/melody", ["A#5", "C6", "G#5", "D#6", "D#5"])

