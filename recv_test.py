
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

def pixel_handler(address, *args):
    print(address, args)

dispatcher = Dispatcher()
dispatcher.map("/pixel", pixel_handler)

server = BlockingOSCUDPServer(
    ("127.0.0.1", 9000),
    dispatcher
)

print("Listening on port 9000...")
server.serve_forever()
