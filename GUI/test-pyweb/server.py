import socket
from flask import Flask
from config import MAIN_DIR, DEBUG

server = Flask(__name__, static_folder=MAIN_DIR, static_url_path="/")


@server.route("/")
def hello():
    return "Hallo World"


if __name__ == "__main__":
    # get available port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()

    # run server
    server.run(port=port, threaded=True, debug=DEBUG)
