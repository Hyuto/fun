import os, sys
import time, signal
import webview
from subprocess import Popen, PIPE
from config import MAIN_DIR, DEBUG

if DEBUG:
    front = Popen("yarn dev -p 8080", stdout=PIPE, shell=True)
    server = Popen([sys.executable, "server.py"], stdout=PIPE)


if __name__ == "__main__":
    if DEBUG:
        time.sleep(1)
        # os.path.join(MAIN_DIR, "index.html")

    window = webview.create_window(
        "react-flask-pywebview app", "http://localhost:8080/", width=600, height=1200
    )
    webview.start(http_server=True, debug=DEBUG)

    os.kill(front.pid, signal.CTRL_C_EVENT)
    os.kill(server.pid, signal.CTRL_C_EVENT)
