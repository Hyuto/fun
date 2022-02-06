import sys
import os
import subprocess
import socket
import signal
import webview
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class AppDebug:
    reactapp = os.path.join(".", "dist")

    def __init__(self):
        self._get_flask_dev_port()
        # window.evaluate_js("window.location.reload()")

    def _get_flask_dev_port(self):
        # get available port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", 0))
        self.port = str(sock.getsockname()[1])
        sock.close()

    def start(self):
        self.server = subprocess.Popen(
            [sys.executable, "server.py", "-p", self.port],
            stdout=subprocess.PIPE,
            shell=True,
        )
        self.react = subprocess.Popen("yarn dev", stdout=subprocess.PIPE, shell=True)

    def stop(self):
        # Windows
        os.kill(self.server.pid, signal.CTRL_C_EVENT)
        os.kill(self.react.pid, signal.CTRL_C_EVENT)


if __name__ == "__main__":
    # watcher
    dev = AppDebug()
    dev.start()

    # App
    window = webview.create_window("react-flask-pywebview-app", f"http://localhost:{dev.port}/")
    webview.start(debug=True)

    # stopping watcher
    dev.stop()
