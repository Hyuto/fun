import os
import webview
from flask import Flask, send_from_directory

DEBUG = True
MAIN_DIR = os.path.join(".", "front", "dist")

if DEBUG:
    os.system('start cmd /K "cd front && npm start"')

server = Flask("__main__", static_folder=MAIN_DIR, static_url_path="/")


@server.route("/")
def serve():
    return send_from_directory(server.static_folder, "index.html")


if __name__ == "__main__":
    window = webview.create_window("test-pyweb", server)
    webview.start(debug=DEBUG)
