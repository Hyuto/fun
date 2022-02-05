import os
import time
import webview
from flask import Flask, send_from_directory

# CONFIG
DEBUG = True
MAIN_DIR = os.path.join(".", "dist")

# SERVER
server = Flask(__name__, static_folder=MAIN_DIR, static_url_path="/")
server.debug = DEBUG


@server.route("/", defaults={"path": ""})
@server.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(server.static_folder + "/" + path):
        return send_from_directory(server.static_folder, path)
    else:
        return send_from_directory(server.static_folder, "index.html")


if __name__ == "__main__":
    window = webview.create_window("react-flask-pywebview app", server)
    # window.evaluate_js("window.location.reload()")
    webview.start(debug=DEBUG)
