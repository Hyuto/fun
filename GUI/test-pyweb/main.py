import webview
from flask import Flask, send_from_directory

DEBUG = False

server = Flask("__main__", template_folder="frontend", static_folder="frontend/static")


@server.route("/")
def serve():
    return send_from_directory(server.template_folder, "index.html")


window = webview.create_window("test-pyweb", server)
webview.start(debug=DEBUG)
