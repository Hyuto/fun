import os
import json
import argparse
import webview
from flask import Flask, render_template, jsonify, request
from functools import wraps
import getpass  # get user

MAIN_DIR = os.path.join(".", "dist")

server = Flask(__name__, template_folder=MAIN_DIR, static_folder=MAIN_DIR, static_url_path="/")
server.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
server.debug = False


def verify_token(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        data = json.loads(request.data)
        token = data.get("token")
        if token == webview.token:
            return function(*args, **kwargs)
        else:
            raise Exception("Authentication error")

    return wrapper


@server.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@server.route("/", defaults={"path": ""})
@server.route("/<path:path>")
def serve(path):
    return render_template("index.html", token=webview.token)


@server.route("/init", methods=["POST"])
@verify_token
def initialize():
    return jsonify({"user": getpass.getuser()})


def DEV(port):
    # run server
    server.run(port=port, threaded=True, debug=True, reloader_type="stat")


if __name__ == "__main__":
    # Development
    parser = argparse.ArgumentParser(description="Flask Server")
    parser.add_argument("-p", "--dev-port", help="Development Port", type=str)
    args = parser.parse_args()
    DEV(args.dev_port)
