from server import server
import webview

# CONFIG
DEBUG = True

if __name__ == "__main__":
    # server additional config
    server.debug = DEBUG
    server.use_reloader = True

    # App
    window = webview.create_window("react-flask-pywebview-app", server)
    # window.evaluate_js("window.location.reload()")
    webview.start(debug=DEBUG)
