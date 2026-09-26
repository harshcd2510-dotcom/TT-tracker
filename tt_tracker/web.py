"""Browser output: MJPEG stream + live stats page.

Serve the annotated feed at http://localhost:PORT/ - the video is a
multipart JPEG stream (<img src="/stream">) and /stats.json returns the
current engine stats, polled once a second by the page.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = """<!doctype html>
<html><head><title>TT Tracker</title><style>
  body{background:#111;color:#ddd;font-family:monospace;margin:0;
       display:flex;flex-direction:column;align-items:center}
  h2{font-weight:normal;margin:12px 0 4px}
  #stats{display:flex;gap:18px;flex-wrap:wrap;justify-content:center;
         max-width:960px;padding:6px 0;font-size:14px}
  #stats b{color:#7fd4ff}
  img{max-width:96vw;max-height:78vh;border:1px solid #333;
      border-radius:4px}
</style></head><body>
<h2>TT Tracker</h2>
<div id="stats"></div>
<img src="/stream">
<script>
const el = document.getElementById('stats');
const keys = [["shots_left","Shots L"],["shots_right","Shots R"],
  ["bounces","Bounces"],["rallies","Rallies"],["longest_rally","Longest"],
  ["avg_rally","Avg rally"],["top_speed_mps","Top m/s"]];
async function tick(){
  try{
    const s = await (await fetch('/stats.json')).json();
    el.innerHTML = keys.map(([k,l]) =>
      `<span>${l}: <b>${(typeof s[k]==='number'?s[k].toFixed(1):s[k])}</b></span>`
    ).join('');
  }catch(e){}
}
setInterval(tick,1000); tick();
</script></body></html>"""


class _State:
    def __init__(self):
        self.jpeg = None
        self.stats = {}
        self.cond = threading.Condition()

    def push_frame(self, img, stats):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self.cond:
            self.jpeg = buf.tobytes()
            self.stats = stats
            self.cond.notify_all()


def make_server(state: _State, port: int) -> ThreadingHTTPServer:

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/html"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE)
            elif self.path == "/stats.json":
                self._send(200, json.dumps(state.stats), "application/json")
            elif self.path == "/stream":
                self._mjpeg()
            else:
                self._send(404, "not found", "text/plain")

        def _mjpeg(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with state.cond:
                        state.cond.wait(timeout=2.0)
                        jpg = state.jpeg
                    if jpg is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpg)).encode()
                        + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
