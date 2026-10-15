
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = {
            "status": "ok",
            "message": "Brain CT research demo API is running.",
            "model": "nakshatrika/brain-tumor-vit-ct",
            "note": "This endpoint is a deployment check, not a prediction service."
        }
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
