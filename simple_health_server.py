#!/usr/bin/env python3
"""Simple HTTP server for health checks"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"status": "healthy", "service": "cabruca-api"}
            self.wfile.write(json.dumps(response).encode())
        elif self.path == "/api":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"message": "Cabruca Segmentation API", "status": "running"}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = """<html>
            <body>
            <h1>Cabruca Segmentation API</h1>
            <p>Service is running</p>
            <ul>
            <li><a href="/health">/health - Health check endpoint</a></li>
            <li><a href="/api">/api - API endpoint</a></li>
            </ul>
            </body>
            </html>"""
            self.wfile.write(html.encode())

    def log_message(self, format, *args):
        # Suppress default logging
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8000), HealthHandler)
    print("Server running on port 8000", flush=True)
    server.serve_forever()
