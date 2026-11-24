"""Module 6: the web app. Standard library only (no framework to install).

  GET  /                 the UI
  GET  /api/papers       corpus listing
  GET  /api/passage?id=  one passage (for citation click-through)
  POST /api/ask          {"question": "..."}  -> cited, verified answer (or abstention)
  POST /api/check        {"claim": "..."}     -> evidence assessment across studies

Swapping to FastAPI later is mechanical: each handler below is a pure function of
(system, payload).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .pipeline import Sciproof, answer_to_dict

STATIC = Path(__file__).parent / "static" / "index.html"
MAX_BODY = 10_000


def handle_ask(system: Sciproof, payload: dict) -> dict:
    q = str(payload.get("question", "")).strip()
    if not q or len(q) > 1000:
        raise ValueError("question must be 1-1000 characters")
    return answer_to_dict(system.ask(q))


def handle_check(system: Sciproof, payload: dict) -> dict:
    c = str(payload.get("claim", "")).strip()
    if not c or len(c) > 1000:
        raise ValueError("claim must be 1-1000 characters")
    a = system.check_claim(c)
    return {"verdict": a.verdict, "explanation": a.explanation, "studies": [asdict(s) for s in a.studies]}


def handle_papers(system: Sciproof) -> dict:
    return {"papers": [dict(paper_id=p.paper_id, title=p.title, year=p.year, study_type=p.study_type)
                       for p in system.papers.values()]}


def handle_passage(system: Sciproof, cid: str) -> dict | None:
    c = system.by_id.get(cid)
    return None if c is None else dict(chunk_id=c.chunk_id, paper_id=c.paper_id, title=c.title,
                                       section=c.section, study_type=c.study_type, year=c.year, text=c.text)


def make_handler(system: Sciproof):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", ctype + "; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.send_header("x-content-type-options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(200, STATIC.read_bytes(), "text/html")
            elif u.path == "/api/papers":
                self._send(200, handle_papers(system))
            elif u.path == "/api/passage":
                p = handle_passage(system, parse_qs(u.query).get("id", [""])[0])
                self._send(200 if p else 404, p or {"error": "not found"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("content-length", 0))
                if n > MAX_BODY:
                    return self._send(413, {"error": "payload too large"})
                payload = json.loads(self.rfile.read(n) or b"{}")
                fn = {"/api/ask": handle_ask, "/api/check": handle_check}.get(urlparse(self.path).path)
                if not fn:
                    return self._send(404, {"error": "not found"})
                self._send(200, fn(system, payload))
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *a):  # quiet
            pass
    return H


def serve(system: Sciproof, host="127.0.0.1", port=8000):
    srv = ThreadingHTTPServer((host, port), make_handler(system))
    print(f"Sciproof on http://{host}:{port}  (corpus: {len(system.papers)} papers, mode: {system.mode})")
    srv.serve_forever()
