import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from better_cheaper_llm.jev import DEFAULT_BASE_URL, JevClient, JevError


@pytest.fixture
def server():
    """A local stand-in for the System One API. Set `statuses` to script responses."""
    seen, statuses = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            status = statuses.pop(0) if statuses else 200
            payload = {"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": 0.9}},
                       "usage": {"input_tokens": 12, "output_tokens": 20}} if status == 200 else {"error": "busy"}
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", seen, statuses
    httpd.shutdown()


def test_request_shape_and_auth(server):
    url, seen, _ = server
    response = JevClient(api_key="k", base_url=url).evaluate("state", {"q": {"type": "noul", "instructions": "?"}})
    assert response["answers"]["q"]["noul"] == 0.9
    assert seen[0]["path"] == "/v1/systemone" and seen[0]["auth"] == "Bearer k"
    assert seen[0]["body"] == {"state": "state", "model": "jev-latest",
                               "questions": {"q": {"type": "noul", "instructions": "?"}}}


def test_retries_when_overloaded(server, monkeypatch):
    monkeypatch.setattr("better_cheaper_llm.jev.time.sleep", lambda s: None)
    url, seen, statuses = server
    statuses.extend([429, 529])
    assert JevClient(base_url=url).evaluate("s", {})["usage"]["input_tokens"] == 12
    assert len(seen) == 3 and seen[0]["auth"] is None  # no key needed for a self-hosted server


def test_gives_up_with_a_readable_error(server, monkeypatch):
    monkeypatch.setattr("better_cheaper_llm.jev.time.sleep", lambda s: None)
    url, _, statuses = server
    statuses.extend([422])
    with pytest.raises(JevError, match="422"):
        JevClient(base_url=url).evaluate("s", {})


def test_typesafe_endpoint_needs_a_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        JevClient()
    assert JevClient(api_key="k").base_url == DEFAULT_BASE_URL
