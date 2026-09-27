"""Local recorder page: press a button, hum, get a file on disk.

A 40-line http.server serving one static page. There is deliberately no upload,
no account and no network path of any kind — `MediaRecorder` captures in the
browser and POSTs the blob to localhost, which writes it under the run directory.
Browsers only expose the microphone on a secure context, and `http://127.0.0.1`
qualifies; that is why this is a server and not a file:// page.
"""

from __future__ import annotations

import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>hum2song — record</title>
<style>
  :root { color-scheme: dark; }
  body { font: 16px/1.5 system-ui, sans-serif; background: #0f1115; color: #e6e6e6;
         display: flex; flex-direction: column; align-items: center; gap: 18px; padding: 48px 16px; }
  h1 { font-size: 22px; letter-spacing: .04em; margin: 0; }
  p.hint { color: #9aa0aa; max-width: 34em; text-align: center; margin: 0; }
  button { font: inherit; padding: 14px 28px; border-radius: 999px; border: 1px solid #2c313c;
           background: #1a1f28; color: #e6e6e6; cursor: pointer; }
  button.rec { background: #7a1f2b; border-color: #a5303f; }
  button.rec.live { animation: pulse 1.2s infinite; }
  @keyframes pulse { 50% { background: #a5303f; } }
  #meter { width: min(420px, 80vw); height: 10px; background: #1a1f28; border-radius: 5px; overflow: hidden; }
  #meter > div { height: 100%; width: 0%; background: #4caf7d; transition: width 60ms linear; }
  #status { min-height: 1.5em; color: #9aa0aa; }
  #status.ok { color: #4caf7d; } #status.err { color: #e57373; }
  audio { width: min(420px, 80vw); }
  code { background: #1a1f28; padding: 1px 6px; border-radius: 4px; }
</style></head>
<body>
<h1>hum2song</h1>
<p class="hint">Hum, whistle or sing one melodic line — no words needed yet.
5–30 seconds works best. Everything stays on this machine.</p>
<button id="rec">● Record</button>
<div id="meter"><div></div></div>
<div id="status">ready</div>
<audio id="player" controls hidden></audio>
<p class="hint" id="next" hidden>Then: <code>hum2song melody recordings/&lt;file&gt; -o run1/</code></p>
<script>
const btn = document.getElementById('rec'), status = document.getElementById('status'),
      bar = document.querySelector('#meter > div'), player = document.getElementById('player'),
      next = document.getElementById('next');
let rec = null, chunks = [], stream = null, raf = 0;

function meter() {
  const ctx = new AudioContext(), src = ctx.createMediaStreamSource(stream),
        an = ctx.createAnalyser(); an.fftSize = 512; src.connect(an);
  const data = new Uint8Array(an.frequencyBinCount);
  (function tick() { an.getByteTimeDomainData(data);
    let peak = 0; for (const v of data) peak = Math.max(peak, Math.abs(v - 128) / 128);
    bar.style.width = (peak * 100) + '%'; raf = requestAnimationFrame(tick); })();
}

btn.onclick = async () => {
  if (rec && rec.state === 'recording') { rec.stop(); return; }
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false,
        noiseSuppression: false, autoGainControl: false } }); }
  catch (e) { status.className = 'err'; status.textContent = 'microphone blocked: ' + e.message; return; }
  chunks = [];
  rec = new MediaRecorder(stream);
  rec.ondataavailable = e => e.data.size && chunks.push(e.data);
  rec.onstop = async () => {
    cancelAnimationFrame(raf); bar.style.width = '0%';
    stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(chunks, { type: rec.mimeType }),
          name = 'hum-' + new Date().toISOString().replace(/[:.]/g, '-') +
                 '.' + (rec.mimeType.includes('mp4') ? 'm4a' : 'webm');
    status.textContent = 'saving…';
    const r = await fetch('/save', { method: 'POST',
      headers: { 'Content-Type': 'audio/webm', 'X-Filename': name }, body: blob });
    const j = await r.json();
    if (r.ok) { status.className = 'ok'; status.textContent = 'saved: ' + j.path;
      player.src = URL.createObjectURL(blob); player.hidden = false; next.hidden = false; }
    else { status.className = 'err'; status.textContent = 'save failed: ' + (j.error || r.status); }
    btn.textContent = '● Record'; btn.classList.remove('live');
  };
  rec.start(); meter();
  btn.textContent = '■ Stop'; btn.classList.add('rec', 'live');
  status.className = ''; status.textContent = 'recording…';
};
</script></body></html>
"""

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


class _Handler(BaseHTTPRequestHandler):
    # Replaced by a bound subclass in serve(); the default only matters if the
    # handler class is used directly in a test.
    out_dir = Path("recordings")

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        import json

        if self.path != "/save":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            name = _SAFE_NAME.sub("_", self.headers.get("X-Filename", f"hum-{int(time.time())}.webm"))
            if not name or name in (".", ".."):
                name = f"hum-{int(time.time())}.webm"
            data = self.rfile.read(length)
            if len(data) < 512:
                raise ValueError("recording is under 512 bytes; the mic captured nothing")
            self.out_dir.mkdir(parents=True, exist_ok=True)
            target = self.out_dir / name
            if target.exists():
                target = self.out_dir / f"{target.stem}-{int(time.time())}{target.suffix}"
            target.write_bytes(data)
            payload = json.dumps({"path": str(target), "bytes": len(data)}).encode()
            self.send_response(200)
        except Exception as exc:  # a failed save must reach the page, not the console
            payload = json.dumps({"error": str(exc)}).encode()
            self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):  # keep the terminal readable
        print(f"[record] {fmt % args}")


def serve(host: str = "127.0.0.1", port: int = 8388, out: Path = Path("recordings")) -> int:
    handler = type("BoundHandler", (_Handler,), {"out_dir": out})
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"hum2song recorder on {url} — recordings land in {out.resolve()}/")
    print("Ctrl+C to stop. The page only talks to this server; nothing leaves the machine.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0
