"""Browser UI: local page fed by Server-Sent Events (pushed, no polling delay)."""

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eloquent</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#8a8a85;--card:#fff;--line:#e4e3de;--fix:#ffe27a;--acc:#2b6cb0;--warn:#a15c00}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#8f8e88;--card:#1f1f1d;--line:#32312e;--fix:#6b5600;--acc:#7db4ff;--warn:#f0b35a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:18px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:14px}
header b{font-size:15px}
.dot{width:9px;height:9px;border-radius:50%;background:#888;display:inline-block;margin-right:6px}
.dot.listening{background:#2e9e5b}.dot.stopping{background:#d4a017}.dot.off{background:#c0392b}
header label{color:var(--mut);cursor:pointer}
header button,header select{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:4px 12px;cursor:pointer}
header button:disabled{opacity:.5;cursor:default}
#err{color:var(--warn);font-size:13px}
main{max-width:860px;margin:0 auto;padding:20px 20px 120px}
.report{background:var(--card);border:2px solid var(--acc);border-radius:12px;padding:16px 20px;margin:0 0 18px;white-space:pre-wrap}
.report b.title{display:block;font-size:14px;color:var(--acc);margin-bottom:8px}
.para{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin:0 0 12px}
.para p{margin:0}
.para .time{display:block;font-size:12px;color:var(--mut);margin-bottom:2px}
.para.cleaning p{color:var(--mut);font-style:italic}
.para.cleaning .time::after{content:" - polishing...";color:var(--acc)}
.para .note{display:block;font-size:12px;color:var(--warn);margin-top:4px}
.spk{display:inline-block;font-size:11px;font-weight:600;padding:0 6px;border-radius:4px;margin-right:8px;background:rgba(128,128,128,.15);cursor:pointer}
.spk0{color:#2b6cb0}.spk1{color:#1f7a4c}.spk2{color:#a15c00}.spk3{color:#9d2d6b}.spk4{color:#6b46c1}.spk5{color:#c05621}
@media (prefers-color-scheme: dark){.spk0{color:#7db4ff}.spk1{color:#6ee7b7}.spk2{color:#fbbf24}.spk3{color:#f472b6}.spk4{color:#c084fc}.spk5{color:#fb923c}}
.fix{border-bottom:2px solid var(--fix)}
.fresh .fix{animation:flash 3.5s ease-out}
@keyframes flash{0%{background:var(--fix)}100%{background:transparent}}
.loose{margin:0 0 6px}
.loose .time{font-size:12px;color:var(--mut);margin-right:8px}
.partial{color:var(--mut);font-style:italic;margin-top:6px}
</style></head>
<body>
<header>
  <b>Eloquent</b>
  <span><span class="dot" id="dot"></span><span id="status">connecting</span></span>
  <button id="toggle" disabled>Start</button>
  <select id="src" title="Audio source">
    <option value="mic">Mic only</option>
    <option value="system">System audio</option>
    <option value="both">Mic + System</option>
  </select>
  <label><input type="checkbox" id="raw"> show raw</label>
  <select id="fmt">
    <option value="summary">Concise Summary</option>
    <option value="bullets">Key Bullet Points</option>
    <option value="meeting">Meeting Minutes</option>
    <option value="lecture">Lecture / Study Notes</option>
    <option value="report">Structured Report</option>
    <option value="article">Clean Article / Memo</option>
    <option value="custom">Custom Prompt…</option>
  </select>
  <button id="gen">Generate</button>
  <button id="rename" hidden>Rename speakers</button>
  <button id="copy">Copy text</button>
  <span id="err"></span>
</header>
<main id="log"></main>
<script>
let state=null, showRaw=false; const seenDone=new Set();
const $=id=>document.getElementById(id);
function el(tag,cls,text){const e=document.createElement(tag); if(cls)e.className=cls; if(text!==undefined)e.textContent=text; return e;}
function spkBadge(spk,spkId){
  if(!spk)return null;
  const key=spkId||spk, n=parseInt((key.split(' ').pop())||'1',10)||1;
  const idx=key==='You'?5:((n-1)%6);
  const b=el('span','spk spk'+idx,spk);
  b.title='Click to rename '+spk;
  b.addEventListener('click',()=>renameOne(key,spk));
  return b;
}
function postRename(names){fetch('/rename',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({names})});}
function renameOne(spkId,cur){
  const v=prompt('Name for '+spkId+' (leave empty to reset):',cur===spkId?'':cur);
  if(v!==null) postRename({[spkId]:v.trim()});
}
function allText(s){
  const out=s.paragraphs.map(p=>{
    const txt=(p.tokens&&!showRaw)?p.tokens.map(t=>t[0]).join(' '):p.raw;
    return (s.multi_speaker&&p.speaker)?(p.speaker+': '+txt):txt;
  });
  if(s.loose.length) out.push(s.loose.map(l=>(s.multi_speaker&&l.speaker)?(l.speaker+': '+l.text):l.text).join(' '));
  return out.join('\n\n');
}
function render(s){
  const nearBottom=window.innerHeight+window.scrollY>=document.body.offsetHeight-120;
  const box=$('log'); box.textContent='';
  if(s.report){
    const r=el('div','report');
    r.append(el('b','title',s.report.label+(s.report.status==='generating'?' (generating…)':'')));
    r.append(document.createTextNode(s.report.status==='error'?s.report.error:s.report.text));
    box.append(r);
  }
  for(const p of s.paragraphs){
    const d=el('div','para '+p.status);
    const meta=el('span','time');
    if(s.multi_speaker&&p.speaker){const b=spkBadge(p.speaker,p.speaker_id); if(b)meta.append(b);}
    meta.append(document.createTextNode(p.time));
    d.append(meta);
    const body=el('p');
    if(p.tokens && !showRaw){
      const fresh=!seenDone.has(p.id); if(fresh){d.classList.add('fresh'); seenDone.add(p.id);}
      p.tokens.forEach(([w,f])=>{body.append(el('span',f?'fix':'',w),' ');});
    } else body.textContent=p.raw;
    d.append(body);
    if(p.note) d.append(el('span','note',p.note));
    box.append(d);
  }
  for(const l of s.loose){
    const d=el('div','loose');
    if(s.multi_speaker&&l.speaker){const b=spkBadge(l.speaker,l.speaker_id); if(b)d.append(b);}
    d.append(el('span','time',l.t), document.createTextNode(l.text));
    box.append(d);
  }
  if(s.partial) box.append(el('div','partial','... '+s.partial));
  $('rename').hidden=!s.multi_speaker;
  if(s.audio_source){$('src').value=s.audio_source;}
  $('src').disabled=(s.status!=='idle');
  $('err').textContent=s.error?('LLM: '+s.error):'';
  $('dot').className='dot '+s.status; $('status').textContent=s.status;
  const t=$('toggle'); t.disabled=!(s.status==='idle'||s.status==='listening');
  t.textContent=s.status==='listening'?'Stop':'Start';
  if(nearBottom) window.scrollTo(0,document.body.scrollHeight);
}
function connect(){
  const es=new EventSource('/events');
  es.onmessage=e=>{state=JSON.parse(e.data); render(state);};
  es.onerror=()=>{$('dot').className='dot off'; $('status').textContent='disconnected';};
}
$('raw').addEventListener('change',e=>{showRaw=e.target.checked; if(state)render(state);});
$('copy').addEventListener('click',()=>{if(state)navigator.clipboard.writeText(state.report&&state.report.text?state.report.text:allText(state));});
$('toggle').addEventListener('click',()=>fetch('/toggle',{method:'POST'}));
$('src').addEventListener('change',e=>fetch('/source',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:e.target.value})}));
$('rename').addEventListener('click',()=>{
  if(!state||!state.speakers||!state.speakers.length)return;
  const names={};
  for(const sp of state.speakers){
    const v=prompt('Name for '+sp.id+' (leave empty to keep/reset):',sp.name===sp.id?'':sp.name);
    if(v===null)return;
    names[sp.id]=v.trim();
  }
  postRename(names);
});
$('gen').addEventListener('click',()=>{
  const fid=$('fmt').value;
  let custom='';
  if(fid==='custom'){custom=prompt('How should the LLM format or analyze this transcript?','Summarize key points')||''; if(!custom)return;}
  fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({format_id:fid,custom_prompt:custom})});
});
connect();
</script></body></html>
"""


class WebUI:
    def __init__(self, engine, port: int):
        self.engine = engine
        self.cond = threading.Condition()
        self.version = 0
        engine.subscribe(self._on_event)
        self.url = self._serve(port)

    def _on_event(self, kind, data):
        if kind == "log":
            return
        with self.cond:
            self.version += 1
            self.cond.notify_all()

    def payload(self) -> bytes:
        e = self.engine
        return e.state.snapshot_json({
            "status": e.status, "llm": e.llm, "audio_source": e.settings.audio_source,
        }).encode("utf-8")

    def _serve(self, port: int) -> str:
        ui = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/events":
                    return self._events()
                if self.path in ("/", "/index.html"):
                    body = PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(404)

            def do_POST(self):
                if self.path == "/toggle":
                    ui.engine.toggle()
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path == "/generate":
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                    ui.engine.generate_report(body.get("format_id", "summary"), body.get("custom_prompt", ""))
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path == "/rename":
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                    ui.engine.rename_speakers(body.get("names") or {})
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path == "/source":
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                    src = body.get("source")
                    if src in ("mic", "system", "both") and ui.engine.status == "idle":
                        ui.engine.settings.audio_source = src
                        ui.engine.settings.save()
                        ui._on_event("source", src)
                    self.send_response(204)
                    self.end_headers()
                    return
                self.send_error(404)

            def _events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with ui.cond:
                            if seen == ui.version:
                                ui.cond.wait(15)
                            changed = seen != ui.version
                            seen = ui.version
                        if changed:
                            self.wfile.write(b"data: " + ui.payload() + b"\n\n")
                        else:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        time.sleep(0.03)   # coalesce bursts of events (~30 updates/s max)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass

            def log_message(self, *a):
                pass

        for p in range(port, port + 10):
            try:
                srv = ThreadingHTTPServer(("127.0.0.1", p), Handler)
                srv.daemon_threads = True
                threading.Thread(target=srv.serve_forever, daemon=True).start()
                return f"http://127.0.0.1:{p}"
            except OSError:
                continue
        raise RuntimeError("could not open a local port for the browser view")


def run_web(engine, port: int, open_browser: bool = True, wav_path=None):
    """Browser view + console log. Ctrl+C stops."""
    def on_log(kind, data):
        if kind == "log":
            print(data[1], flush=True)
        elif kind == "session_ended":
            print(f"Stopped. Files in: {data['dir']}", flush=True)

    engine.subscribe(on_log)
    ui = WebUI(engine, port)
    engine.load()
    print(f"Browser view: {ui.url}   (Ctrl+C to quit)", flush=True)
    if open_browser:
        webbrowser.open(ui.url)
    engine.start(wav_path)
    try:
        while True:
            time.sleep(0.5)
            if wav_path and engine.status == "idle":
                time.sleep(1.0)
                break
    except KeyboardInterrupt:
        print("Stopping... finishing the last sentence and cleanup", flush=True)
        engine.stop()
        engine.wait_idle(40)
        time.sleep(0.5)  # let the browser pick up the final state
