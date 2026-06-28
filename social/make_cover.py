"""Build a dark terminal-style cover (1600x640) with the exact Mesh API logo,
JetBrains Mono coding font, and Mesh brand purple. Writes cover.html; render with Chrome."""
import os, re

HERE = os.path.dirname(__file__)
LOGO_PATH = "/Users/raushan/Desktop/to delete/rough_repo/tetris-opus-arena/tetris_contest_v2/assets/mesh-mark.svg"

with open(LOGO_PATH) as f:
    logo_svg = f.read()
# drop fixed width/height so CSS controls size; keep the viewBox + gradients intact
logo_svg = re.sub(r'width="\d+"\s*height="\d+"', 'width="100%" height="100%"', logo_svg, count=1)

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&display=swap" rel="stylesheet">
<style>
  :root{ --bg:#0a0c10; --line:#20242d; --ink:#e8eaed; --mut:#8b929c;
         --p:#a78bff; --pd:#7c5cf0; --green:#5fcf80; }
  *{box-sizing:border-box; margin:0; padding:0;}
  html,body{background:#000;}
  .cover{ width:1600px; height:640px; background:var(--bg);
    font-family:'JetBrains Mono',ui-monospace,Menlo,monospace; color:var(--ink);
    display:flex; flex-direction:column; overflow:hidden; position:relative; }
  .cover::before{ content:""; position:absolute; inset:0; opacity:.55;
    background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),
      linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
    background-size:34px 34px; }
  .bar{ height:50px; display:flex; align-items:center; gap:9px; padding:0 22px;
    border-bottom:1px solid var(--line); position:relative; z-index:1; }
  .dot{ width:12px; height:12px; border-radius:50%; background:#3a3f49; }
  .bar .tt{ margin-left:14px; color:var(--mut); font-size:15px; }
  .body{ flex:1; padding:30px 48px 30px; position:relative; z-index:1;
    display:flex; flex-direction:column; justify-content:center; }
  .brand{ display:flex; align-items:center; gap:14px; margin-bottom:16px; }
  .brand .logo{ width:48px; height:48px; display:block; }
  .brand .name{ font-size:27px; font-weight:800; }
  .brand .name b{ color:var(--p); }
  .brand .cur{ width:11px; height:25px; background:var(--p); display:inline-block;
    margin-left:2px; transform:translateY(3px); animation:bl 1.1s steps(1) infinite; }
  @keyframes bl{ 50%{opacity:0;} }
  .prompt{ color:var(--mut); font-size:17px; margin-bottom:18px; }
  .prompt .pp{ color:var(--p); }
  .prompt .dollar{ color:var(--green); }
  h1{ font-size:41px; line-height:1.16; font-weight:800; letter-spacing:-.01em; }
  .stats{ display:flex; gap:54px; margin-top:40px; }
  .stat .big{ font-size:47px; font-weight:800; color:var(--p); line-height:1; }
  .stat .lab{ font-size:14px; color:var(--mut); margin-top:9px; }
  .foot{ margin-top:20px; color:var(--mut); font-size:16px; }
  .foot .arrow{ color:var(--p); }
</style></head><body>
<div class="cover">
  <div class="bar">
    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    <span class="tt">mesh-api ~ research/compress-at-the-gate</span>
  </div>
  <div class="body">
    <div class="brand">
      <span class="logo">__LOGO__</span>
      <span class="name">mesh<b>_api</b></span>
      <span class="cur"></span>
    </div>
    <div class="prompt"><span class="dollar">$</span> mesh bench <span class="pp">--gateway</span> --compress classical <span class="pp">--vs</span> neural</div>
    <h1>At the gateway, 1970s NLP beat<br>a neural prompt compressor.</h1>
    <div class="stats">
      <div class="stat"><div class="big">96.5%</div><div class="lab">answers kept &nbsp;(vs 38.7% neural)</div></div>
      <div class="stat"><div class="big">1213&times;</div><div class="lab">faster per request</div></div>
      <div class="stat"><div class="big">0 MB</div><div class="lab">model weights &nbsp;(vs 312 MB)</div></div>
    </div>
    <div class="foot"><span class="arrow">&rarr;</span> github.com/aifiesta/compress-at-the-gate &nbsp;&middot;&nbsp; query-aware classical prompt compression</div>
  </div>
</div>
</body></html>"""

out = HTML.replace("__LOGO__", logo_svg)
with open(os.path.join(HERE, "cover.html"), "w") as f:
    f.write(out)
print("wrote cover.html")
