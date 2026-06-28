"""
Plain black-and-white X article, written as two people chatting, now with three
REAL animated replays embedded:
  1. compression demo on an actual benchmark item (real per-sentence scores)
  2. terminal replay of the real run, fast-forwarded to ~15s
  3. Chart.js graphs built from the real result numbers
No em dashes. Self-contained (Chart.js + data inlined).
"""
import os, json

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
N = json.load(open(os.path.join(ROOT, "numbers.json")))
DEMO = open(os.path.join(HERE, "demo_data.json")).read()
CHARTJS = open(os.path.join(HERE, "chart.umd.min.js")).read()


def msg(who, *paras):
    body = "".join(f"<p>{p}</p>" for p in paras)
    cls = "a" if who == "Sam" else "b"
    return f'<div class="m {cls}"><span class="who">{who}</span>{body}</div>'


# ----- the three widget blocks (filled by JS) -----
W_DEMO = """
<div class="m a"><div class="widget" id="demo">
  <div class="wlabel">the data, and what we did &middot; one real question from the run</div>
  <div class="dq"><b>Question:</b> <span id="dq"></span> &nbsp; <b>Answer:</b> <span class="ans" id="da"></span></div>
  <div class="dtabs">
    <button class="dt on" data-m="BM25+Q">BM25 (reads the question)</button>
    <button class="dt" data-m="Neural">Selective Context (ignores it)</button>
    <button class="dplay" id="dplay">&#9654; play</button>
  </div>
  <div id="dlist" class="dlist"></div>
  <div id="dverdict" class="verdict"></div>
</div></div>"""

W_RET = """
<div class="m a"><div class="widget">
  <div class="wlabel">the result &middot; did the answer survive, all methods, 200 questions</div>
  <canvas id="cRet" height="230"></canvas>
</div></div>"""

W_LAT = """
<div class="m a"><div class="widget">
  <div class="wlabel">the result &middot; time to compress one 1024-token prompt</div>
  <canvas id="cLat" height="210"></canvas>
</div></div>"""

W_TERM = """
<div class="m a"><div class="widget">
  <div class="wlabel">what we did &middot; the actual run, sped up (real took about 6 min)</div>
  <div class="term" id="term">
    <div class="tbar"><span class="d"></span><span class="d"></span><span class="d"></span>
      <span class="tt">run_all.py &mdash; bash</span><span class="clock" id="tclock">0:00</span></div>
    <pre id="tbody"></pre>
  </div>
  <button class="dplay" id="tplay">&#9654; replay</button>
</div></div>"""

turns = [
    msg("Riya", "what were you heads down on all week"),
    msg("Sam", "prompt compression, but for the gateway. you know how Mesh sits in front of like 300 models and every request goes through it first."),
    msg("Sam", "if you can cut the dead weight out of a prompt before it hits the model, you save tokens and you dodge the lost in the middle thing where models ignore stuff buried in the center."),
    msg("Riya", "isn't that solved though. just use LLMLingua or one of those"),
    msg("Sam", "that was my assumption too. but all of those run a small language model over your prompt to score which tokens matter. so the compressor itself needs a forward pass."),
    msg("Sam", "fine inside one app. at the gateway it is rough, because you pay that cost on every single request, for every tenant. it adds straight to the time before the first token comes back, and it wants its own GPU."),
    msg("Sam", "in the LongLLMLingua paper one of these methods had a net speedup of 0.6x. so after you count the compressor, the whole request got slower. you added a model to save time and it cost you time."),
    msg("Riya", "ok that is funny. so what did you do instead"),
    msg("Sam", "went backwards on purpose. classical stuff. BM25, TF-IDF, TextRank, centroid plus MMR. old information retrieval methods. no model, they just rank sentences."),
    msg("Sam", "i built a benchmark from SQuAD where the answer is hidden in a pile of distractor paragraphs, then checked one thing: after compressing, did the answer survive."),
    msg("Sam", "easier to just show you. here is one real question from the run. pick a method and hit play."),
    W_DEMO,
    msg("Riya", "oh that is clear. BM25 reads the question, finds the sentence with the field goal, keeps it. the neural one just keeps whatever looks unusual and throws the answer out."),
    msg("Sam", "right. now do that across 200 questions and it is the same story every time."),
    W_RET,
    msg("Riya", "the old methods are at the top and the neural one is near the bottom, basically tied with random"),
    msg("Sam", "because the split is not classical vs neural. it is whether the method reads the question. once you use it, even plain TF-IDF jumps from " + N["retTFIDF"] + " to " + N["retTFIDFQ"] + " percent."),
    msg("Riya", "alright but accuracy is half of it. what about speed and cost"),
    msg("Sam", "that part is not close at all."),
    W_LAT,
    msg("Sam", "classical lands around " + N["latBMQ"] + " ms. the neural one takes " + N["latNeural"] + " ms on the same machine, about " + N["latSpeedup"] + " times slower, and it carries " + N["neuralParamMB"] + " MB of weights. the classical ones carry none."),
    msg("Riya", "ok real question. did you actually run all this or is this a nice theory with made up numbers"),
    msg("Sam", "fair, and i would ask the same. here is the actual run, sped up. 200 questions, and the neural part is a real distilgpt2 doing real forward passes."),
    W_TERM,
    msg("Sam", "and to be sure it was not a lucky draw, i ran it again on a fresh sample with a different seed. BM25 went 96.5 to 95, TF-IDF+Q 98.5 to 96.7, the neural one 38.7 to 40. same story, different data."),
    msg("Riya", "good. now tell me where it is weak, because that looks too clean"),
    msg("Sam", "three honest things."),
    msg("Sam", "one, my neural baseline is a small model on CPU. fair same machine test, but on a GPU its latency would drop a lot, so the " + N["latSpeedup"] + "x is specific to this setup. the accuracy gap is the real point, and that is about it ignoring the question, not its size."),
    msg("Sam", "two, i measured whether the answer text survives, not whether a model then answers right. strong proxy, not the final word. i wrote the end to end version that runs answers through Mesh, but i have not run it yet, no key handy."),
    msg("Sam", "three, these only keep or drop whole sentences, they cannot rewrite. and it is English SQuAD with short answers. multi hop or other languages might move the numbers."),
    msg("Riya", "that is the part i trust the most. so what is the one line"),
    msg("Sam", "people study prompt compression like a modeling problem. at the gateway it is a systems problem, and the cheapest tool wins. a 1970s ranking function beat a neural net here, mostly because it bothered to read the question."),
    msg("Riya", "put that in the post"),
    msg("Sam", "already did."),
]

body = "\n".join(turns)

CSS = """
body{background:#fff;color:#000;margin:0;
  font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:18px;line-height:1.55;}
.wrap{max-width:640px;margin:0 auto;padding:40px 22px 80px;}
h1{font-size:30px;line-height:1.2;margin:0 0 6px;}
.sub{color:#444;font-size:17px;margin:0 0 6px;}
.meta{color:#777;font-size:14px;margin:0 0 26px;border-bottom:1px solid #000;padding-bottom:18px;}
.m{margin:0 0 20px;}
.m .who{display:block;font-weight:700;font-size:13px;margin-bottom:2px;}
.m.b .who:after{content:" asks";font-weight:400;color:#888;}
.m p{margin:0 0 8px;}
.foot{border-top:1px solid #000;margin-top:30px;padding-top:18px;font-size:15px;color:#333;}
.foot a{color:#000;}
/* widgets */
.widget{border:1px solid #000;padding:14px;margin:4px 0;background:#fff;}
.wlabel{font-size:12px;letter-spacing:.03em;text-transform:uppercase;color:#666;margin-bottom:10px;
  font-family:ui-monospace,Menlo,monospace;}
.dq{font-size:15px;margin-bottom:10px;}
.ans{border:1px solid #000;padding:0 6px;font-weight:700;}
.dtabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;}
.dt,.dplay{font:inherit;font-size:13px;border:1px solid #000;background:#fff;padding:5px 10px;cursor:pointer;}
.dt.on{background:#000;color:#fff;}
.dplay{margin-left:auto;}
.dlist{font-size:13.5px;font-family:ui-monospace,Menlo,monospace;}
.srow{display:flex;align-items:center;gap:8px;padding:3px 0;border-bottom:1px solid #eee;
  transition:opacity .35s;}
.srow .bar{flex:0 0 90px;height:9px;border:1px solid #000;position:relative;}
.srow .bar i{position:absolute;left:0;top:0;bottom:0;width:0;background:#000;transition:width .5s;}
.srow .tx{flex:1;color:#000;}
.srow.drop{opacity:.32;}
.srow.drop .tx{text-decoration:line-through;}
.srow.gold .tx{font-weight:700;}
.srow .tag{font-size:11px;border:1px solid #000;padding:0 4px;white-space:nowrap;}
.srow .mark{flex:0 0 18px;text-align:center;font-weight:700;}
.verdict{margin-top:12px;font-weight:700;font-size:15px;min-height:22px;}
.verdict.ok:before{content:"\\2713  ";}
.verdict.no:before{content:"\\2717  ";}
/* terminal */
.term{background:#000;color:#e6e6e6;font-family:ui-monospace,Menlo,monospace;font-size:13px;
  border:1px solid #000;}
.tbar{background:#1c1c1c;padding:6px 10px;display:flex;align-items:center;gap:6px;}
.tbar .d{width:10px;height:10px;border-radius:50%;background:#555;display:inline-block;}
.tbar .tt{color:#999;font-size:12px;margin-left:8px;}
.tbar .clock{margin-left:auto;color:#9fe89f;font-size:12px;}
.term pre{margin:0;padding:12px;white-space:pre-wrap;word-break:break-word;min-height:230px;max-height:330px;overflow:auto;}
.term .cmd{color:#9fe89f;}
.term .ok{color:#fff;}
.term .cur{background:#e6e6e6;color:#000;}
canvas{max-width:100%;}
"""

JS = """
const DEMO = __DEMO__;

/* ---------- 1. compression demo ---------- */
(function(){
  const item = DEMO.item, methods = DEMO.methods;
  const qel = document.getElementById('dq'), ael = document.getElementById('da');
  qel.textContent = item.question; ael.textContent = item.answer;
  const list = document.getElementById('dlist');
  let cur = 'BM25+Q';
  function short(t){ return t.length>120 ? t.slice(0,118)+'\\u2026' : t; }
  function render(){
    list.innerHTML='';
    item.sentences.forEach(s=>{
      const row=document.createElement('div'); row.className='srow'; row.id='s'+s.i;
      const gold = s.i===item.gold_sentence_idx;
      if(gold) row.classList.add('gold');
      row.innerHTML = '<span class="mark" id="m'+s.i+'"></span>'+
        '<span class="bar"><i id="b'+s.i+'"></i></span>'+
        '<span class="tx">'+short(s.text)+(gold?' <span class="tag">answer here: '+item.answer+'</span>':'')+'</span>';
      list.appendChild(row);
    });
    document.getElementById('dverdict').textContent='';
    document.getElementById('dverdict').className='verdict';
  }
  function play(){
    render();
    const M=methods[cur], sc=M.scores, kept=new Set(M.kept);
    let i=0;
    (function step(){
      if(i<sc.length){
        document.getElementById('b'+i).style.width=Math.round(sc[i]*100)+'%';
        i++; setTimeout(step, 90);
      } else { setTimeout(decide, 350); }
    })();
    function decide(){
      item.sentences.forEach(s=>{
        const row=document.getElementById('s'+s.i), mk=document.getElementById('m'+s.i);
        if(kept.has(s.i)){ mk.textContent='\\u2713'; }
        else { row.classList.add('drop'); mk.textContent=''; }
      });
      const v=document.getElementById('dverdict');
      if(M.answer_kept){ v.className='verdict ok'; v.textContent='answer kept. compressed to '+M.kept_tokens+' tokens.'; }
      else { v.className='verdict no'; v.textContent='answer dropped. the model never sees \\u201c'+item.answer+'\\u201d.'; }
    }
  }
  document.querySelectorAll('.dt').forEach(b=>b.onclick=()=>{
    document.querySelectorAll('.dt').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); cur=b.dataset.m; render();
  });
  document.getElementById('dplay').onclick=play;
  render();
  let done=false;
  new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting&&!done){done=true;setTimeout(play,400);}}),
    {threshold:.4}).observe(document.getElementById('demo'));
})();

/* ---------- 2. terminal replay ---------- */
(function(){
  const lines = DEMO.run_log.slice();
  const pre=document.getElementById('tbody'), clock=document.getElementById('tclock');
  const TOTAL=15000, REAL=351; // 15s render maps to ~5:51 real
  let timer=null;
  function fmt(s){ const m=Math.floor(s/60); return m+':'+String(Math.floor(s%60)).padStart(2,'0'); }
  function run(){
    if(timer) clearInterval(timer);
    pre.innerHTML='<span class="cmd">$ python run_all.py</span>\\n';
    let i=0; const per=TOTAL/(lines.length+2); const t0=Date.now();
    timer=setInterval(()=>{
      const el=Math.min(REAL, (Date.now()-t0)/TOTAL*REAL); clock.textContent=fmt(el);
      if(i>=lines.length){ clearInterval(timer); clock.textContent=fmt(REAL);
        pre.innerHTML+='<span class="cur"> </span>'; pre.scrollTop=pre.scrollHeight; return; }
      let ln=lines[i].replace(/&/g,'&amp;').replace(/</g,'&lt;');
      const cls = /=====|DONE|wrote|retention=|ms\\/call|@1024/.test(ln)?'ok':'';
      pre.innerHTML += '<span class="'+cls+'">'+ln+'</span>\\n';
      pre.scrollTop=pre.scrollHeight; i++;
    }, per);
  }
  document.getElementById('tplay').onclick=run;
  let done=false;
  new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting&&!done){done=true;run();}}),
    {threshold:.3}).observe(document.getElementById('term'));
})();

/* ---------- 3. charts ---------- */
function bwBar(id, rows, opts){
  const labels=rows.map(r=>r.m.replace('SelectiveContext(neural)','Selective Context'));
  const data=rows.map(r=>r.v);
  const colors=rows.map(r=>r.neural?'#bbb':'#000');
  const ctx=document.getElementById(id);
  return new Chart(ctx,{type:'bar',data:{labels,datasets:[{data,backgroundColor:colors,
    borderColor:'#000',borderWidth:1,barPercentage:.78}]},
    options:Object.assign({indexAxis:'y',animation:{duration:1400},responsive:true,
      plugins:{legend:{display:false},tooltip:{enabled:true}},
      scales:{x:opts.x, y:{ticks:{color:'#000',font:{size:11}},grid:{display:false}}}},opts.extra||{})});
}
let chartsDone=false;
function makeCharts(){
  if(chartsDone) return; chartsDone=true;
  bwBar('cRet', DEMO.charts.retention, {x:{min:0,max:100,title:{display:true,text:'answer retention (%)',color:'#000'},ticks:{color:'#000'},grid:{color:'#eee'}}});
  bwBar('cLat', DEMO.charts.latency, {x:{type:'logarithmic',title:{display:true,text:'ms per request (log scale)',color:'#000'},ticks:{color:'#000'},grid:{color:'#eee'}}});
}
new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){makeCharts();}}),
  {threshold:.3}).observe(document.getElementById('cRet'));
"""

HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compress at the gate</title>
<style>__CSS__</style>
<script>__CHARTJS__</script>
</head><body><div class="wrap">
<h1>We beat a neural prompt compressor with 1970s NLP</h1>
<p class="sub">A chat about a thing we tried at the gateway, with the actual run you can watch.</p>
<p class="meta">Mesh API routing team. June 2026. Real measurements, on a laptop.</p>
__BODY__
<div class="foot">
The paper, the benchmark, and all the code are open. It reproduces offline.
The only network call is a one time download of SQuAD and a small model.<br><br>
Paper (PDF): [link] &nbsp; Code and benchmark: [link] &nbsp; meshapi.ai
</div></div>
<script>__JS__</script>
</body></html>"""

out = (HTML.replace("__CSS__", CSS).replace("__CHARTJS__", CHARTJS)
           .replace("__BODY__", body).replace("__JS__", JS.replace("__DEMO__", DEMO)))
open(os.path.join(HERE, "x_article.html"), "w").write(out)
print("wrote x_article.html with 3 live replays (", len(out)//1024, "KB )")
