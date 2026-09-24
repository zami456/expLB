"""Build the shareable audit-record page from the two markdown deliverables.

The markdown is base64-embedded and rendered client-side by marked (pinned, from
cdnjs), so the text on the page is byte-identical to the files in audit/ -- no
hand transcription, which in an audit deliverable would be the wrong risk to take.

  python audit/scripts/p10_build_record_page.py <output.html>
"""
import base64
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS = [("brief", "audit/BRIEF_AS_EXECUTED.md"),
        ("report", "audit/REPORT_PLAIN.md")]

PAGE = r"""<title>expLB Audit Record</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --paper:#f5f7f8; --card:#ffffff; --ink:#14181c; --ink2:#4d5a64; --ink3:#78858f;
  --rule:#dde3e7; --rule2:#eaeef1; --accent:#0e6b64; --accent-soft:#e3efee;
  --on-accent:#ffffff;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --serif:"Newsreader",Georgia,"Times New Roman",serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --paper:#101417; --card:#161b1f; --ink:#e7edf1; --ink2:#a4b1ba; --ink3:#7d8b94;
  --rule:#2a3339; --rule2:#1f272c; --accent:#5fcabf; --accent-soft:#16302e;
  --on-accent:#0b1013;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --paper:#101417; --card:#161b1f; --ink:#e7edf1; --ink2:#a4b1ba; --ink3:#7d8b94;
  --rule:#2a3339; --rule2:#1f272c; --accent:#5fcabf; --accent-soft:#16302e;
  --on-accent:#0b1013;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:15px;
  line-height:1.65;margin:0}
.wrap{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:0 72px}
header.top{position:sticky;top:env(safe-area-inset-top,0px);z-index:20;background:var(--paper);
  border-bottom:1px solid var(--rule);padding-block:14px;margin-bottom:28px}
.topin{max-width:1180px;margin:0 auto;padding-inline:20px;display:flex;flex-wrap:wrap;gap:14px;
  align-items:baseline;justify-content:space-between}
.brand{font-family:var(--serif);font-size:20px;font-weight:600;letter-spacing:-.01em}
.brand span{color:var(--accent)}
.meta{font-family:var(--mono);font-size:11.5px;color:var(--ink3);letter-spacing:.02em}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tab{font-family:var(--mono);font-size:12px;letter-spacing:.04em;text-transform:uppercase;
  padding:7px 13px;border:1px solid var(--rule);border-radius:2px;background:transparent;
  color:var(--ink2);cursor:pointer;transition:border-color .15s,color .15s}
.tab:hover{border-color:var(--accent);color:var(--ink)}
.tab[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
.tab:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.layout{display:grid;grid-template-columns:230px minmax(0,1fr);gap:40px;align-items:start}
@media(max-width:900px){.layout{grid-template-columns:1fr;gap:20px}}
nav.toc{position:sticky;top:calc(env(safe-area-inset-top,0px) + 78px);
  max-height:calc(100vh - 130px);overflow-y:auto;font-size:13px;line-height:1.45;
  border-left:2px solid var(--rule2);padding-left:14px}
@media(max-width:900px){nav.toc{position:static;max-height:none;border-left:0;
  border-top:1px solid var(--rule);padding-left:0;padding-top:14px}}
nav.toc h4{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink3);margin:0 0 10px;font-weight:500}
nav.toc a{display:block;color:var(--ink2);text-decoration:none;padding:3px 0}
nav.toc a:hover{color:var(--accent)}
nav.toc a.lvl3{padding-left:12px;font-size:12.4px;color:var(--ink3)}
article{min-width:0;background:var(--card);border:1px solid var(--rule);border-radius:3px;
  padding:34px 38px 44px}
@media(max-width:700px){article{padding:22px 18px 30px}}
article h1{font-family:var(--serif);font-size:clamp(26px,4.4vw,38px);line-height:1.15;
  font-weight:600;letter-spacing:-.018em;margin:0 0 6px;text-wrap:balance}
article h2{font-family:var(--serif);font-size:clamp(20px,2.8vw,25px);font-weight:600;
  letter-spacing:-.012em;margin:44px 0 12px;padding-top:18px;border-top:1px solid var(--rule2);
  text-wrap:balance;scroll-margin-top:100px}
article h3{font-family:var(--sans);font-size:15.5px;font-weight:600;margin:28px 0 8px;
  scroll-margin-top:100px}
article h4{font-family:var(--mono);font-size:12px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--ink3);font-weight:500;margin:22px 0 6px}
article p{margin:0 0 13px;max-width:70ch}
article ul,article ol{margin:0 0 14px;padding-left:22px;max-width:70ch}
article li{margin:0 0 5px}
article li::marker{color:var(--accent)}
article a{color:var(--accent);text-underline-offset:2px}
article strong{font-weight:600}
article em{font-style:italic;color:var(--ink2)}
article hr{border:0;border-top:1px solid var(--rule2);margin:32px 0}
article code{font-family:var(--mono);font-size:.87em;background:var(--rule2);padding:1.5px 5px;
  border-radius:2px;word-break:break-word}
article pre{background:var(--rule2);border:1px solid var(--rule);border-radius:3px;
  padding:13px 15px;overflow-x:auto;margin:0 0 16px}
article pre code{background:none;padding:0;font-size:12.5px;line-height:1.55}
article blockquote{margin:0 0 16px;padding:12px 18px;border-left:3px solid var(--accent);
  background:var(--accent-soft);border-radius:0 3px 3px 0}
article blockquote p{margin:0;max-width:none}
.tablewrap{overflow-x:auto;margin:0 0 18px;border:1px solid var(--rule);border-radius:3px}
article table{border-collapse:collapse;width:100%;font-size:13.4px;font-variant-numeric:tabular-nums}
article th{font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--ink3);font-weight:500;text-align:left;padding:9px 13px;
  border-bottom:1px solid var(--rule);background:var(--rule2);white-space:nowrap}
article td{padding:8px 13px;border-bottom:1px solid var(--rule2);vertical-align:top}
article tr:last-child td{border-bottom:0}
footer.foot{margin-top:34px;font-family:var(--mono);font-size:11.5px;color:var(--ink3);
  text-align:center;letter-spacing:.02em}
.loading{font-family:var(--mono);font-size:13px;color:var(--ink3);padding:40px 0}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<header class="top">
  <div class="topin">
    <div>
      <div class="brand">exp<span>LB</span> &mdash; audit record</div>
      <div class="meta">Persona-coordinate replication on Llama-3.3-70B</div>
    </div>
    <div class="tabs" role="tablist">
      <button class="tab" role="tab" id="t-brief" aria-selected="true">Brief, as executed</button>
      <button class="tab" role="tab" id="t-report" aria-selected="false">Report</button>
    </div>
  </div>
</header>

<div class="wrap">
  <div class="layout">
    <nav class="toc" aria-label="Contents"><h4>Contents</h4><div id="toc"></div></nav>
    <article id="doc"><p class="loading">Rendering&hellip;</p></article>
  </div>
  <footer class="foot">Every figure traceable to a file opened or a script run during the audit.</footer>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
<script>
var SRC = {brief:"__BRIEF__", report:"__REPORT__"};
function dec(b){ return new TextDecoder().decode(Uint8Array.from(atob(b), function(c){return c.charCodeAt(0);})); }
var docEl = document.getElementById('doc'), tocEl = document.getElementById('toc');
function slug(s, i){
  return 'h' + i + '-' + s.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'').slice(0,40);
}
function render(key){
  var md = dec(SRC[key]);
  if (!window.marked){
    var pre = document.createElement('pre'); pre.textContent = md;
    docEl.replaceChildren(pre); tocEl.replaceChildren(); return;
  }
  docEl.innerHTML = marked.parse(md, {mangle:false, headerIds:false});
  docEl.querySelectorAll('table').forEach(function(t){
    var w = document.createElement('div'); w.className = 'tablewrap';
    t.parentNode.insertBefore(w, t); w.appendChild(t);
  });
  var items = [];
  docEl.querySelectorAll('h2, h3').forEach(function(h, i){
    var id = slug(h.textContent, i); h.id = id;
    items.push({id:id, text:h.textContent, lvl:h.tagName === 'H3'});
  });
  tocEl.replaceChildren.apply(tocEl, items.map(function(it){
    var a = document.createElement('a');
    a.href = '#' + it.id; a.textContent = it.text;
    if (it.lvl) a.className = 'lvl3';
    return a;
  }));
  document.querySelectorAll('.tab').forEach(function(b){
    b.setAttribute('aria-selected', String(b.id === 't-' + key));
  });
  window.scrollTo({top:0});
}
document.getElementById('t-brief').addEventListener('click', function(){ render('brief'); });
document.getElementById('t-report').addEventListener('click', function(){ render('report'); });
render('brief');
</script>
"""


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "audit_record.html")
    page = PAGE
    for key, rel in DOCS:
        raw = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        b64 = base64.b64encode(raw.encode("utf-8")).decode("ascii")
        print(f"  {rel}: {len(raw):,} chars")
        page = page.replace("__" + key.upper() + "__", b64)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"wrote {out}  ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
