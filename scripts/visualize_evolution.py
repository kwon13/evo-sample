"""
Evolution 스냅샷을 HTML dashboard로 시각화.

저장된 JSON 데이터를 읽어서 dashboard 생성.
학습 중 or 학습 후 언제든 실행 가능.

Usage:
  # 최신 스냅샷만
  python scripts/visualize_evolution.py

  # 전체 히스토리 (step별 슬라이더)
  python scripts/visualize_evolution.py --all

  # 특정 경로
  python scripts/visualize_evolution.py --log_dir ./rq_output/verl_ckpt/evolution_logs
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_snapshots(log_dir: Path, all_steps: bool = False):
    """JSON 스냅샷 로드."""
    if all_steps:
        files = sorted(log_dir.glob("evo_step_*.json"), key=lambda f: int(f.stem.split("_")[-1]))
        return [json.loads(f.read_text()) for f in files]
    else:
        latest = log_dir / "latest.json"
        if latest.exists():
            return [json.loads(latest.read_text())]
        return []


def build_html(snapshots: list[dict], output_path: Path):
    """Feasibility test와 동일한 HTML dashboard 생성."""
    if not snapshots:
        print("No snapshots found.")
        return

    latest = snapshots[-1]
    n_h = max(c["h_bin"] for c in latest["grid"]) + 1
    n_d = max(c["div_bin"] for c in latest["grid"]) + 1
    seed_labels = latest.get("seed_labels", {})

    # Step별 grid 데이터
    all_grids = []
    history = []
    for snap in snapshots:
        cells = []
        for c in snap["grid"]:
            cells.append({
                "h": c["h_bin"], "d": c["div_bin"],
                "rq": round(c.get("rq_score", 0), 4) if c["has_champion"] else 0,
                "p": round(c.get("p_hat", 0), 3) if c["has_champion"] else 0,
                "H": round(c.get("h_score", 0), 3) if c["has_champion"] else 0,
                "gen": c.get("generation", 0) if c["has_champion"] else 0,
                "has": 1 if c["has_champion"] else 0,
                "problem": c.get("problem", "")[:200],
                "answer": c.get("answer", "")[:50],
            })
        all_grids.append(cells)
        history.append({
            "step": snap["global_step"],
            **snap["metrics"],
        })

    # 챔피언 데이터 (최신)
    champs = []
    for c in sorted(latest["grid"], key=lambda x: -(x.get("rq_score", 0))):
        if c["has_champion"]:
            champs.append(c)

    import html as html_mod
    h_labels = json.dumps([f"H{i}" for i in range(n_h)])
    d_labels = json.dumps({str(k): v for k, v in seed_labels.items()})
    snapshots_json = json.dumps(all_grids)
    history_json = json.dumps(history)
    champs_json = json.dumps(champs, ensure_ascii=False, default=str)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>RQ-Evolve Evolution Dashboard</title>
<style>
body {{ font-family: system-ui; background: #1a1a2e; color: #eee; padding: 20px; }}
h1 {{ color: #e94560; }} h2 {{ color: #0f3460; background: #e94560; padding: 8px 16px; border-radius: 4px; margin: 20px 0 10px; }}
.section {{ background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
table.grid {{ border-collapse: collapse; }} table.grid th {{ background: #0f3460; padding: 6px; font-size: 11px; }}
table.grid td {{ width: 70px; height: 35px; text-align: center; font-size: 10px; border: 1px solid #1a1a2e; cursor: pointer; }}
table.grid td.empty {{ background: #0d1b2a; color: #333; }}
table.grid td:hover {{ outline: 2px solid #e94560; }}
table.grid td.new {{ box-shadow: inset 0 0 0 2px #ffd700; }}
table.grid td.improved {{ box-shadow: inset 0 0 0 2px #00ff88; }}
.row-label {{ background: #0f3460; padding: 4px 8px; font-size: 10px; text-align: right; white-space: nowrap; }}
.tooltip {{ display:none; position:fixed; background:#16213e; border:1px solid #e94560; border-radius:6px; padding:12px; max-width:500px; z-index:100; font-size:12px; line-height:1.5; }}
.slider-container {{ display:flex; align-items:center; gap:12px; margin-bottom:12px; }}
.slider-container input {{ flex:1; accent-color:#e94560; }}
.slider-label {{ font-size:18px; font-weight:bold; color:#e94560; min-width:120px; }}
.line {{ fill:none; stroke:#e94560; stroke-width:2; }} .line-max {{ fill:none; stroke:#ffd700; stroke-width:2; }} .line-hard {{ fill:none; stroke:#00ff88; stroke-width:2; }}
.champ {{ background:#0d1b2a; border-radius:6px; padding:12px; margin:8px 0; }}
.champ .rq {{ color:#e94560; font-size:18px; font-weight:bold; }}
.champ .problem {{ color:#aaa; font-size:12px; margin-top:6px; }}
.champ code {{ background:#1a1a2e; padding:8px; display:block; font-size:11px; max-height:200px; overflow-y:auto; margin-top:6px; white-space:pre-wrap; }}
.btn {{ background:#0f3460; color:#eee; border:1px solid #e94560; padding:4px 12px; border-radius:4px; cursor:pointer; font-size:12px; }}
.btn:hover {{ background:#e94560; }}
</style></head><body>
<h1>RQ-Evolve Evolution Dashboard</h1>
<p style="color:#aaa">Snapshots: {len(snapshots)} steps</p>
<h2>MAP-Elites Grid</h2>
<div class="section">
<div class="slider-container"><span style="color:#aaa">Step:</span>
<input type="range" id="stepSlider" min="0" max="{len(snapshots)-1}" value="{len(snapshots)-1}">
<span class="slider-label" id="stepLabel">Latest</span>
<button class="btn" id="btnPlay" onclick="playAnim()">Play</button></div>
<table class="grid" id="gridTable"></table></div>
<div class="tooltip" id="tooltip"></div>
<h2>Evolution History</h2>
<div class="section"><div style="display:flex;gap:20px;flex-wrap:wrap;">
<div style="flex:1;min-width:300px;"><p style="color:#aaa">Mean/Max R_Q</p><svg id="c1" viewBox="0 0 600 180"></svg></div>
<div style="flex:1;min-width:300px;"><p style="color:#aaa">Coverage / H2+ Champions</p><svg id="c2" viewBox="0 0 600 180"></svg></div>
</div></div>
<h2>Champions</h2>
<div class="section" id="champList"></div>
<script>
const H_LABELS={h_labels}, D_MAP={d_labels}, SNAPS={snapshots_json}, HIST={history_json}, CHAMPS={champs_json};
const N_H={n_h}, N_D={n_d};
function rqColor(r){{ if(r<=0)return'#0d1b2a'; const t=Math.min(r/0.8,1); return `rgb(${{35+t*198|0}},${{27+t*42|0}},${{46+t*50|0}})`; }}
const slider=document.getElementById('stepSlider'), label=document.getElementById('stepLabel');
function renderGrid(i){{
  const g=SNAPS[i], p=i>0?SNAPS[i-1]:null; label.textContent=i===0?'Init':'Step '+HIST[i]?.step;
  let h='<tr><th></th>'; H_LABELS.forEach(l=>h+=`<th>${{l}}</th>`); h+='</tr>';
  for(let d=0;d<N_D;d++){{ h+=`<tr><td class="row-label">${{D_MAP[d]||'D'+d}}</td>`;
    for(let hh=0;hh<N_H;hh++){{ const c=g.find(x=>x.h===hh&&x.d===d), pc=p?p.find(x=>x.h===hh&&x.d===d):null;
      if(c&&c.has){{ let cls='',delta=''; if(pc){{ if(!pc.has){{cls='new';delta=' NEW';}}else if(c.rq>pc.rq+.001){{cls='improved';delta=` +${{(c.rq-pc.rq).toFixed(4)}}`;}} }}
        h+=`<td class="${{cls}}" style="background:${{rqColor(c.rq)}}" data-info="${{(D_MAP[c.d]||'D'+c.d)+' H'+c.h+delta+'\\nRQ='+c.rq+' p='+c.p+' H='+c.H+'\\ngen='+c.gen+'\\nQ: '+c.problem+'\\nA: '+c.answer}}">${{c.rq.toFixed(3)}}</td>`;
      }}else h+='<td class="empty">·</td>';
    }} h+='</tr>';
  }} document.getElementById('gridTable').innerHTML=h;
}}
slider.addEventListener('input',()=>renderGrid(+slider.value)); renderGrid(+slider.value);
function playAnim(){{ slider.value=0;renderGrid(0);let i=0;const iv=setInterval(()=>{{i++;if(i>=SNAPS.length){{clearInterval(iv);return;}}slider.value=i;renderGrid(i);}},800);}}
const tt=document.getElementById('tooltip');
document.getElementById('gridTable').addEventListener('mouseover',e=>{{const i=e.target.getAttribute('data-info');if(i){{tt.innerHTML=i.replace(/\\n/g,'<br>');tt.style.display='block';}}}});
document.getElementById('gridTable').addEventListener('mousemove',e=>{{tt.style.left=(e.clientX+14)+'px';tt.style.top=(e.clientY+14)+'px';}});
document.getElementById('gridTable').addEventListener('mouseout',()=>tt.style.display='none');
function drawChart(id,data,lines){{ const svg=document.getElementById(id); if(!data.length)return; const W=600,H=160,P=40,n=data.length;
  const maxY=Math.max(...lines.flatMap(l=>data.map(d=>d[l.key]||0)),0.01)*1.15;
  let s=`<line x1="${{P}}" y1="${{H}}" x2="${{W}}" y2="${{H}}" stroke="#333"/>`;
  lines.forEach((l,li)=>{{ const pts=data.map((d,i)=>`${{P+i*(W-P)/Math.max(n-1,1)}},${{H-(d[l.key]||0)/maxY*H}}`).join(' ');
    s+=`<polyline points="${{pts}}" class="${{l.cls}}"/><text x="${{W-80}}" y="${{15+li*15}}" fill="${{l.color}}" font-size="10">${{l.label}}</text>`;
  }}); svg.innerHTML=s; }}
drawChart('c1',HIST,[{{key:'grid_mean_rq',cls:'line',color:'#e94560',label:'Mean RQ'}},{{key:'grid_max_rq',cls:'line-max',color:'#ffd700',label:'Max RQ'}}]);
drawChart('c2',HIST,[{{key:'grid_coverage',cls:'line',color:'#e94560',label:'Coverage'}},{{key:'hard_champions',cls:'line-hard',color:'#00ff88',label:'H2+'}}]);
const cd=document.getElementById('champList');
CHAMPS.forEach(c=>{{ cd.innerHTML+=`<div class="champ"><span class="rq">RQ=${{(c.rq_score||0).toFixed(4)}}</span> p=${{(c.p_hat||0).toFixed(2)}} H=${{(c.h_score||0).toFixed(2)}} gen=${{c.generation}}<div class="problem">Q: ${{c.problem||''}}<br>A: ${{c.answer||''}}</div><code>${{c.source_code||''}}</code></div>`; }});
</script></body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize evolution snapshots")
    parser.add_argument("--log_dir", default="./rq_output/verl_ckpt/evolution_logs")
    parser.add_argument("--output", default="./rq_output/evolution_dashboard.html")
    parser.add_argument("--all", action="store_true", help="모든 step 포함 (슬라이더)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        sys.exit(1)

    snapshots = load_snapshots(log_dir, all_steps=args.all)
    print(f"Loaded {len(snapshots)} snapshot(s)")

    build_html(snapshots, Path(args.output))


if __name__ == "__main__":
    main()
