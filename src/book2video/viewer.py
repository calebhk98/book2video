from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from book2video.assembly import FFmpegAssembler
from book2video.io_utils import read_json
from book2video.review import load_selected_and_takes, save_selection_override


class TakeSelectionRequest(BaseModel):
    clip_id: str
    take_number: int = Field(ge=1)


class ViewerState:
    def __init__(self, run_dir: str | Path, auto_assemble_on_selection: bool = False) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.auto_assemble_on_selection = auto_assemble_on_selection

    def _media_url(self, path: str) -> str:
        source = Path(path).resolve()
        try:
            rel = source.relative_to(self.run_dir)
        except ValueError as exc:
            raise RuntimeError(
                f"Viewer expects generated media under the run directory; {source} is outside {self.run_dir}."
            ) from exc
        return "/media/" + rel.as_posix()

    def snapshot(self) -> dict[str, Any]:
        selected, takes = load_selected_and_takes(self.run_dir)
        clip_specs = read_json(self.run_dir / "06_clips.json") if (self.run_dir / "06_clips.json").exists() else []
        spec_by_id = {x["clip_id"]: x for x in clip_specs}
        clips: list[dict[str, Any]] = []
        for index, clip in enumerate(selected):
            clip_takes = sorted(takes.get(clip.clip_id, []), key=lambda t: t.take_number)
            clips.append(
                {
                    "index": index,
                    "clip_id": clip.clip_id,
                    "scene_id": clip.scene_id,
                    "shot_id": clip.shot_id,
                    "selected_take": clip.selected_take,
                    "selected_url": self._media_url(clip.selected_path),
                    "target_duration_sec": spec_by_id.get(clip.clip_id, {}).get("target_duration_sec"),
                    "transition_after": clip.transition_after,
                    "judgement_reason": clip.judgement.reason if clip.judgement else "",
                    "takes": [
                        {
                            "take_number": take.take_number,
                            "url": self._media_url(take.output_path),
                            "generation_seconds": take.generation_seconds,
                            "provider_id": take.provider_id,
                        }
                        for take in clip_takes
                    ],
                }
            )
        return {
            "run_dir": str(self.run_dir),
            "clips": clips,
            "rough_cut_url": "/media/rough_cut.mp4" if (self.run_dir / "rough_cut.mp4").exists() else None,
        }

    def select(self, clip_id: str, take_number: int) -> dict[str, Any]:
        selected, takes = load_selected_and_takes(self.run_dir)
        if clip_id not in takes:
            raise KeyError(f"Unknown clip: {clip_id}")
        if not any(t.take_number == take_number for t in takes[clip_id]):
            raise ValueError(f"Clip {clip_id} has no take {take_number}")
        save_selection_override(self.run_dir, clip_id, take_number)
        if self.auto_assemble_on_selection:
            selected, _ = load_selected_and_takes(self.run_dir)
            FFmpegAssembler().assemble(selected, self.run_dir)
        return self.snapshot()

    def assemble(self) -> str | None:
        selected, _ = load_selected_and_takes(self.run_dir)
        return FFmpegAssembler().assemble(selected, self.run_dir)


def viewer_html(*, preload_adjacent: bool = True) -> str:
    # Three video elements are being tried here so the previous/current/next clips can remain loaded at once.
    # This may reduce pauses at ordinary cuts without requiring the whole chapter to live in browser memory.
    preload_js = "true" if preload_adjacent else "false"
    return rf'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Book2Video Viewer</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
body {{ margin:0; background:#111; color:#eee; height:100vh; display:grid; grid-template-columns:minmax(0,1fr) 420px; overflow:hidden; }}
main {{ display:flex; flex-direction:column; min-width:0; }}
#playerWrap {{ flex:1; position:relative; background:#000; min-height:0; overflow:hidden; }}
.videoSlot {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; display:none; background:#000; }}
.videoSlot.active {{ display:block; }}
#transport {{ padding:.75rem 1rem; background:#181818; border-top:1px solid #333; }}
#title {{ font-weight:700; margin-bottom:.45rem; }}
.controls {{ display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; }}
button {{ border:1px solid #555; background:#282828; color:#eee; padding:.45rem .7rem; border-radius:6px; cursor:pointer; }}
button:hover {{ background:#353535; }} button.selected {{ background:#eee; color:#111; }}
#sidebar {{ border-left:1px solid #333; overflow:auto; background:#161616; }}
#sidebarHeader {{ position:sticky; top:0; z-index:3; padding:1rem; background:#161616; border-bottom:1px solid #333; }}
.clip {{ padding:.75rem 1rem; border-bottom:1px solid #2d2d2d; cursor:pointer; }}
.clip.current {{ outline:2px solid #888; outline-offset:-2px; background:#222; }}
.clipHead {{ display:flex; justify-content:space-between; gap:.5rem; font-size:.88rem; }}
.takeButtons {{ display:flex; gap:.35rem; flex-wrap:wrap; margin-top:.55rem; }}
.takeButtons button {{ padding:.28rem .5rem; font-size:.8rem; }}
.small {{ color:#aaa; font-size:.78rem; margin-top:.4rem; }}
#status {{ color:#aaa; margin-left:auto; font-size:.85rem; }}
@media(max-width:950px){{ body{{grid-template-columns:1fr; grid-template-rows:62vh 38vh}} #sidebar{{border-left:0;border-top:1px solid #333}} }}
</style>
</head>
<body>
<main>
  <div id="playerWrap">
    <video id="slot0" class="videoSlot" controls preload="auto" playsinline></video>
    <video id="slot1" class="videoSlot" controls preload="auto" playsinline></video>
    <video id="slot2" class="videoSlot" controls preload="auto" playsinline></video>
  </div>
  <div id="transport">
    <div id="title">Loading…</div>
    <div class="controls">
      <button onclick="previousClip()">◀ Previous clip</button>
      <button onclick="nextClip()">Next clip ▶</button>
      <span>Current take:</span><span id="currentTakeButtons"></span>
      <button onclick="rebuild()">Rebuild rough_cut.mp4</button>
      <span id="status"></span>
    </div>
  </div>
</main>
<aside id="sidebar"><div id="sidebarHeader"><strong>All clips</strong><div class="small">Click a clip to jump. Changing the current clip's take restarts that clip from the beginning. Adjacent clips are preloaded when enabled.</div></div><div id="clips"></div></aside>
<script>
const PRELOAD_ADJACENT = {preload_js};
let state = null;
let currentIndex = 0;
let activeSlot = null;
const statusEl = document.getElementById('status');
const slots = ['slot0','slot1','slot2'].map(id => ({{
  el: document.getElementById(id),
  index: null,
  url: null,
}}));

async function refreshState(){{
  const r = await fetch('/api/state');
  state = await r.json();
  renderSidebar();
  if(state.clips.length && activeSlot === null) activateIndex(0, false, true);
}}
function esc(s){{ return (s??'').toString().replace(/[&<>"']/g, c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[c])); }}
function clipAt(i){{ return state?.clips?.[i]; }}
function currentPlayer(){{ return activeSlot?.el ?? null; }}
function desiredIndices(center){{
  if(!PRELOAD_ADJACENT) return [center];
  return [center-1, center, center+1].filter(i => i >= 0 && i < (state?.clips?.length ?? 0));
}}
function findSlot(index){{ return slots.find(s => s.index === index) ?? null; }}
function clearSlot(slot){{
  if(!slot) return;
  slot.el.pause();
  slot.el.removeAttribute('src');
  slot.el.load();
  slot.el.classList.remove('active');
  slot.index = null;
  slot.url = null;
}}
function chooseReusableSlot(index){{
  const existing = findSlot(index);
  if(existing) return existing;
  const desired = new Set(desiredIndices(currentIndex));
  const empty = slots.find(s => s.index === null);
  if(empty) return empty;
  const outside = slots.find(s => !desired.has(s.index));
  if(outside) return outside;
  return [...slots].sort((a,b) => Math.abs((b.index ?? 999)-currentIndex)-Math.abs((a.index ?? 999)-currentIndex))[0];
}}
function assignSlot(index, force=false){{
  if(index < 0 || index >= (state?.clips?.length ?? 0)) return null;
  const clip = clipAt(index);
  let slot = findSlot(index);
  if(slot && slot.url === clip.selected_url && !force) return slot;
  if(!slot) slot = chooseReusableSlot(index);
  const wasActive = slot === activeSlot;
  if(wasActive) slot.el.pause();
  slot.index = index;
  slot.url = clip.selected_url;
  slot.el.src = clip.selected_url;
  slot.el.preload = 'auto';
  slot.el.load();
  return slot;
}}
function primeAround(index){{
  const wanted = new Set(desiredIndices(index));
  for(const i of wanted) assignSlot(i, false);
  for(const slot of slots){{
    if(slot.index !== null && !wanted.has(slot.index) && slot !== activeSlot) clearSlot(slot);
  }}
}}
function whenMetadata(slot, fn){{
  if(slot.el.readyState >= 1) fn();
  else slot.el.addEventListener('loadedmetadata', fn, {{once:true}});
}}
function updateTitle(){{
  const c = clipAt(currentIndex);
  if(!c) return;
  document.getElementById('title').textContent = `${{currentIndex+1}}/${{state.clips.length}} — ${{c.clip_id}} — take ${{c.selected_take}}`;
}}
function activateIndex(index, shouldPlay, restart=true){{
  if(!state?.clips?.length) return;
  currentIndex = Math.max(0, Math.min(index, state.clips.length-1));
  primeAround(currentIndex);
  const slot = assignSlot(currentIndex, false);
  for(const other of slots) other.el.classList.toggle('active', other === slot);
  activeSlot = slot;
  whenMetadata(slot, () => {{
    if(restart) slot.el.currentTime = 0;
    if(shouldPlay) slot.el.play().catch(()=>{{}});
  }});
  updateTitle();
  renderSidebar();
  document.getElementById(`clip-${{currentIndex}}`)?.scrollIntoView({{block:'nearest'}});
  // Loading the new far-side neighbor after the visible switch gives the next cut time to warm up.
  primeAround(currentIndex);
}}
function renderSidebar(){{
  if(!state) return;
  const root = document.getElementById('clips');
  root.innerHTML = state.clips.map((c,i)=>`
    <div class="clip ${{i===currentIndex?'current':''}}" id="clip-${{i}}" onclick="jumpToClip(${{i}})">
      <div class="clipHead"><strong>${{esc(c.clip_id)}}</strong><span>${{i+1}}/${{state.clips.length}}</span></div>
      <div class="small">${{esc(c.scene_id)}} · ${{esc(c.shot_id)}}${{c.target_duration_sec?` · target ${{Number(c.target_duration_sec).toFixed(1)}}s`:''}}</div>
      <div class="takeButtons">${{c.takes.map(t=>`<button class="${{t.take_number===c.selected_take?'selected':''}}" onclick="event.stopPropagation(); selectTake('${{esc(c.clip_id)}}',${{t.take_number}},${{i}})">Take ${{t.take_number}}</button>`).join('')}}</div>
      ${{c.judgement_reason?`<div class="small">${{esc(c.judgement_reason)}}</div>`:''}}
    </div>`).join('');
  renderCurrentTakeButtons();
}}
function renderCurrentTakeButtons(){{
  const c=clipAt(currentIndex); const root=document.getElementById('currentTakeButtons');
  if(!c){{ root.innerHTML=''; return; }}
  root.innerHTML=c.takes.map(t=>`<button class="${{t.take_number===c.selected_take?'selected':''}}" onclick="selectTake('${{esc(c.clip_id)}}',${{t.take_number}},${{currentIndex}})">${{t.take_number}}</button>`).join(' ');
}}
function jumpToClip(index){{
  const p = currentPlayer();
  const shouldPlay = !!p && !p.paused && !p.ended;
  if(p) p.pause();
  activateIndex(index, shouldPlay, true);
}}
async function selectTake(clipId,takeNumber,index){{
  const oldClip = clipAt(index);
  if(oldClip?.selected_take === takeNumber) return;
  const changingCurrent = index === currentIndex;
  const p = currentPlayer();
  const shouldResume = changingCurrent && !!p && !p.paused && !p.ended;
  statusEl.textContent='Saving selection…';
  const r=await fetch('/api/select',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{clip_id:clipId,take_number:takeNumber}})}});
  if(!r.ok){{statusEl.textContent=await r.text();return;}}
  state=await r.json();

  // A changed take is a different performance, so the current clip is intentionally restarted from frame zero.
  const loaded = findSlot(index);
  if(loaded) assignSlot(index, true);
  if(changingCurrent){{
    activateIndex(currentIndex, shouldResume, true);
  }} else {{
    primeAround(currentIndex);
    renderSidebar();
  }}
  statusEl.textContent=`${{clipId}} → take ${{takeNumber}}`;
}}
function nextClip(){{
  if(currentIndex>=state.clips.length-1) return;
  const p=currentPlayer(); const shouldPlay=!!p && !p.paused && !p.ended;
  if(p) p.pause();
  activateIndex(currentIndex+1, shouldPlay, true);
}}
function previousClip(){{
  if(currentIndex<=0) return;
  const p=currentPlayer(); const shouldPlay=!!p && !p.paused && !p.ended;
  if(p) p.pause();
  activateIndex(currentIndex-1, shouldPlay, true);
}}
for(const slot of slots){{
  slot.el.addEventListener('ended',()=>{{
    if(slot !== activeSlot) return;
    if(currentIndex < state.clips.length-1) activateIndex(currentIndex+1, true, true);
  }});
}}
async function rebuild(){{statusEl.textContent='Rebuilding…';const r=await fetch('/api/assemble',{{method:'POST'}});const d=await r.json();statusEl.textContent=d.rough_cut?'rough_cut.mp4 rebuilt':'No output was assembled';}}
document.addEventListener('keydown',e=>{{if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;if(e.key===']')nextClip();if(e.key==='[')previousClip();const n=Number(e.key);if(n>=1&&n<=9){{const c=clipAt(currentIndex);if(c?.takes.some(t=>t.take_number===n))selectTake(c.clip_id,n,currentIndex);}}}});
refreshState();
</script>
</body>
</html>'''


def create_viewer_app(
    run_dir: str | Path,
    *,
    auto_assemble_on_selection: bool = False,
    preload_adjacent: bool = True,
) -> FastAPI:
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "08_selected.json").exists():
        raise FileNotFoundError(f"Not a completed Book2Video run: {run_dir}")
    state = ViewerState(run_dir, auto_assemble_on_selection=auto_assemble_on_selection)
    app = FastAPI(title="Book2Video Viewer")
    app.mount("/media", StaticFiles(directory=str(run_dir)), name="media")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return viewer_html(preload_adjacent=preload_adjacent)

    @app.get("/api/state")
    async def api_state() -> dict[str, Any]:
        return state.snapshot()

    @app.post("/api/select")
    async def api_select(selection: TakeSelectionRequest) -> dict[str, Any]:
        try:
            return state.select(selection.clip_id, selection.take_number)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/assemble")
    async def api_assemble() -> dict[str, Any]:
        return {"rough_cut": state.assemble()}

    return app


def serve_viewer(
    run_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    auto_assemble_on_selection: bool = False,
    preload_adjacent: bool = True,
) -> None:
    import uvicorn

    app = create_viewer_app(
        run_dir,
        auto_assemble_on_selection=auto_assemble_on_selection,
        preload_adjacent=preload_adjacent,
    )
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
