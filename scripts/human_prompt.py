#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human-in-the-loop interactive console for ED4 AI
- Chat with the LLM using current screen context
- Inject next action or flag bad decisions in real-time
- Inspect latest frame metadata and open the image

Commands:
  /chat <message>      : Ask the model with current context
  /action <key>        : Force next action (left,right,up,down,space,enter,z,x,a,s,1,2,esc)
  /flag                : Flag current/last decision as bad (penalize)
  /show                : Show latest meta summary
  /open                : Open latest.png with default viewer (Windows)
  /interval <n>        : Guidance: set HERO4_LLM_INTERVAL for next run
  /events <csv>        : Guidance: set HERO4_LLM_EVENTS for next run
  /help                : Show help
  /quit                : Exit

Env:
  HERO4_SNAPSHOT_DIR (default: snapshots)
  HERO4_OLLAMA_URL   (default: http://localhost:11434)
  HERO4_MODEL_NAME   (default: qwen2.5-coder:7b)
"""

import os
import os
import sys
import json
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageStat
try:
    # Reuse RAG DB from main AI to store chats
    from isolated_rag_ai import RAGDatabase
except Exception:
    RAGDatabase = None
from datetime import datetime

SNAP_DIR = os.environ.get('HERO4_SNAPSHOT_DIR', 'snapshots')
OLLAMA_URL = os.environ.get('HERO4_OLLAMA_URL', 'http://localhost:11434')
MODEL_NAME = os.environ.get('HERO4_MODEL_NAME', 'qwen2.5-coder:7b')

LATEST_IMG = os.path.join(SNAP_DIR, 'latest.png')
LATEST_META = os.path.join(SNAP_DIR, 'latest.json')
FEEDBACK_FILE = os.environ.get('HERO4_FEEDBACK_FILE', 'feedback.json')

ALLOWED_ACTIONS = {'left','right','up','down','space','enter','z','x','a','s','1','2','esc'}

async def llm_generate(prompt: str, model: str = MODEL_NAME, url: str = OLLAMA_URL) -> str:
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {'temperature': 0.2, 'max_tokens': 256, 'num_ctx': 2048},
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{url}/api/generate", json=payload) as resp:
            if resp.status != 200:
                return f"[error] LLM HTTP {resp.status}"
            data = await resp.json()
            return data.get('response', '')

def load_latest_meta():
    try:
        with open(LATEST_META, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def get_latest_image_stats():
    try:
        if not os.path.exists(LATEST_IMG):
            return None
        im = Image.open(LATEST_IMG).convert('RGB')
        w, h = im.size
        st = ImageStat.Stat(im)
        mean_r, mean_g, mean_b = [float(x) for x in st.mean]
        return {
            'path': os.path.abspath(LATEST_IMG),
            'width': w,
            'height': h,
            'mean_rgb': [round(mean_r,1), round(mean_g,1), round(mean_b,1)],
        }
    except Exception:
        return None

def ensure_chat_dir():
    ts = datetime.now().strftime('%Y%m%d')
    d = os.path.join(SNAP_DIR, f'chat_{ts}')
    os.makedirs(d, exist_ok=True)
    return d

def next_chat_index(chat_dir: str) -> int:
    try:
        existing = [p for p in os.listdir(chat_dir) if p.startswith('chat_') and p.endswith('.json')]
        idxs = []
        for name in existing:
            try:
                i = int(name.split('_')[1].split('.')[0])
                idxs.append(i)
            except Exception:
                pass
        return (max(idxs) + 1) if idxs else 1
    except Exception:
        return 1

def save_chat_snapshot(user_msg: str, model_resp: str, meta: dict):
    chat_dir = ensure_chat_dir()
    idx = next_chat_index(chat_dir)
    base = os.path.join(chat_dir, f'chat_{idx:04d}')
    # copy latest.png if present
    img_saved = None
    try:
        if os.path.exists(LATEST_IMG):
            # Open and resave to ensure a consistent copy
            im = Image.open(LATEST_IMG)
            im.save(base + '.png')
            img_saved = base + '.png'
    except Exception:
        img_saved = None
    info = {
        'ts': datetime.now().isoformat(),
        'user_message': user_msg,
        'model_response': model_resp,
        'latest_meta': meta,
        'latest_image_stats': get_latest_image_stats(),
        'saved_image': img_saved,
    }
    try:
        with open(base + '.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def build_chat_prompt(user_msg: str) -> str:
    meta = load_latest_meta() or {}
    step = meta.get('step')
    situation = meta.get('situation')
    action = meta.get('action')
    flagged = meta.get('flagged')
    brightness = meta.get('brightness')
    movement = meta.get('movement')
    stats = get_latest_image_stats()
    if stats:
        img_desc = f"{stats['width']}x{stats['height']} meanRGB={stats['mean_rgb']}"
    else:
        img_desc = "(이미지 없음)"
    desc = f"밝기 {brightness} mv {movement}" if brightness is not None else "(메타 없음)"
    ctx = (
        f"ED4 실시간 컨텍스트\n"
        f"- step: {step}\n- situation: {situation}\n- last_action: {action}\n- flagged: {flagged}\n"
        f"- latest.png: {os.path.abspath(LATEST_IMG)}\n- image: {img_desc}\n- meta: {desc}\n\n"
    )
    prompt = (
        f"당신은 ED4 플레이 보조입니다. 아래 컨텍스트를 참고해 사용자 질의에 간결하고 실행가능한 조언을 하세요.\n\n"
        f"{ctx}"
        f"[사용자 질문]\n{user_msg}\n\n"
        f"[응답 형식]\n- 분석 요약 1~2줄\n- 권장 행동 또는 판단 근거 1~2줄\n"
    )
    return prompt

def write_feedback(next_action: str = None, flag_bad: bool = False):
    data = {}
    if next_action:
        data['next_action'] = next_action
    if flag_bad:
        data['flag_bad'] = True
    if not data:
        return False
    try:
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[error] feedback write failed: {e}")
        return False

HELP = __doc__.strip()

def print_summary():
    meta = load_latest_meta()
    print("\n--- latest ---")
    print(f"image: {os.path.abspath(LATEST_IMG)}")
    if meta:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    else:
        print("(no meta)")
    print("--------------\n")

async def main():
    os.makedirs(SNAP_DIR, exist_ok=True)
    db = None
    if RAGDatabase is not None:
        try:
            db = RAGDatabase()
        except Exception:
            db = None
    print("🗣️ 인간-상호작용 콘솔 (타이핑 후 Enter) — /help 로 명령 보기")
    print_summary()
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break
        if not line:
            continue
        if line == '/quit':
            break
        if line == '/help':
            print(HELP)
            continue
        if line == '/show':
            print_summary()
            continue
        if line == '/open':
            try:
                os.startfile(os.path.abspath(LATEST_IMG))
            except Exception as e:
                print(f"[error] open: {e}")
            continue
        if line.startswith('/interval '):
            _, v = line.split(' ', 1)
            print(f"다음 실행에서 환경변수 HERO4_LLM_INTERVAL={v} 로 설정하세요.")
            continue
        if line.startswith('/events '):
            _, v = line.split(' ', 1)
            print(f"다음 실행에서 환경변수 HERO4_LLM_EVENTS={v} 로 설정하세요.")
            continue
        if line.startswith('/action '):
            _, act = line.split(' ', 1)
            act = act.strip().lower()
            if act not in ALLOWED_ACTIONS:
                print(f"[warn] invalid action: {act}")
                continue
            ok = write_feedback(next_action=act)
            print("✔ next_action queued" if ok else "✖ failed")
            continue
        if line == '/flag':
            ok = write_feedback(flag_bad=True)
            print("✔ flag_bad queued" if ok else "✖ failed")
            continue
        if line.startswith('/chat '):
            _, msg = line.split(' ', 1)
            prompt = build_chat_prompt(msg)
            print("… 모델 응답 대기 …")
            try:
                resp = await llm_generate(prompt)
                print("\n=== LLM ===")
                print(resp.strip())
                print("=== END ===\n")
                # Save per-chat snapshot + meta
                save_chat_snapshot(user_msg=msg, model_resp=resp, meta=load_latest_meta() or {})
                # Save to RAG DB as human guidance
                if db is not None:
                    meta = load_latest_meta() or {}
                    situation = meta.get('situation') or 'general'
                    step = meta.get('step')
                    try:
                        db.store_human_chat(user_message=msg, model_response=resp, situation_type=situation, screen_step=step)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[error] chat: {e}")
            continue
        # default: treat as chat
        prompt = build_chat_prompt(line)
        print("… 모델 응답 대기 …")
        try:
            resp = await llm_generate(prompt)
            print("\n=== LLM ===")
            print(resp.strip())
            print("=== END ===\n")
            save_chat_snapshot(user_msg=line, model_resp=resp, meta=load_latest_meta() or {})
            if db is not None:
                meta = load_latest_meta() or {}
                situation = meta.get('situation') or 'general'
                step = meta.get('step')
                try:
                    db.store_human_chat(user_message=line, model_response=resp, situation_type=situation, screen_step=step)
                except Exception:
                    pass
        except Exception as e:
            print(f"[error] chat: {e}")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except RuntimeError:
        # Fallback for environments with existing loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
