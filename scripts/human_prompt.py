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
import sys
import json
import asyncio
import aiohttp
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

def build_chat_prompt(user_msg: str) -> str:
    meta = load_latest_meta() or {}
    step = meta.get('step')
    situation = meta.get('situation')
    action = meta.get('action')
    flagged = meta.get('flagged')
    brightness = meta.get('brightness')
    movement = meta.get('movement')
    desc = f"밝기 {brightness} mv {movement}" if brightness is not None else "(메타 없음)"
    ctx = (
        f"ED4 실시간 컨텍스트\n"
        f"- step: {step}\n- situation: {situation}\n- last_action: {action}\n- flagged: {flagged}\n"
        f"- latest.png: {os.path.abspath(LATEST_IMG)}\n- meta: {desc}\n\n"
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
        except Exception as e:
            print(f"[error] chat: {e}")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except RuntimeError:
        # Fallback for environments with existing loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
