# -*- coding: utf-8 -*-
"""
RAG 성공률 리포트
- hero4_rag.db 의 screen_actions 테이블을 읽어 상황/행동별 성공률을 집계합니다.
- ai_reasoning(JSON) 에서 situation_type 을 추출하여 그룹핑합니다.

예)
  python scripts/rag_success_report.py --db hero4_rag.db --min-count 10
"""
from __future__ import annotations
import json
import sqlite3
import argparse
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=str, default='hero4_rag.db')
    ap.add_argument('--min-count', type=int, default=5, help='최소 표본 수 필터')
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute("""
        SELECT action_taken, result_success, ai_reasoning
        FROM screen_actions
        WHERE action_taken IS NOT NULL
    """)

    stats = defaultdict(lambda: defaultdict(lambda: {'n':0,'ok':0}))

    for action, success, reasoning_json in cur.fetchall():
        try:
            r = json.loads(reasoning_json) if reasoning_json else {}
        except Exception:
            r = {}
        scene = r.get('situation_type') or 'unknown'
        bucket = stats[scene][action]
        bucket['n'] += 1
        if int(success or 0) == 1:
            bucket['ok'] += 1

    for scene, actions in stats.items():
        print(f"\n[scene] {scene}")
        rows = []
        for a, s in actions.items():
            n, ok = s['n'], s['ok']
            if n >= args.min_count:
                rows.append((ok/n if n else 0.0, n, ok, a))
        rows.sort(reverse=True)
        for rate, n, ok, a in rows[:20]:
            print(f"  {a:12s}  rate={rate:5.2f}  n={n:4d}  ok={ok:4d}")

    conn.close()

if __name__ == '__main__':
    main()
