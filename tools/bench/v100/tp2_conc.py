#!/usr/bin/env python3
"""Concurrent-request test against the TP2 proxy (stdlib only; needs --max-inflight >= 2 on the proxy).

  conc_test.py load   --n 2 --level 8K --tokens 256 --stagger 0      # N clients at once, per-client and aggregate tok/s
  conc_test.py drop   --level 8K                                     # client A drops mid-stream while B runs; B must finish, follow-up must pass
  conc_test.py stress --rounds 10 --n 4 --tokens 48                  # bursts of N near-simultaneous requests; expects every one to answer 200
"""
import argparse, http.client, json, os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ctx_prompt.py lives next to this file in the repo
import ctx_prompt

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["load", "drop", "stress"])
ap.add_argument("--url", default="http://127.0.0.1:18881")
ap.add_argument("--model", default="qwen38-ninfer-v100")
ap.add_argument("--key-file", default="/home/x/tpx/api-key")
ap.add_argument("--level", default="8K")
ap.add_argument("--kind", default="code")
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--stagger", type=float, default=0.0, help="seconds between client starts")
ap.add_argument("--rounds", type=int, default=1)
a = ap.parse_args()
key = open(a.key_file).read().strip()
host, port = a.url.split("//")[1].split(":"); port = int(port)

def prompt_for(i):
    blocks = ctx_prompt.LEVELS[a.level][a.kind]
    return ctx_prompt.build_prompt(a.kind, blocks, offset=1000 * (i + 1) + 7)

def chat_body(text, max_tokens, stream=False, seed=1):
    return json.dumps({"model": a.model, "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0, "seed": seed, "stream": stream})

def one(i, out, max_tokens, stream=False, drop_after=None):
    t0 = time.time()
    c = http.client.HTTPConnection(host, port, timeout=3600)
    try:
        c.request("POST", "/v1/chat/completions", body=chat_body(prompt_for(i), max_tokens, stream),
                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        r = c.getresponse()
        if stream:
            chunks = 0
            while True:
                line = r.readline()
                if not line: break
                if line.startswith(b"data:"): chunks += 1
                if drop_after and chunks >= drop_after:
                    if c.sock is not None: c.sock.close()
                    out[i] = {"status": r.status, "dropped_after": chunks, "wall": time.time() - t0}; return
            out[i] = {"status": r.status, "chunks": chunks, "wall": time.time() - t0}
            return
        data = r.read()
        d = json.loads(data) if r.status == 200 else {}
        t = d.get("timings", {})
        out[i] = {"status": r.status, "wall": time.time() - t0, "gen": t.get("predicted_n"), "tok_s": t.get("predicted_per_second"),
                  "prompt_n": t.get("prompt_n"), "prefill_s": (t.get("prompt_ms") or 0) / 1000, "err": data[:120].decode(errors="replace") if r.status != 200 else ""}
    except Exception as e:
        out[i] = {"status": "EXC", "err": repr(e)[:160], "wall": time.time() - t0}
    finally:
        c.close()

if a.mode == "load":
    for rnd in range(a.rounds):
        out = {}; ths = []; t0 = time.time()
        for i in range(a.n):
            th = threading.Thread(target=one, args=(i, out, a.tokens)); th.start(); ths.append(th); time.sleep(a.stagger)
        for th in ths: th.join()
        wall = time.time() - t0
        gen = sum((o.get("gen") or 0) for o in out.values())
        for i in sorted(out):
            o = out[i]; print(f"  client{i}: status={o['status']} prompt={o.get('prompt_n')} prefill={o.get('prefill_s', 0):.1f}s gen={o.get('gen')} tok/s={o.get('tok_s') or 0:.1f} wall={o['wall']:.1f}s {o.get('err','')}")
        print(f"round {rnd}: n={a.n} level={a.level} total generated {gen} tokens in {wall:.1f}s wall -> aggregate {gen / wall:.1f} tok/s")
elif a.mode == "drop":
    out = {}
    tb = threading.Thread(target=one, args=(1, out, 400)); tb.start()
    time.sleep(0.2)
    ta = threading.Thread(target=one, args=(0, out, 400, True, 20)); ta.start()
    ta.join(); tb.join()
    print("A (dropped):", out.get(0)); print("B (kept):   ", out.get(1))
    ok = out.get(1, {}).get("status") == 200 and out.get(1, {}).get("gen") == 400
    f = {}; one(2, f, 30); print("follow-up:", f.get(2))
    print("DROP TEST", "PASS" if ok and f.get(2, {}).get("status") == 200 else "FAIL")
else:
    bad = 0
    for rnd in range(a.rounds):
        out = {}; ths = []
        for i in range(a.n):
            th = threading.Thread(target=one, args=(i, out, a.tokens)); th.start(); ths.append(th); time.sleep(a.stagger)
        for th in ths: th.join()
        st = [out[i]["status"] for i in sorted(out)]
        bad += sum(1 for s in st if s != 200)
        print(f"round {rnd}: statuses {st}  tok/s {[round(out[i].get('tok_s') or 0) for i in sorted(out)]}")
    print("STRESS", "PASS" if bad == 0 else f"FAIL ({bad} non-200)")
