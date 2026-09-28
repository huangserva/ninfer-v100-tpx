#!/usr/bin/env python3
"""Functional scenarios for the TP2 service (proxy + two ranks). Stdlib only.

After every scenario it sends a small follow-up request and reads the lockstep file, so a
scenario passes only if the service still answers AND both ranks sit on the same unit count.

    python3 tp2_scenarios.py --port 18881 --model qwen38-ninfer --key-file /path/to/api-key \
        --lockstep-file /dev/shm/ninfer_tpx_lockstep            # all scenarios
    python3 tp2_scenarios.py ... basic stream image             # a subset

Scenarios: basic, stream, noseed, drop_stream, drop_prefill, timeout, maxtok, stop, image,
image_drop, anthropic, responses. `--container NAME` reads the lockstep file through
`docker exec` when the service runs in a container whose /dev/shm is not the host's.
"""
import argparse, base64, http.client, json, mmap, os, socket, struct, subprocess, sys, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx_prompt  # noqa: E402

ALL = ['basic', 'stream', 'noseed', 'drop_stream', 'drop_prefill', 'timeout', 'maxtok', 'stop',
       'image', 'image_drop', 'anthropic', 'responses']


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18881)
    ap.add_argument("--model", default="qwen38-ninfer")
    ap.add_argument("--key", default=None)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--lockstep-file", default="/dev/shm/ninfer_tp2_lockstep")
    ap.add_argument("--container", default=None, help="docker container to read the lockstep file in")
    ap.add_argument("scenarios", nargs="*", default=[])
    a = ap.parse_args(argv)
    a.key = a.key or (open(a.key_file).read().strip() if a.key_file else "k")
    return a


class Client:
    def __init__(self, a):
        self.a = a
        self.H = {"Content-Type": "application/json", "Authorization": "Bearer " + a.key}

    def units(self):
        a = self.a
        if a.container:
            code = (f"import struct;d=open('{a.lockstep_file}','rb').read(80);"
                    "print(*struct.unpack_from('<QQ',d,64))")
            out = subprocess.run(["docker", "exec", a.container, "python3", "-c", code],
                                 capture_output=True, text=True).stdout.split()
            return tuple(int(x) for x in out)
        with open(a.lockstep_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            v = struct.unpack_from('<QQ', mm, 64)
            mm.close()
            return v

    def post(self, path, body, timeout=600):
        c = http.client.HTTPConnection(self.a.host, self.a.port, timeout=timeout)
        c.request('POST', path, json.dumps(body, ensure_ascii=False).encode(), self.H)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def chat(self, content, **kw):
        b = {"model": self.a.model, "messages": [{"role": "user", "content": content}],
             "max_tokens": 128, "enable_thinking": False}
        b.update(kw)
        return b

    def check(self, tag, extra=""):
        time.sleep(1.5)
        st, data = self.post('/v1/chat/completions', self.chat("用一句话介绍你自己。", temperature=0, max_tokens=40))
        u = self.units()
        ok = st == 200 and u[0] == u[1]
        txt = json.loads(data)['choices'][0]['message']['content'][:40] if st == 200 else data[:120]
        print(f"[{'PASS' if ok else 'FAIL'}] {tag} {extra} | follow-up {st} units={u} {txt!r}", flush=True)
        return ok

    def stream_and_drop(self, body, after_chunks=None, after_seconds=None, path='/v1/chat/completions'):
        s = socket.create_connection((self.a.host, self.a.port))
        data = json.dumps(body, ensure_ascii=False).encode()
        head = (f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                f"Authorization: Bearer {self.a.key}\r\nContent-Length: {len(data)}\r\n\r\n")
        s.sendall(head.encode() + data)
        t0 = time.time()
        n = 0
        s.settimeout(1.0)
        while True:
            if after_seconds is not None and time.time() - t0 > after_seconds:
                break
            try:
                d = s.recv(4096)
            except socket.timeout:
                continue
            if not d:
                break
            n += 1
            if after_chunks is not None and n >= after_chunks:
                break
        s.close()
        return n, time.time() - t0


def png(w=64, h=64):
    raw = b''.join(b'\x00' + b''.join(bytes((x * 4 % 256, y * 4 % 256, 128)) for x in range(w)) for y in range(h))

    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + \
        chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')


def run(a):
    c = Client(a)
    fails = 0
    for w in a.scenarios or ALL:
        t = time.time()
        if w == 'basic':
            st, d = c.post('/v1/chat/completions', c.chat("解释什么是张量并行。", temperature=0, max_tokens=200))
            ok = c.check(w, f"status={st} tok={json.loads(d)['usage']['completion_tokens'] if st == 200 else d[:100]}")
        elif w == 'stream':
            n, dt = c.stream_and_drop(c.chat("写一首关于显卡的短诗。", temperature=0.7, top_p=0.8, seed=7, stream=True, max_tokens=300))
            ok = c.check(w, f"chunks={n} {dt:.1f}s")
        elif w == 'noseed':
            st1, d1 = c.post('/v1/chat/completions', c.chat("随便说三个城市名。", temperature=1.0, max_tokens=60))
            ok = c.check(w, f"status={st1}")
        elif w == 'drop_stream':
            n, dt = c.stream_and_drop(c.chat("从1数到500，用逗号分隔。", temperature=0, stream=True, max_tokens=2000), after_chunks=30)
            ok = c.check(w, f"dropped after {n} chunks {dt:.1f}s")
        elif w == 'drop_prefill':
            n, dt = c.stream_and_drop(c.chat(ctx_prompt.build_prompt('code', 200), temperature=0, stream=True, max_tokens=200), after_seconds=8)
            ok = c.check(w, f"dropped during prefill after {dt:.1f}s (chunks {n})")
        elif w == 'timeout':
            try:
                c.post('/v1/chat/completions', c.chat(ctx_prompt.build_prompt('zh-doc', 150), temperature=0, max_tokens=300), timeout=12)
                r = "no timeout"
            except (socket.timeout, TimeoutError):
                r = "client timed out at 12s"
            ok = c.check(w, r)
        elif w == 'maxtok':
            st, d = c.post('/v1/chat/completions', c.chat("详细介绍北京的历史。", temperature=0, max_tokens=5))
            j = json.loads(d)
            ok = c.check(w, f"finish={j['choices'][0]['finish_reason']} tok={j['usage']['completion_tokens']}")
        elif w == 'stop':
            st, d = c.post('/v1/chat/completions', c.chat("列出五种水果，每行一种。", temperature=0, max_tokens=200, stop=["\n"]))
            j = json.loads(d)
            ok = c.check(w, f"finish={j['choices'][0]['finish_reason']} text={j['choices'][0]['message']['content']!r}")
        elif w == 'image':
            img = base64.b64encode(png()).decode()
            b = {"model": a.model, "max_tokens": 80, "temperature": 0, "enable_thinking": False,
                 "messages": [{"role": "user", "content": [
                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                     {"type": "text", "text": "描述这张图片的颜色。"}]}]}
            st, d = c.post('/v1/chat/completions', b)
            txt = json.loads(d)['choices'][0]['message']['content'][:60] if st == 200 else d[:200]
            ok = c.check(w, f"status={st} {txt!r}")
        elif w == 'image_drop':
            img = base64.b64encode(png(256, 256)).decode()
            b = {"model": a.model, "max_tokens": 400, "temperature": 0, "stream": True, "enable_thinking": False,
                 "messages": [{"role": "user", "content": [
                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                     {"type": "text", "text": "非常详细地描述这张图片。"}]}]}
            n, dt = c.stream_and_drop(b, after_chunks=10)
            ok = c.check(w, f"dropped after {n} chunks")
        elif w == 'anthropic':
            st, d = c.post('/v1/messages', {"model": a.model, "max_tokens": 60, "temperature": 0,
                                            "messages": [{"role": "user", "content": "你好"}]})
            ok = c.check(w, f"status={st} {d[:80]!r}")
        elif w == 'responses':
            st, d = c.post('/v1/responses', {"model": a.model, "max_output_tokens": 60, "input": "说一句问候。"})
            ok = c.check(w, f"status={st} {d[:80]!r}")
        else:
            print(f"unknown scenario {w}", flush=True)
            ok = False
        fails += 0 if ok else 1
        print(f"   ({w} took {time.time()-t:.1f}s)", flush=True)
    print(f"{'ALL PASS' if fails == 0 else f'{fails} FAILED'}", flush=True)
    return fails


if __name__ == "__main__":
    sys.exit(1 if run(parse_args()) else 0)
