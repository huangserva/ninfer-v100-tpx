#!/usr/bin/env python3
"""Fault injection for the TP2 service: break one rank while a request runs, expect an error
for that request and automatic recovery by the proxy watchdog.

    python3 tp2_fault.py kill  --port 18881 --model qwen38-ninfer --key-file api-key   # SIGKILL rank 1
    python3 tp2_fault.py stop  ...                                                      # SIGSTOP rank 1 (stall)
    python3 tp2_fault.py proxy --container ninfer-tpx ...                               # kill the proxy (pid 1)

Rank processes are found by their listen port (`--rank-ports`, the proxy default is 18940,18941).
Expected: the in-flight request fails (503 or a dropped stream), then the follow-up request passes
within about 30 s and both ranks report the same unit count.
"""
import argparse, os, signal, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp2_scenarios as ts  # noqa: E402


def pid_of(port):
    out = subprocess.run(['pgrep', '-f', f'port {port}'], capture_output=True, text=True).stdout.split()
    if not out:
        sys.exit(f"no process listening on port {port} (rank not running?)")
    return int(out[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["kill", "stop", "proxy"])
    ap.add_argument("--rank-ports", default="18940,18941")
    args, rest = ap.parse_known_args()
    a = ts.parse_args(rest)
    c = ts.Client(a)
    ports = [int(x) for x in args.rank_ports.split(",")]
    res = {}

    def req():
        t = time.time()
        try:
            st, d = c.post('/v1/chat/completions', c.chat("从1数到3000，用逗号分隔。", temperature=0, max_tokens=4000), timeout=900)
            res['r'] = (st, d[:160])
        except Exception as e:
            res['r'] = ('EXC', repr(e)[:160])
        res['t'] = time.time() - t
    th = threading.Thread(target=req)
    th.start()
    time.sleep(6)
    if a.container:
        if args.mode == 'kill':
            code = ("import os,signal\nfor p in os.listdir('/proc'):\n  if p.isdigit():\n    try:\n"
                    "      c=open('/proc/'+p+'/cmdline','rb').read().split(b'\\0')\n    except OSError: continue\n"
                    f"    if b'{ports[1]}' in c: os.kill(int(p),signal.SIGKILL); print('killed',p)")
            subprocess.run(["docker", "exec", a.container, "python3", "-c", code])
            print('killed rank1 in container', flush=True)
        elif args.mode == 'stop':
            sys.exit("stop mode is not supported inside a container; run it on the host")
        else:
            subprocess.run(["docker", "exec", a.container, "kill", "-9", "1"])
            print('killed proxy (pid 1); docker should restart the container', flush=True)
    else:
        if args.mode == 'proxy':
            sys.exit("proxy mode needs --container")
        p1 = pid_of(ports[1])
        if args.mode == 'kill':
            os.kill(p1, signal.SIGKILL)
            print('killed rank1', p1, flush=True)
        else:
            os.kill(p1, signal.SIGSTOP)
            print('stopped rank1', p1, flush=True)
    th.join()
    print('client got', res['r'], f"after {res['t']:.1f}s", flush=True)
    for i in range(300):
        try:
            st, d = c.post('/v1/chat/completions', c.chat("用一句话介绍你自己。", temperature=0, max_tokens=40), timeout=60)
            if st == 200:
                break
        except Exception:
            pass
        time.sleep(3)
    ok = c.check(args.mode + '_recovery', f"recovered after ~{i*3}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
