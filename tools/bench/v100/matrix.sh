#!/usr/bin/env bash
# The speed table of the README / article: 8K, 32K, 128K, 256K x code / zh-doc, one cold run and
# one prefix-cached run per cell, 1024 generated tokens each. Works against a single-GPU
# ninfer-serve or the TP2 proxy; the server decides the numbers, this script only asks.
#
#   URL=http://127.0.0.1:18881 MODEL=qwen38-ninfer KEYFILE=/path/to/api-key \
#     tools/bench/v100/matrix.sh results/v100-tp2
#
#   LEVELS="8K 32K 128K 256K"   levels to run (256K needs --max-context 262144 on the server)
#   KINDS="code zh-doc"
#   MODE=matrix                 matrix: seeds 1,2 per cell (cold, then cached)
#   MODE=robust                 the headline numbers: 186K code / 193K zh, seeds 1,2,3,4
#
# Each level uses its own block-number offset so prompts of different levels share no prefix
# and every first run is a true cold prefill; the JSON rows carry the real prompt_n.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: matrix.sh <output-dir>}
mkdir -p "$out"
URL=${URL:-http://127.0.0.1:18881}
MODEL=${MODEL:-qwen38-ninfer}
KEYFILE=${KEYFILE:-}
LEVELS=${LEVELS:-"8K 32K 128K 256K"}
KINDS=${KINDS:-"code zh-doc"}
MODE=${MODE:-matrix}
MAXTOK=${MAXTOK:-1024}
keyarg=()
[[ -n "$KEYFILE" ]] && keyarg=(--key-file "$KEYFILE")
jsonl="$out/rows.jsonl"
: > "$jsonl"
blocks() { python3 - "$1" "$2" <<'PY'
import sys, os
sys.path.insert(0, os.environ["HERE"]); import ctx_prompt
print(ctx_prompt.LEVELS[sys.argv[1]][sys.argv[2]])
PY
}
export HERE="$here"
echo "# $(date '+%F %T') url=$URL model=$MODEL mode=$MODE" | tee "$out/run.log"
if [[ "$MODE" == robust ]]; then
  for kind in $KINDS; do
    n=$(blocks 186K "$kind")
    python3 "$here/decode_bench.py" --url "$URL" --model "$MODEL" "${keyarg[@]}" \
      --kind "$kind" --blocks "$n" --seeds 1,2,3,4 --max-tokens "$MAXTOK" --tag "186K" --out "$jsonl" \
      | tee -a "$out/run.log"
  done
else
  i=0
  for lvl in $LEVELS; do
    off=$((i * 100000)); i=$((i + 1))
    for kind in $KINDS; do
      n=$(blocks "$lvl" "$kind")
      python3 "$here/decode_bench.py" --url "$URL" --model "$MODEL" "${keyarg[@]}" \
        --kind "$kind" --blocks "$n" --offset "$off" --seeds 1,2 --max-tokens "$MAXTOK" \
        --tag "$lvl" --out "$jsonl" | tee -a "$out/run.log"
    done
  done
fi
python3 - "$jsonl" <<'PY' | tee "$out/table.md"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if not r.get("summary") and "error" not in r]
print("| level | kind | prompt_n | cold tok/s | cached tok/s | accept | ms/round | cold prefill s |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
seen = {}
for r in rows:
    k = (r["tag"], r["kind"]); seen.setdefault(k, []).append(r)
for (tag, kind), rs in seen.items():
    cold = rs[0]; warm = [x["tok_s"] for x in rs[1:]]
    acc = [x["accept"] for x in rs if x["accept"] is not None]
    msr = [x["ms_per_round"] for x in rs if x["ms_per_round"]]
    f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
    print(f"| {tag} | {kind} | {cold['prompt_n']} | {f(cold['tok_s'])} | "
          f"{f(sum(warm)/len(warm)) if warm else '-'} | {f(sum(acc)/len(acc), 3) if acc else '-'} | "
          f"{f(sum(msr)/len(msr)) if msr else '-'} | {f(cold['prefill_s'])} |")
PY
echo "rows: $jsonl  table: $out/table.md"
