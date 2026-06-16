#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# vLLM Smart Gateway — end-to-end smoke test (run on the inference host).
#
# Validates a RUNNING gateway over its HTTP API: model listing, status,
# cold-start completion, warm routing, streaming, and concurrency. It is
# version-agnostic (only uses the OpenAI-compatible endpoints + /gateway/status),
# so it works against any build of the gateway.
#
# Usage:
#   GATEWAY_URL=http://localhost:9003 ./scripts/smoke_test.sh [MODEL] [MODEL2]
#
# Env:
#   GATEWAY_URL    base URL of the gateway        (default http://localhost:9003)
#   SMOKE_MODEL    model name to exercise         (default: first from /v1/models)
#   SMOKE_MODEL2   second model (placement test)  (default: second from /v1/models, if any)
#   COLD_TIMEOUT   seconds to allow a cold start  (default 1200 — first run downloads weights)
#   WARM_TIMEOUT   seconds for a warm request     (default 120)
#   CONCURRENCY    parallel warm requests         (default 5)
#   MAX_TOKENS     tokens per completion          (default 16)
#   SKIP_MODEL2    set to 1 to skip the 2nd-model placement test
#
# Exit code: 0 if all REQUIRED checks pass, 1 otherwise. (Some checks WARN-only.)
# Requires: bash, curl, python3.
# ---------------------------------------------------------------------------
set -uo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:9003}"
COLD_TIMEOUT="${COLD_TIMEOUT:-1200}"
WARM_TIMEOUT="${WARM_TIMEOUT:-120}"
CONCURRENCY="${CONCURRENCY:-5}"
MAX_TOKENS="${MAX_TOKENS:-16}"

PASS=0; FAIL=0; WARN=0
c_g=$'\033[32m'; c_r=$'\033[31m'; c_y=$'\033[33m'; c_b=$'\033[1m'; c_0=$'\033[0m'
pass()  { printf "  ${c_g}✓${c_0} %s\n" "$1"; PASS=$((PASS+1)); }
fail()  { printf "  ${c_r}✗ %s${c_0}\n" "$1"; FAIL=$((FAIL+1)); }
warn()  { printf "  ${c_y}! %s${c_0}\n" "$1"; WARN=$((WARN+1)); }
head()  { printf "\n${c_b}== %s ==${c_0}\n" "$1"; }

# json_get <json-string> <python-expr over `d`>  -> prints result or empty on error
json_get() { python3 -c "import sys,json
try:
    d=json.load(sys.stdin)
    print($2)
except Exception:
    pass" <<<"$1" 2>/dev/null; }

command -v curl >/dev/null   || { echo "curl not found"; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }

# ---------------------------------------------------------------------------
head "Preflight: gateway reachable @ $GATEWAY_URL"
models_json="$(curl -fsS --max-time 15 "$GATEWAY_URL/v1/models" 2>/dev/null)"
if [[ -z "$models_json" ]]; then
    fail "GET /v1/models — gateway not reachable (is it up? correct GATEWAY_URL?)"
    echo; echo "${c_r}Aborting: cannot reach the gateway.${c_0}"; exit 1
fi
pass "GET /v1/models reachable"
mapfile -t ALL_MODELS < <(json_get "$models_json" "'\n'.join(m['id'] for m in d.get('data',[]))")
if [[ ${#ALL_MODELS[@]} -eq 0 ]]; then
    fail "No models listed by /v1/models (check ALLOWED_MODELS_JSON / models.yaml)"; exit 1
fi
pass "Models available: ${ALL_MODELS[*]}"

MODEL="${1:-${SMOKE_MODEL:-${ALL_MODELS[0]}}}"
MODEL2="${2:-${SMOKE_MODEL2:-${ALL_MODELS[1]:-}}}"
echo "  Using primary model: ${c_b}$MODEL${c_0}${MODEL2:+   secondary: $MODEL2}"

# ---------------------------------------------------------------------------
head "Gateway status endpoint"
status_json="$(curl -fsS --max-time 15 "$GATEWAY_URL/gateway/status" 2>/dev/null)"
if [[ -z "$status_json" ]]; then
    fail "GET /gateway/status returned nothing"
else
    pass "GET /gateway/status OK"
    total="$(json_get "$status_json" "d.get('total_gpu_vram_mib')")"
    [[ -n "$total" && "$total" != "0" ]] && pass "Total managed VRAM: ${total} MiB" \
        || warn "total_gpu_vram_mib is 0/absent — VRAM probe may have failed (check docker.sock + nvidia runtime)"
    pools="$(json_get "$status_json" "list((d.get('pools') or {}).keys())")"
    [[ -n "$pools" ]] && echo "  Pools: $pools (gpus: $(json_get "$status_json" "len(d.get('gpus') or {})"))"
fi

# completion helpers --------------------------------------------------------
# do_chat <model> <timeout> -> sets HTTP (status) and BODY (file)
BODY="$(mktemp)"; trap 'rm -f "$BODY"' EXIT
do_chat() {
    local model="$1" tmo="$2"
    HTTP="$(curl -s -o "$BODY" -w '%{http_code}' --max-time "$tmo" \
        -H 'Content-Type: application/json' \
        -X POST "$GATEWAY_URL/v1/chat/completions" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: pong\"}],\"max_tokens\":$MAX_TOKENS,\"temperature\":0}" )"
}

# ---------------------------------------------------------------------------
head "Cold start: first completion for '$MODEL' (may take minutes — downloads + loads)"
t0=$(date +%s); do_chat "$MODEL" "$COLD_TIMEOUT"; t1=$(date +%s)
if [[ "$HTTP" == "200" ]]; then
    content="$(json_get "$(cat "$BODY")" "d['choices'][0]['message']['content']")"
    if [[ -n "$content" ]]; then
        pass "Cold completion OK in $((t1-t0))s — reply: $(echo "$content" | head -c 60)"
    else
        fail "HTTP 200 but no choices[0].message.content (body: $(head -c 200 "$BODY"))"
    fi
else
    fail "Cold completion HTTP $HTTP (body: $(head -c 300 "$BODY"))"
    echo "  Hint: check 'docker logs vllm_gateway' for the per-model vLLM command + errors."
fi

# ---------------------------------------------------------------------------
head "Status reflects the loaded model"
status_json="$(curl -fsS --max-time 15 "$GATEWAY_URL/gateway/status" 2>/dev/null)"
ac_count="$(json_get "$status_json" "len(d.get('active_containers') or {})")"
[[ -n "$ac_count" && "$ac_count" -ge 1 ]] 2>/dev/null && pass "active_containers: $ac_count" \
    || warn "no active_containers reported (older build, or already idle-unloaded)"
# Per-container fields present on the multi-GPU builds (gpu_uuids/status) — informational.
gpu_uuids="$(json_get "$status_json" "[v.get('gpu_uuids') for v in (d.get('active_containers') or {}).values()]")"
[[ -n "$gpu_uuids" && "$gpu_uuids" != "[None]" && "$gpu_uuids" != "[]" ]] \
    && pass "Container pinned to GPU(s): $gpu_uuids" \
    || warn "no gpu_uuids in status (pre-multi-GPU build, or single-pool)"

# ---------------------------------------------------------------------------
head "Warm request (fast path — model already loaded)"
t0=$(date +%s); do_chat "$MODEL" "$WARM_TIMEOUT"; t1=$(date +%s)
[[ "$HTTP" == "200" ]] && pass "Warm completion OK in $((t1-t0))s" \
    || fail "Warm completion HTTP $HTTP (body: $(head -c 200 "$BODY"))"

# ---------------------------------------------------------------------------
head "Streaming request"
stream_out="$(curl -s --max-time "$WARM_TIMEOUT" -H 'Content-Type: application/json' \
    -X POST "$GATEWAY_URL/v1/chat/completions" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count: 1 2 3\"}],\"max_tokens\":$MAX_TOKENS,\"stream\":true}" 2>/dev/null)"
n_chunks="$(grep -c '^data:' <<<"$stream_out" 2>/dev/null || echo 0)"
if [[ "$n_chunks" -ge 1 ]] 2>/dev/null && grep -q '\[DONE\]' <<<"$stream_out"; then
    pass "Streaming OK ($n_chunks SSE chunks, [DONE] received)"
else
    fail "Streaming returned $n_chunks chunks / no [DONE] (got: $(echo "$stream_out" | head -c 200))"
fi

# ---------------------------------------------------------------------------
head "Concurrency: $CONCURRENCY parallel warm requests (exercises locking)"
pids=(); rc_dir="$(mktemp -d)"
for i in $(seq 1 "$CONCURRENCY"); do
    ( code="$(curl -s -o /dev/null -w '%{http_code}' --max-time "$WARM_TIMEOUT" \
        -H 'Content-Type: application/json' -X POST "$GATEWAY_URL/v1/chat/completions" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi $i\"}],\"max_tokens\":$MAX_TOKENS}")"
      echo "$code" > "$rc_dir/$i" ) &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
ok=$(grep -l '^200$' "$rc_dir"/* 2>/dev/null | wc -l)
rm -rf "$rc_dir"
[[ "$ok" -eq "$CONCURRENCY" ]] && pass "All $CONCURRENCY concurrent requests returned 200" \
    || fail "Only $ok/$CONCURRENCY concurrent requests returned 200 (possible serialization/lock issue)"

# ---------------------------------------------------------------------------
if [[ -n "${MODEL2:-}" && "${SKIP_MODEL2:-0}" != "1" && "$MODEL2" != "$MODEL" ]]; then
    head "Placement / eviction: second model '$MODEL2'"
    t0=$(date +%s); do_chat "$MODEL2" "$COLD_TIMEOUT"; t1=$(date +%s)
    if [[ "$HTTP" == "200" ]]; then
        pass "Second model served in $((t1-t0))s"
        status_json="$(curl -fsS --max-time 15 "$GATEWAY_URL/gateway/status" 2>/dev/null)"
        echo "  Resident now: $(json_get "$status_json" "[v.get('model_id') for v in (d.get('active_containers') or {}).values()]")"
        # Re-hit model 1: if it was evicted this is a cold start again; either way it must succeed.
        do_chat "$MODEL" "$COLD_TIMEOUT"
        [[ "$HTTP" == "200" ]] && pass "Model 1 still serves after model 2 (swap/co-resident OK)" \
            || fail "Model 1 failed after model 2 placement (HTTP $HTTP)"
    else
        fail "Second model HTTP $HTTP (body: $(head -c 200 "$BODY"))"
    fi
else
    head "Placement test skipped (need a distinct SMOKE_MODEL2; or SKIP_MODEL2=1)"
fi

# ---------------------------------------------------------------------------
head "Summary"
printf "  ${c_g}%d passed${c_0}, ${c_r}%d failed${c_0}, ${c_y}%d warnings${c_0}\n" "$PASS" "$FAIL" "$WARN"
if [[ "$FAIL" -eq 0 ]]; then
    printf "${c_g}${c_b}SMOKE TEST PASSED${c_0}\n"; exit 0
else
    printf "${c_r}${c_b}SMOKE TEST FAILED${c_0} — see failures above; check 'docker logs vllm_gateway'.\n"; exit 1
fi
