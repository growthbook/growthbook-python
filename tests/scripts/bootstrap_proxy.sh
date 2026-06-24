#!/usr/bin/env bash
# Boot a local growthbook-proxy in Docker and verify the /api/eval endpoint
# is reachable. Prints the env vars you need to export for
# `verify_remote_eval.py --real`.
#
# Required env vars in:
#   GB_API_KEY     Server-side secret API key from GrowthBook → Settings → API Keys
#                  (the proxy uses this to fetch features from the GrowthBook API)
#   GB_CLIENT_KEY  SDK Connection client key (e.g. sdk-...)
#                  Settings → SDK Connections → your connection → "Client Key"
#
# Optional:
#   GB_API_HOST    Upstream GrowthBook API (default: https://api.growthbook.io)
#   GB_PROXY_PORT  Host port to bind (default: 3300)
#   GB_PROXY_NAME  Container name (default: growthbook-proxy-verify)
#
# Usage:
#   chmod +x tests/scripts/bootstrap_proxy.sh
#   GB_API_KEY=... GB_CLIENT_KEY=sdk-... ./tests/scripts/bootstrap_proxy.sh
#   eval "$(./tests/scripts/bootstrap_proxy.sh --export)"   # to set env vars in the current shell

set -euo pipefail

: "${GB_API_KEY:?GB_API_KEY is required (server-side secret from GrowthBook Settings → API Keys)}"
: "${GB_CLIENT_KEY:?GB_CLIENT_KEY is required (SDK Connection client key)}"

API_HOST="${GB_API_HOST:-https://api.growthbook.io}"
PORT="${GB_PROXY_PORT:-3300}"
NAME="${GB_PROXY_NAME:-growthbook-proxy-verify}"
EXPORT_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --export) EXPORT_ONLY=1 ;;
    --stop)
      docker rm -f "$NAME" >/dev/null 2>&1 || true
      echo "stopped $NAME" >&2
      exit 0
      ;;
  esac
done

log() { [ "$EXPORT_ONLY" -eq 0 ] && echo "$@" >&2 || true; }

# If a previous container is still around, reuse it if healthy; otherwise replace.
if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
  log "→ Reusing running container $NAME"
elif docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  log "→ Removing stale container $NAME"
  docker rm -f "$NAME" >/dev/null
fi

if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
  log "→ Pulling growthbook/proxy:latest (cached if local)"
  docker pull growthbook/proxy:latest >/dev/null
  log "→ Starting $NAME on :$PORT (upstream: $API_HOST)"
  docker run -d \
    --name "$NAME" \
    -p "$PORT:3300" \
    -e GROWTHBOOK_API_HOST="$API_HOST" \
    -e SECRET_API_KEY="$GB_API_KEY" \
    -e CACHE_ENGINE=memory \
    -e USE_HTTP=1 \
    growthbook/proxy:latest >/dev/null
fi

URL="http://localhost:$PORT"

# Wait for the proxy to accept connections (max ~10s).
log "→ Waiting for proxy to accept connections at $URL..."
for _ in $(seq 1 50); do
  if curl -sf -o /dev/null "$URL/healthcheck" 2>/dev/null \
       || curl -sf -o /dev/null --max-time 1 "$URL/" 2>/dev/null; then
    break
  fi
  sleep 0.2
done

# Real probe: POST /api/eval/<key> with a minimal payload. If the proxy is wired
# wrong (bad API key, upstream unreachable, etc.), this surfaces it cleanly.
log "→ Probing POST $URL/api/eval/${GB_CLIENT_KEY:0:8}..."
PROBE_OUT=$(curl -sS -w '\n__HTTP__%{http_code}' -X POST \
  "$URL/api/eval/$GB_CLIENT_KEY" \
  -H "Content-Type: application/json" \
  -d '{"attributes":{"id":"probe"},"forcedFeatures":[],"forcedVariations":{},"url":""}' \
  || echo "__HTTP__000")
HTTP_CODE=$(printf '%s' "$PROBE_OUT" | awk -F'__HTTP__' '{print $2}' | tail -n1)
BODY=$(printf '%s' "$PROBE_OUT" | sed 's/__HTTP__[0-9]*$//')

if [ "$HTTP_CODE" != "200" ]; then
  log "${R:-}✗ Proxy probe failed (HTTP $HTTP_CODE)${X:-}"
  log "  Response: $BODY"
  log ""
  log "  Common causes:"
  log "    - GB_API_KEY isn't a server-side secret (must start 'secret_')"
  log "    - GB_CLIENT_KEY isn't tied to a published SDK Connection"
  log "    - GB_API_HOST unreachable from inside Docker (try $API_HOST in a browser)"
  log "    - The SDK Connection has no environment or features published yet"
  log "  Tail container logs: docker logs $NAME"
  exit 1
fi

# Sanity-check the response shape so we catch wire-format drift early.
if ! printf '%s' "$BODY" | grep -q '"features"'; then
  log "${R:-}✗ Proxy returned 200 but body lacks 'features' key${X:-}"
  log "  Body: $BODY"
  exit 1
fi

if [ "$EXPORT_ONLY" -eq 1 ]; then
  # Print plain export lines for `eval $(./bootstrap_proxy.sh --export)`
  echo "export GB_PROXY_URL=$URL"
  echo "export GB_CLIENT_KEY=$GB_CLIENT_KEY"
  exit 0
fi

log ""
log "✓ Proxy ready at $URL"
log ""
log "Run the verification suite against it:"
log ""
log "    export GB_PROXY_URL=$URL"
log "    export GB_CLIENT_KEY=$GB_CLIENT_KEY"
log "    python3 tests/scripts/verify_remote_eval.py --real"
log ""
log "Tear down when you're done:"
log "    $0 --stop"
