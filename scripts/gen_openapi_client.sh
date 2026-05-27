#!/usr/bin/env bash
# Generate TypeScript types for the studio API from /openapi.json.
#
# Usage:
#   ./scripts/gen_openapi_client.sh                       # server already running on :8000
#   OAS_OPENAPI_URL=http://localhost:8000 ./scripts/gen_openapi_client.sh
#   OAS_OPENAPI_FILE=./openapi.json ./scripts/gen_openapi_client.sh   # offline / from a snapshot
#
# Output: apps/web/src/lib/api.generated.ts

set -euo pipefail

cd "$(dirname "$0")/.."

OUT="apps/web/src/lib/api.generated.ts"
URL="${OAS_OPENAPI_URL:-http://localhost:8000}"
FILE="${OAS_OPENAPI_FILE:-}"

# Resolve openapi-typescript: prefer the workspace install, fall back to npx.
if [[ -x "apps/web/node_modules/.bin/openapi-typescript" ]]; then
  GEN="apps/web/node_modules/.bin/openapi-typescript"
elif command -v openapi-typescript >/dev/null 2>&1; then
  GEN="openapi-typescript"
else
  GEN="npx --yes openapi-typescript@7"
fi

if [[ -n "$FILE" ]]; then
  $GEN "$FILE" --output "$OUT"
else
  $GEN "$URL/openapi.json" --output "$OUT"
fi

echo "Wrote $OUT"
