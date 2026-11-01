#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_dir"
exec codex resume 01a19174-e52e-7e40-9f3a-69ffab792609 --cd "$project_dir" "$@"
