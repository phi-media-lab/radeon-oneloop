#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

required_files=(
  README.md
  LICENSE
  NOTICE
  THIRD_PARTY_NOTICES.md
  AGENTS.md
  .codex/config.toml
  configs/formal_radeon_only.yaml
  data/registry.yaml
  ops/formal_run_registry.yaml
  ops/job_manifest.schema.json
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    printf 'missing required file: %s\n' "$path" >&2
    exit 1
  fi
done

python3 -m json.tool ops/job_manifest.schema.json >/dev/null

if command -v ruby >/dev/null 2>&1; then
  ruby -e 'require "yaml"; ARGV.each { |p| YAML.load_file(p) }' \
    configs/*.yaml data/*.yaml ops/*.yaml
else
  printf '%s\n' 'warning: ruby unavailable; YAML syntax was not parsed' >&2
fi

if git grep -nE '(gho_|github_pat_|AKIA[0-9A-Z]{16}|BEGIN (RSA |OPENSSH )?PRIVATE KEY)' -- . \
  ':!ops/validate_scaffold.sh'; then
  printf '%s\n' 'possible secret detected' >&2
  exit 1
fi

git diff --check
PYTHONPATH=src python3 -m unittest discover -s tests -v
bash -n ops/*.sh
printf '%s\n' 'scaffold validation passed'
