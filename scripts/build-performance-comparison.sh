#!/usr/bin/env bash
# Build one committed baseline and the current working tree with identical flags.
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
revision=${1:-HEAD}
if (( $# > 0 )); then shift; fi
before_commit=$(git -C "$repo_root" rev-parse --verify "${revision}^{commit}")
host_target=$(rustc -vV | sed -n 's/^host: //p')
build_target=$host_target
clean_flags=(--release)
remaining=("$@")
for ((index=0; index < ${#remaining[@]}; index++)); do
    case ${remaining[index]} in
        --) break ;;
        --target)
            build_target=${remaining[index+1]}
            clean_flags+=(--target "${remaining[index+1]}")
            break
            ;;
        --target=*)
            build_target=${remaining[index]#--target=}
            clean_flags+=("${remaining[index]}")
            break
            ;;
    esac
done
target_label=$(python3 - "$build_target" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).stem)
PY
)
output="$repo_root/target/performance-comparison/$target_label"
temporary=$(mktemp -d "${TMPDIR:-/tmp}/boxcraft-comparison.XXXXXX")
trap 'rm -rf "$temporary"' EXIT
mkdir -p "$output" "$temporary/before"
git -C "$repo_root" archive "$before_commit" | tar -x -C "$temporary/before"

# Cargo can reuse same-named workspace artifacts across source directories.
# Borrow dependency caches in an isolated target, then force the two Boxcraft
# crates to rebuild for each variant. Never clean the user's target directory.
python3 - "$repo_root/target" "$temporary/build" "$target_label" "$host_target" <<'PY'
from pathlib import Path
import shutil
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.mkdir()
caches = [Path('release')]
if sys.argv[3] != sys.argv[4] or (source / sys.argv[3]).is_dir():
    caches.append(Path(sys.argv[3]) / 'release')
for cache in caches:
    if (source / cache).is_dir():
        shutil.copytree(source / cache, destination / cache)
PY

# Give the baseline the same deterministic seed hook as the current game.
# This is the only change to its sources; simulation and rendering stay intact.
python3 - "$temporary/before/boxcraft/src/ui.rs" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
hook = '''    let requested_seed = std::env::args()
        .skip_while(|arg| arg != "--seed")
        .nth(1)
        .or_else(|| std::env::var("BOXCRAFT_SEED").ok());
    if let Some(seed) = requested_seed.and_then(|value| value.parse::<u64>().ok()) {
        return seed;
    }
'''
needle = 'fn random_world_seed() -> u64 {\n'
if 'std::env::var("BOXCRAFT_SEED")' not in source:
    if source.count(needle) != 1:
        raise SystemExit('Baseline has no supported seed hook; choose another revision.')
    path.write_text(source.replace(needle, needle + hook))
PY

build_variant() {
    local variant=$1 source_root=$2
    shift 2
    if ! (
        cd "$source_root"
        CARGO_TARGET_DIR="$temporary/build" cargo clean "${clean_flags[@]}" -p boxcraft -p boxcraft-core
        CARGO_TARGET_DIR="$temporary/build" cargo rustc --locked --release \
            -p boxcraft --bin boxcraft --message-format=json "$@"
    ) > "$output/$variant-build.json"; then
        python3 - "$output/$variant-build.json" <<'PY'
import json
from pathlib import Path
import sys

for line in Path(sys.argv[1]).read_text().splitlines():
    event = json.loads(line)
    if event.get('reason') == 'compiler-message' and event['message']['level'] == 'error':
        print(event['message'].get('rendered') or event['message']['message'], file=sys.stderr)
PY
        return 1
    fi
    python3 - "$output/$variant-build.json" "$output/$variant-boxcraft" <<'PY'
import json
from pathlib import Path
import shutil
import sys

executables = []
for line in Path(sys.argv[1]).read_text().splitlines():
    event = json.loads(line)
    if event.get('reason') == 'compiler-artifact' and event['target']['name'] == 'boxcraft':
        if event.get('executable'):
            executables.append(event['executable'])
if len(executables) != 1:
    raise SystemExit('Cargo did not report exactly one Boxcraft executable.')
shutil.copy2(executables[0], sys.argv[2])
PY
}

build_variant before "$temporary/before" "$@"
build_variant after "$repo_root" "$@"
{
    printf 'baseline: %s\n' "$before_commit"
    printf 'target: %s\n' "$build_target"
    printf 'after: current working tree based on %s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    printf 'flags: cargo rustc --locked --release -p boxcraft --bin boxcraft'
    if (( $# > 0 )); then printf ' %q' "$@"; fi
    printf '\nseed: --seed 7 (pass this when launching both binaries)\n'
    printf '\n'
    rustc --version
    git -C "$repo_root" status --short
} > "$output/build-info.txt"
printf 'Comparison binaries: %s/{before,after}-boxcraft\n' "$output"
