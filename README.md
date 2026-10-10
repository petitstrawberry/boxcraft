# Boxcraft

Boxcraft is a cross-platform first-person voxel sandbox written in Rust. Explore
procedurally generated terrain, reshape it block by block, and watch the world
move through a full day and night. Boxcraft runs as a native desktop application
and can also be built for Scarlet OS.

## Highlights

- Procedural worlds with oceans, beaches, plains, forests, deserts, mountains,
  snow-covered peaks, caves, and trees
- First-person movement, collision, jumping, block breaking, and block placement
- Sunlight, ambient occlusion, torch light, and a 20-minute day/night cycle
- Chunk streaming, background meshing, far-terrain level of detail, and an
  adjustable render distance
- A generated pixel-art texture atlas with no external game assets

## Run on desktop

You need Git, a recent Rust toolchain, and a graphics adapter supported by WGPU.
Clone the repository and run the desktop frontend:

```bash
git clone https://github.com/petitstrawberry/boxcraft.git
cd boxcraft
cargo run --release -p boxcraft
```

The release profile is recommended because world generation and meshing are
CPU-intensive. Desktop builds use Winit for window and input integration and
WGPU for rendering; on macOS, WGPU uses Metal.

The first build fetches ScarletUI and SGFX from their Git repositories.

## Controls

| Action | Input |
| --- | --- |
| Capture the pointer | Click the terrain or select **Capture pointer** |
| Look around | Move the mouse while the pointer is captured |
| Move | `W`, `A`, `S`, `D` |
| Jump | `Space` |
| Break a block | Left click |
| Place the selected block | Right click |
| Select a block | `1`–`9` |
| Release the pointer / close settings | `Esc` |
| Open or close settings | `O` |
| Decrease or increase render distance | `-` / `+` |
| Generate a new world | `R` |
| Toggle fullscreen | `F11` |

Scarlet's native gamepad input also works without capturing the pointer:

| Action | Gamepad / Joy-Con pair |
| --- | --- |
| Move at variable speed | Left stick |
| Look around | Right stick |
| Jump | South button (`B` on Joy-Con) |
| Break / place a block | Right / left trigger (`ZR` / `ZL`) |
| Previous / next block | Left / right shoulder (`L` / `R`) |
| Open / close settings | Start (`+`) |
| Toggle fullscreen | Select (`−`) |

Sticks have a radial dead zone. Losing focus or disconnecting a controller
releases its input. While settings are open, game movement stops and gamepad
menu navigation is enabled. Desktop Winit gamepad delivery is not implemented.

The numbered slots contain Grass, Dirt, Stone, Wood, Leaves, Sand, Snow, Air,
and Torch in that order. Air occupies slot `8` for inspection but cannot be
placed.

## Architecture

The workspace separates the game domain from platform integration:

- `boxcraft-core` is dependency-free and contains deterministic world
  generation, lighting, meshing, raycasts, player physics, and camera math.
- `boxcraft` provides the ScarletUI/SGFX frontend, input handling, texture
  generation, chunk streaming, and background mesh workers.

The frontend selects its platform backend at compile time:

| Target | Window and input | World rendering |
| --- | --- | --- |
| Desktop | ScarletUI with Winit | SGFX with WGPU |
| Scarlet OS | ScarletUI with SWS | SGFX with the runtime-selected VirGL, Adreno A6xx, or Maxwell backend |

## Development

Run the formatter, tests, and workspace check with Cargo:

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo check --workspace
```

For a reproducible environment, the included Nix flake supplies the Rust
toolchain and the additional Scarlet SDK tools:

```bash
nix develop
```

If you use direnv, entering the repository can activate the same environment
automatically after one approval:

```bash
direnv allow
```

## Performance comparisons

Run the repeatable CPU workloads with fixed seeds:

```bash
cargo run --release -p boxcraft-core --example performance
cargo test -p boxcraft performance_idle_work -- --ignored --nocapture
```

On an Apple M3 Pro, the median across seeds 7, 11 and 42 changed from
839 ms to 595 ms for world generation and from 138 ms to 56 ms for the
96 far chunks at render distance 5. These are CPU timings, not QEMU results or
GPU/FPS measurements. The full voxel/light data and every vertex/index in the
render-distance-5 meshes matched the baseline for all three seeds.

A second optimization pass reduced generation from 548 ms to 265 ms and near
meshing from 5.28 ms to 4.56 ms relative to the first optimized version (medians
of three runs each for seeds 7, 11 and 42 on the same Mac). Bulk writes resolve
copy-on-write once per storage page, cave generation reuses noise lattice
layers, and each face shares its lighting samples across four corners.
Terrain, propagated light and the complete near/far meshes remained
bit-identical. These timings measure CPU work independently of the 120 Hz
display limit; they do not establish an FPS improvement on Scarlet.

The rendering pass now culls mesh bounds against the six clip planes extracted
from the actual projection, including the laid-out canvas aspect ratio. Across
216 views (seeds 7/11/42, 24 yaw angles and three pitches at render distance 5),
this reduces submitted triangles from 18,634,786 to 14,472,854 (22.3%) and mesh
draws from 4,720 to 3,472 (26.4%) compared with the previous conservative test.
These are submission counts, not measured GPU time or FPS. Reproduce them with
`cargo test -p boxcraft performance_draw_workload -- --ignored --nocapture`.

Build a committed baseline and the current working tree with identical release
flags (requires Python 3):

```bash
scripts/build-performance-comparison.sh a341f37
```

The default command builds binaries for the current host. On Apple Silicon
macOS, run them with:

```bash
./target/performance-comparison/aarch64-apple-darwin/before-boxcraft --seed 7
./target/performance-comparison/aarch64-apple-darwin/after-boxcraft --seed 7
```

The binaries, Cargo build logs and build metadata are saved under
`target/performance-comparison/<target>/`, so builds for different platforms
stay separate. Scarlet RISC-V binaries are placed in
`target/performance-comparison/riscv64gc-unknown-scarlet/`; run those inside
Scarlet under QEMU. macOS cannot execute them directly.
The baseline receives only the same seed
override as the current game; its terrain, lighting and renderer stay intact.
Run both with `--seed 7` (or `BOXCRAFT_SEED=7`) and the same render distance,
window size, movement route and edit sequence. Reset also reuses this seed.
For QEMU, keep its CPU count, memory, GPU backend and image identical between
runs; record startup/loading time, movement FPS and edit stalls separately.

For Scarlet's dynamic SGFX backend, pass your SDK's linker flags after `--`.
For example, with a matching VirGL plugin already built:

```bash
scripts/build-performance-comparison.sh a341f37 --target riscv64gc-unknown-scarlet -- \
  -C link-arg=--as-needed \
  -C link-arg="$PWD/target/riscv64gc-unknown-scarlet/release/libsgfx_scarlet_virgl.so" \
  -C link-arg=--unresolved-symbols=ignore-all
```

The shared input follows SGFX's dynamic-backend build procedure: it lets LLD
emit imports provided by `/bin/scarlet-ld`. Audit the resulting ELF: only
`dlopen`, `dlsym` and `dlerror` may remain undefined, with no `DT_NEEDED` backend
dependency. The comparison script only builds binaries; it does not start QEMU.

## Build for Scarlet OS

Enter the Nix development shell, then build either supported Scarlet userspace
target:

```bash
cargo build --release -p boxcraft --target riscv64gc-unknown-scarlet
cargo build --release -p boxcraft --target aarch64-unknown-scarlet
```

To include Boxcraft in a Scarlet image, add a Cargo layer to the desired image
definition in a Scarlet checkout:

```toml
[[layers]]
kind = "cargo"
source = { git = "https://github.com/petitstrawberry/boxcraft" }
package = "boxcraft"
bin = "boxcraft"
to = "/system/scarlet/bin/boxcraft"
```

Build or run an existing Scarlet project with the SDK commands:

```bash
cargo scarlet image --project projects/riscv64-limine-full
cargo scarlet run --project projects/riscv64-limine-full --release

cargo scarlet image --project projects/aarch64-limine-full
cargo scarlet run --project projects/aarch64-limine-full --release
```

After the desktop starts, launch `/system/scarlet/bin/boxcraft` from a terminal
or launcher integration.

The CoachZ/SC7180 image build also rebuilds a sibling Boxcraft checkout against
the same in-tree Adreno backend used by SWS. This keeps its SGFX command stream
and the kernel validator in lockstep while the hardware driver is developed.

## License

Boxcraft is licensed under the [MIT License](LICENSE).
