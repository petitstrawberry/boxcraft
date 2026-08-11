# Boxcraft

Boxcraft is a small first-person voxel sandbox for Scarlet OS. It is written in
Rust and uses ScarletUI for all window chrome and HUD rendering, with an SGFX
canvas for the depth-tested world.

Boxcraft is Scarlet-only: host builds intentionally retain a small command-line
stub instead of a desktop frontend.

## Workspace layout

The workspace follows the same split used by Vellum:

- `boxcraft-core` is the dependency-free game domain: deterministic terrain,
  visible-face meshing, raycasts, player physics, and camera matrices.
- `boxcraft` is the ScarletUI application crate. It translates the core mesh
  into a textured SGFX triangle list and owns window/input integration.

## Terrain rendering

The frontend builds one small procedural RGBA8 pixel-art atlas at startup. Core
mesh vertices provide a material, normalized atlas UV, sky-light, and
ambient-occlusion value; the frontend uses that metadata to sample the grass,
dirt, stone, wood, leaves, sand, and water tiles. A slowly moving directional
sun is combined with the baked occlusion and sky ambient light before each
retained mesh update, so the same textured terrain shifts naturally from sunrise
through night without changing the camera, depth-tested frame, or input path.

External framework dependencies are Git dependencies. In particular, ScarletUI
is fetched from `https://github.com/petitstrawberry/scarlet-ui`; the manifest
intentionally has no local ScarletUI path or `[patch]` override. The local
`boxcraft-core` workspace dependency remains a path dependency by design.

## Development environment

The included flake supplies the Scarlet Rust toolchain, `cargo-scarlet`, and
the image tooling. With Nix and direnv installed:

```bash
direnv allow
```

Or enter it directly:

```bash
nix develop
```

The host-safe checks exercise the pure core and the host stub:

```bash
cargo test -p boxcraft-core
cargo check -p boxcraft
```

Build the Scarlet application for either supported userspace target:

```bash
cargo build -p boxcraft --target riscv64gc-unknown-scarlet
cargo build -p boxcraft --target aarch64-unknown-scarlet
```

## Add Boxcraft to a Scarlet image

Scarlet images are composed from cargo layers. Add the following layer to the
desktop bundle (or another image layer list) in your Scarlet checkout:

```toml
[[layers]]
kind = "cargo"
source = { git = "https://github.com/petitstrawberry/boxcraft" }
package = "boxcraft"
bin = "boxcraft"
to = "/system/scarlet/bin/boxcraft"
```

Then build or run one of Scarlet's existing projects with the SDK commands:

```bash
cargo scarlet image --project projects/riscv64-limine-full
cargo scarlet run --project projects/riscv64-limine-full --release

cargo scarlet image --project projects/aarch64-limine-full
cargo scarlet run --project projects/aarch64-limine-full --release
```

After the desktop starts, launch `/system/scarlet/bin/boxcraft` from a terminal
or launcher integration.

## Controls

- Click the terrain or use **Capture pointer** to capture the pointer; `Esc`
  releases it.
- `W`, `A`, `S`, `D` move; `Space` jumps; relative mouse movement looks.
- While captured, left click breaks a block and right click places the selected
  block.
- `1`–`6` select Grass, Dirt, Stone, Wood, Leaves, or Sand. `7` selects Air
  (useful for inspecting the empty slot, but it cannot be placed).
- `R` regenerates and resets the world; `F11` toggles fullscreen.

## ScarletUI requirements

Boxcraft requires a ScarletUI revision with the SGFX depth-tested canvas,
stable dynamic mesh handles and revisions, retained RGBA8 textures with
per-vertex texture coordinates, SWS pointer lock, relative pointer motion, and
mouse-button view modifiers. Terrain edits publish a new revision of one stable
mesh handle; normal frames reuse that mesh and only update the camera transform,
with a lightweight lighting revision as the day cycle advances.

## License

Boxcraft is licensed under the MIT License.
