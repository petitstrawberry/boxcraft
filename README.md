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
| Scarlet OS | ScarletUI with SWS | SGFX with the runtime-selected VirGL or Adreno A6xx backend |

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
