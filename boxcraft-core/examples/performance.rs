//! Repeatable CPU workload: cargo run --release -p boxcraft-core --example performance

use std::hint::black_box;
use std::time::Instant;

use boxcraft_core::{Block, CHUNK_SIZE, IVec3, VisibleSpace, World, mesh_chunk, mesh_chunk_lod};

fn main() {
    for seed in [7, 11, 42] {
        let started = Instant::now();
        let world = World::generate(seed);
        let generation = started.elapsed();
        let spawn = world.spawn_point();
        let viewer = IVec3::new(spawn.x as i32, spawn.y as i32 + 1, spawn.z as i32);
        let center = (viewer.x / CHUNK_SIZE, viewer.z / CHUNK_SIZE);
        let started = Instant::now();
        let visible = VisibleSpace::from_world(&world, viewer);
        let visibility = started.elapsed();
        let started = Instant::now();
        let mut near_triangles = 0;
        for dz in -2..=2 {
            for dx in -2..=2 {
                near_triangles += black_box(mesh_chunk(&world, center.0 + dx, center.1 + dz))
                    .indices
                    .len()
                    / 3;
            }
        }
        let near = started.elapsed();
        let started = Instant::now();
        let mut far_triangles = 0;
        for dz in -5_i32..=5 {
            for dx in -5_i32..=5 {
                if dx.abs().max(dz.abs()) <= 2 {
                    continue;
                }
                far_triangles += black_box(mesh_chunk_lod(
                    &world,
                    center.0 + dx,
                    center.1 + dz,
                    &visible,
                ))
                .indices
                .len()
                    / 3;
            }
        }
        let far = started.elapsed();
        let started = Instant::now();
        for block in [Block::Stone, Block::Air, Block::Torch, Block::Air] {
            let mut snapshot = world.clone();
            snapshot.set(viewer, block);
            black_box(snapshot.recompute_light_after_edit(viewer));
            black_box(mesh_chunk(&snapshot, center.0, center.1));
        }
        let edits = started.elapsed();
        println!(
            "seed={seed} generation={:.2}ms visibility={:.2}ms near={:.2}ms far={:.2}ms edits={:.2}ms triangles={near_triangles}/{far_triangles}",
            generation.as_secs_f64() * 1_000.0,
            visibility.as_secs_f64() * 1_000.0,
            near.as_secs_f64() * 1_000.0,
            far.as_secs_f64() * 1_000.0,
            edits.as_secs_f64() * 1_000.0,
        );
    }
}
