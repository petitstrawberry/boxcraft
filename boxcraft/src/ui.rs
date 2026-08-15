//! ScarletUI frontend for Boxcraft.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use boxcraft_core::{Block, CHUNK_SIZE, Game, Mat4, PlayerInput, Vec3, mesh_chunk, mesh_chunk_lod};
use scarlet_ui::prelude::*;
use scarlet_ui::{
    ApplicationRunExt, ComponentElement, HeaderBar, KeyCode, KeyEvent, MouseButton, PlatformWindow,
    SgfxCanvas, SgfxCanvasDraw, SgfxCanvasFrame, SgfxCanvasHandle, SgfxCanvasVertex, SgfxMesh,
    SgfxMeshHandle, SgfxTexture, hstack, vstack, zstack,
};

const APP_ID: &str = "org.scarlet-os.boxcraft";
const WINDOW_WIDTH: f32 = 1100.0;
const WINDOW_HEIGHT: f32 = 760.0;
const REFERENCE_ASPECT: f32 = 16.0 / 9.0;
const CAMERA_FOV: f32 = 70.0_f32.to_radians();
const REACH: f32 = 6.0;
const LOOK_SENSITIVITY: f32 = 0.003;
const WORLD_SEED: u64 = 0xB0CA_FE00_2026_0001;
const ATLAS_TILE_SIZE: u32 = 32;
const ATLAS_COLUMNS: u32 = 4;
const ATLAS_ROWS: u32 = 4;
const DAY_LENGTH_SECONDS: f32 = 150.0;
const SUNLIGHT_UPDATES_PER_SECOND: f32 = 4.0;
/// Terrain chunks built per idle tick while streaming the world in.
const CHUNKS_PER_FRAME: usize = 3;
/// Default and inclusive bounds for the configurable render distance.
const DEFAULT_RENDER_DISTANCE: i32 = 3;
const MIN_RENDER_DISTANCE: i32 = 1;
const MAX_RENDER_DISTANCE: i32 = 6;
/// Chunks within this Chebyshev radius keep individual meshes for edits. Each
/// chunk now has one textured draw; illumination is carried by its vertices.
const NEAR_CHUNK_RADIUS: i32 = 2;

/// Run the Boxcraft application.
///
/// # Returns
///
/// Success after the application exits, or a ScarletUI error.
pub fn run() -> scarlet_ui::Result<()> {
    let mut app = BoxcraftApp::new();
    app.run()
}

#[derive(Clone, Copy, Default)]
struct PressedKeys {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    jump: bool,
}

impl PressedKeys {
    fn player_input(self) -> PlayerInput {
        PlayerInput {
            forward: self.forward,
            backward: self.backward,
            left: self.left,
            right: self.right,
            jump: self.jump,
        }
    }
}

struct Runtime {
    last_idle: Instant,
    fps_sample_start: Instant,
    frames_since_sample: u32,
    day_seconds: f32,
    sunlight_step: u64,
    build_queue: VecDeque<(i32, i32)>,
    queued_chunks: HashSet<(i32, i32)>,
    player_chunk: (i32, i32),
    far_dirty: bool,
    far_chunks: Vec<(i32, i32)>,
}

impl Runtime {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            last_idle: now,
            fps_sample_start: now,
            frames_since_sample: 0,
            day_seconds: 0.0,
            sunlight_step: 0,
            build_queue: VecDeque::new(),
            queued_chunks: HashSet::new(),
            player_chunk: (i32::MAX, i32::MAX),
            far_dirty: false,
            far_chunks: Vec::new(),
        }
    }
}

#[derive(Clone)]
struct BoxcraftApp {
    game: State<Arc<Mutex<Game>>>,
    keys: State<PressedKeys>,
    seed: State<u64>,
    pointer_lock_desired: State<bool>,
    pointer_lock_applied: State<bool>,
    pointer_lock_pending: State<bool>,
    fullscreen_desired: State<bool>,
    fullscreen_applied: State<bool>,
    fullscreen_pending: State<bool>,
    decorations_hidden: State<bool>,
    settings_visible: State<bool>,
    render_distance: State<i32>,
    canvas_frame: State<Arc<SgfxCanvasFrame>>,
    near_core_meshes: State<Arc<HashMap<(i32, i32), Arc<boxcraft_core::Mesh>>>>,
    near_meshes: State<Arc<HashMap<(i32, i32), Arc<SgfxMesh>>>>,
    far_core_meshes: State<Arc<HashMap<(i32, i32), Arc<boxcraft_core::Mesh>>>>,
    far_meshes: State<Arc<HashMap<(i32, i32), Arc<SgfxMesh>>>>,
    mesh_revision: State<u64>,
    frame_revision: State<u64>,
    sun_phase: State<f32>,
    fps: State<u32>,
    fps_text: State<String>,
    position: State<String>,
    selected_block: State<String>,
    status: State<String>,
    canvas_handle: SgfxCanvasHandle,
    chunk_handles: Arc<Mutex<HashMap<(i32, i32), SgfxMeshHandle>>>,
    handle_pool: Arc<Mutex<Vec<SgfxMeshHandle>>>,
    block_atlas: Arc<SgfxTexture>,
    runtime: Arc<Mutex<Runtime>>,
}

impl BoxcraftApp {
    fn new() -> Self {
        let initial_frame = Arc::new(
            SgfxCanvasFrame::new(0, Color::rgb(0.045, 0.075, 0.13))
                .depth_tested()
                .reference_aspect(REFERENCE_ASPECT),
        );
        let app = Self {
            game: State::new(
                StateId::new(1),
                Arc::new(Mutex::new(Game::generated(WORLD_SEED))),
            ),
            keys: State::new(StateId::new(2), PressedKeys::default()),
            seed: State::new(StateId::new(3), WORLD_SEED),
            pointer_lock_desired: State::new(StateId::new(4), false),
            pointer_lock_applied: State::new(StateId::new(5), false),
            pointer_lock_pending: State::new(StateId::new(6), false),
            fullscreen_desired: State::new(StateId::new(7), false),
            fullscreen_applied: State::new(StateId::new(8), false),
            fullscreen_pending: State::new(StateId::new(9), false),
            decorations_hidden: State::new(StateId::new(21), false),
            settings_visible: State::new(StateId::new(22), false),
            render_distance: State::new(StateId::new(23), DEFAULT_RENDER_DISTANCE),
            canvas_frame: State::new(StateId::new(10), initial_frame),
            near_core_meshes: State::new(
                StateId::new(25),
                Arc::new(HashMap::<(i32, i32), Arc<boxcraft_core::Mesh>>::new()),
            ),
            near_meshes: State::new(
                StateId::new(11),
                Arc::new(HashMap::<(i32, i32), Arc<SgfxMesh>>::new()),
            ),
            far_core_meshes: State::new(
                StateId::new(26),
                Arc::new(HashMap::<(i32, i32), Arc<boxcraft_core::Mesh>>::new()),
            ),
            far_meshes: State::new(
                StateId::new(24),
                Arc::new(HashMap::<(i32, i32), Arc<SgfxMesh>>::new()),
            ),
            mesh_revision: State::new(StateId::new(12), 0),
            frame_revision: State::new(StateId::new(13), 0),
            sun_phase: State::new(StateId::new(20), 0.0),
            fps: State::new(StateId::new(14), 0),
            fps_text: State::new(StateId::new(15), String::from("FPS: 0")),
            position: State::new(StateId::new(16), String::from("Position: loading")),
            selected_block: State::new(StateId::new(17), String::from("1: Grass")),
            status: State::new(
                StateId::new(18),
                String::from("Click the terrain to capture the pointer"),
            ),
            canvas_handle: SgfxCanvasHandle::new(),
            chunk_handles: Arc::new(Mutex::new(HashMap::new())),
            handle_pool: Arc::new(Mutex::new(Vec::new())),
            block_atlas: block_texture_atlas(),
            runtime: Arc::new(Mutex::new(Runtime::new())),
        };
        app.refresh_chunk_set();
        app.update_hud();
        app.refresh_frame();
        app
    }

    fn with_game<R>(&self, operation: impl FnOnce(&mut Game) -> R) -> R {
        let game = self.game.get();
        let mut guard = match game.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        operation(&mut guard)
    }

    fn clear_pressed_keys(&self) {
        self.keys.set(PressedKeys::default());
    }

    fn request_pointer_lock(&self) {
        self.pointer_lock_desired.set(true);
        self.status.set(String::from("Requesting pointer capture…"));
    }

    fn release_pointer_lock(&self) {
        self.pointer_lock_desired.set(false);
        self.clear_pressed_keys();
        self.status
            .set(String::from("Pointer released — UI controls are available"));
    }

    fn toggle_fullscreen(&self) {
        self.fullscreen_desired.set(!self.fullscreen_desired.get());
    }

    fn change_render_distance(&self, delta: i32) {
        self.render_distance.update(|distance| {
            *distance = (*distance + delta).clamp(MIN_RENDER_DISTANCE, MAX_RENDER_DISTANCE)
        });
        self.refresh_chunk_set();
        self.status.set(format!(
            "Render distance: {} chunks",
            self.render_distance.get()
        ));
    }

    fn toggle_settings(&self) {
        self.settings_visible.update(|visible| *visible = !*visible);
    }

    fn reset_world(&self) {
        self.seed.update(|seed| *seed = seed.wrapping_add(1));
        let seed = self.seed.get();
        self.game.set(Arc::new(Mutex::new(Game::generated(seed))));
        self.clear_pressed_keys();
        let retired: Vec<SgfxMeshHandle> = {
            let mut handles = self
                .chunk_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            handles.drain().map(|(_, handles)| handles).collect()
        };
        self.handle_pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .extend(retired);
        self.near_meshes
            .update(|chunks| *chunks = Arc::new(HashMap::<(i32, i32), Arc<SgfxMesh>>::new()));
        self.near_core_meshes.set(Arc::new(
            HashMap::<(i32, i32), Arc<boxcraft_core::Mesh>>::new(),
        ));
        self.far_core_meshes.set(Arc::new(
            HashMap::<(i32, i32), Arc<boxcraft_core::Mesh>>::new(),
        ));
        self.far_meshes
            .set(Arc::new(HashMap::<(i32, i32), Arc<SgfxMesh>>::new()));
        {
            let mut runtime = self
                .runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            runtime.build_queue.clear();
            runtime.queued_chunks.clear();
            runtime.player_chunk = (i32::MAX, i32::MAX);
            runtime.far_dirty = false;
            runtime.far_chunks.clear();
        }
        self.refresh_chunk_set();
        self.update_hud();
        self.status
            .set(String::from("Generated a fresh Boxcraft world"));
    }

    fn select_block(&self, slot: usize, block: Block) {
        self.with_game(|game| game.player.select_block(slot, block));
        self.selected_block
            .set(format!("{}: {}", slot + 1, block_name(block)));
    }

    fn handle_key(&self, event: KeyEvent) -> bool {
        match event {
            KeyEvent::Pressed { keycode, .. } => match keycode {
                KeyCode::Escape => {
                    if self.settings_visible.get() {
                        self.settings_visible.set(false);
                    } else {
                        self.release_pointer_lock();
                    }
                    true
                }
                KeyCode::F(11) => {
                    self.toggle_fullscreen();
                    true
                }
                KeyCode::Char('r') | KeyCode::Char('R') => {
                    self.reset_world();
                    true
                }
                KeyCode::Char('1') => {
                    self.select_block(0, Block::Grass);
                    true
                }
                KeyCode::Char('2') => {
                    self.select_block(1, Block::Dirt);
                    true
                }
                KeyCode::Char('3') => {
                    self.select_block(2, Block::Stone);
                    true
                }
                KeyCode::Char('4') => {
                    self.select_block(3, Block::Wood);
                    true
                }
                KeyCode::Char('5') => {
                    self.select_block(4, Block::Leaves);
                    true
                }
                KeyCode::Char('6') => {
                    self.select_block(5, Block::Sand);
                    true
                }
                KeyCode::Char('7') => {
                    self.select_block(6, Block::Snow);
                    true
                }
                KeyCode::Char('8') => {
                    self.select_block(7, Block::Air);
                    true
                }
                KeyCode::Char('9') => {
                    self.select_block(8, Block::Torch);
                    true
                }
                KeyCode::Char('o') | KeyCode::Char('O') => {
                    self.settings_visible.update(|visible| *visible = !*visible);
                    true
                }
                KeyCode::Char('-') | KeyCode::Char('_') => {
                    self.change_render_distance(-1);
                    true
                }
                KeyCode::Char('=') | KeyCode::Char('+') => {
                    self.change_render_distance(1);
                    true
                }
                keycode if self.pointer_lock_applied.get() => self.set_movement_key(keycode, true),
                _ => false,
            },
            KeyEvent::Released { keycode, .. } if self.pointer_lock_applied.get() => {
                self.set_movement_key(keycode, false)
            }
            KeyEvent::Released { .. } => false,
            KeyEvent::Char { .. } => false,
        }
    }

    fn set_movement_key(&self, keycode: KeyCode, pressed: bool) -> bool {
        let matched = matches!(
            keycode,
            KeyCode::Char('w' | 'W' | 'a' | 'A' | 's' | 'S' | 'd' | 'D') | KeyCode::Space
        );
        if !matched {
            return false;
        }
        self.keys.update(|keys| match keycode {
            KeyCode::Char('w' | 'W') => keys.forward = pressed,
            KeyCode::Char('s' | 'S') => keys.backward = pressed,
            KeyCode::Char('a' | 'A') => keys.left = pressed,
            KeyCode::Char('d' | 'D') => keys.right = pressed,
            KeyCode::Space => keys.jump = pressed,
            _ => {}
        });
        true
    }

    fn handle_mouse_delta(&self, dx: i32, dy: i32) {
        if !self.pointer_lock_applied.get() {
            return;
        }
        self.with_game(|game| {
            game.player.look(
                dx as f32 * LOOK_SENSITIVITY,
                -(dy as f32) * LOOK_SENSITIVITY,
            );
        });
    }

    fn handle_mouse_button(&self, button: MouseButton, pressed: bool) -> bool {
        if !pressed {
            return false;
        }
        if !self.pointer_lock_applied.get() {
            self.request_pointer_lock();
            return true;
        }

        let mut edited = None;
        let changed = self.with_game(|game| match button {
            MouseButton::Left => {
                edited = game
                    .world
                    .raycast(
                        game.player.camera().position,
                        game.player.camera().forward(),
                        REACH,
                    )
                    .map(|hit| hit.position);
                game.player.break_block(&mut game.world, REACH).is_some()
            }
            MouseButton::Right => {
                edited = game
                    .player
                    .place_block(&mut game.world, REACH)
                    .map(|position| position);
                edited.is_some()
            }
            MouseButton::Middle => false,
        });
        if changed {
            if let Some(position) = edited {
                self.rebuild_edited_chunks(position);
            }
        }
        true
    }

    /// Rebuild the chunks touched by a block edit, plus neighbours when the
    /// edit sits on a chunk border and affects their ambient occlusion.
    fn rebuild_edited_chunks(&self, position: boxcraft_core::IVec3) {
        let chunk = (
            (position.x as i32).div_euclid(CHUNK_SIZE),
            (position.z as i32).div_euclid(CHUNK_SIZE),
        );
        let local_x = position.x.rem_euclid(CHUNK_SIZE);
        let local_z = position.z.rem_euclid(CHUNK_SIZE);
        let mut rebuilt = false;
        // Ambient occlusion and smooth light reach one block across the
        // border, so only rebuild a neighbour when the edit is edge-adjacent.
        let mut offsets = vec![(0, 0)];
        if local_x == 0 {
            offsets.extend([(-1, 0), (-1, -1), (-1, 1)]);
        }
        if local_x == CHUNK_SIZE - 1 {
            offsets.extend([(1, 0), (1, -1), (1, 1)]);
        }
        if local_z == 0 {
            offsets.extend([(0, -1)]);
        }
        if local_z == CHUNK_SIZE - 1 {
            offsets.extend([(0, 1)]);
        }
        for offset in offsets {
            rebuilt |= self.build_near_chunk(chunk.0 + offset.0, chunk.1 + offset.1);
        }
        if rebuilt {
            // An edit can change visibility, AO, and propagated light across
            // the far-ring boundary. Invalidate the cached far geometry so it
            // is rebuilt from the new world state once the near queue settles.
            self.far_core_meshes.set(Arc::new(
                HashMap::<(i32, i32), Arc<boxcraft_core::Mesh>>::new(),
            ));
            self.far_meshes
                .set(Arc::new(HashMap::<(i32, i32), Arc<SgfxMesh>>::new()));
            self.runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .far_dirty = true;
            self.refresh_frame();
        }
    }

    /// Reconcile the set of resident chunks with the player's render distance.
    ///
    /// Chunks within the near radius keep individual meshes for cheap block
    /// edits; the remaining ring keeps one retained mesh per chunk so the
    /// renderer can cull the part of the ring behind the camera. Mesh handles
    /// come from a recycled pool because the SGFX renderer's mesh cache never
    /// evicts retired handles.
    fn refresh_chunk_set(&self) -> bool {
        let mut changed = false;
        let player_chunk = self.with_game(|game| {
            let position = game.player.position;
            (
                (position.x as i32).div_euclid(CHUNK_SIZE),
                (position.z as i32).div_euclid(CHUNK_SIZE),
            )
        });
        let distance = self.render_distance.get();
        let near_radius = distance.min(NEAR_CHUNK_RADIUS);
        let desired: HashSet<(i32, i32)> = (-distance..=distance)
            .flat_map(|dz| (-distance..=distance).map(move |dx| (dx, dz)))
            .map(|(dx, dz)| (player_chunk.0 + dx, player_chunk.1 + dz))
            .collect();
        let near: HashSet<(i32, i32)> = (-near_radius..=near_radius)
            .flat_map(|dz| (-near_radius..=near_radius).map(move |dx| (dx, dz)))
            .map(|(dx, dz)| (player_chunk.0 + dx, player_chunk.1 + dz))
            .collect();
        let mut far: Vec<(i32, i32)> = desired
            .iter()
            .copied()
            .filter(|chunk| !near.contains(chunk))
            .collect();
        far.sort_unstable();

        let removed: Vec<(i32, i32)> = self
            .near_meshes
            .get()
            .keys()
            .copied()
            .filter(|chunk| !near.contains(chunk))
            .collect();
        if !removed.is_empty() {
            changed = true;
            self.near_core_meshes.update(|chunks| {
                let mut next = (**chunks).clone();
                for chunk in &removed {
                    next.remove(chunk);
                }
                *chunks = Arc::new(next);
            });
            self.near_meshes.update(|chunks| {
                let mut next = (**chunks).clone();
                for chunk in &removed {
                    next.remove(chunk);
                }
                *chunks = Arc::new(next);
            });
            // Return retired handles to the pool instead of dropping
            // them: the renderer caches by handle and never evicts, so fresh
            // handles for every streamed chunk would grow the cache unbounded.
            let retired: Vec<SgfxMeshHandle> = {
                let mut handles = self
                    .chunk_handles
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                removed
                    .iter()
                    .filter_map(|chunk| handles.remove(chunk))
                    .collect()
            };
            self.handle_pool
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .extend(retired);
        }

        // A chunk can move from the far ring into the near ring while the
        // rebuild is still deferred. Remove its old far draw immediately so
        // the same handle is never submitted with two revisions in one frame.
        let far_set: HashSet<(i32, i32)> = far.iter().copied().collect();
        let stale_far: Vec<(i32, i32)> = self
            .far_meshes
            .get()
            .keys()
            .copied()
            .filter(|chunk| !far_set.contains(chunk))
            .collect();
        if !stale_far.is_empty() {
            changed = true;
            self.far_core_meshes.update(|meshes| {
                let mut next = (**meshes).clone();
                for chunk in &stale_far {
                    next.remove(chunk);
                }
                *meshes = Arc::new(next);
            });
            self.far_meshes.update(|meshes| {
                let mut next = (**meshes).clone();
                for chunk in &stale_far {
                    next.remove(chunk);
                }
                *meshes = Arc::new(next);
            });
        }

        let mut runtime = self
            .runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let allowed: HashSet<(i32, i32)> = desired.clone();
        runtime
            .queued_chunks
            .retain(|chunk| allowed.contains(chunk));
        {
            let queued = runtime.queued_chunks.clone();
            runtime.build_queue.retain(|chunk| queued.contains(chunk));
        }
        runtime.player_chunk = player_chunk;
        if runtime.far_chunks != far {
            runtime.far_chunks = far;
            runtime.far_dirty = true;
            changed = true;
        }
        let mut missing: Vec<(i32, i32)> = desired
            .into_iter()
            .filter(|chunk| {
                near.contains(chunk)
                    && !self.near_meshes.get().contains_key(chunk)
                    && !runtime.queued_chunks.contains(chunk)
            })
            .collect();
        missing.sort_by_key(|chunk| {
            (chunk.0 - player_chunk.0)
                .abs()
                .max((chunk.1 - player_chunk.1).abs())
        });
        for chunk in missing {
            runtime.queued_chunks.insert(chunk);
            runtime.build_queue.push_back(chunk);
        }
        changed
    }

    /// Stream a few queued chunks into GPU meshes; returns whether any moved.
    fn process_build_queue(&self, budget: usize) -> bool {
        let mut built = false;
        for _ in 0..budget {
            let next = {
                let mut runtime = self
                    .runtime
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                loop {
                    match runtime.build_queue.pop_front() {
                        Some(chunk) => {
                            runtime.queued_chunks.remove(&chunk);
                            break Some(chunk);
                        }
                        None => break None,
                    }
                }
            };
            let Some(chunk) = next else { break };
            built |= self.build_near_chunk(chunk.0, chunk.1);
        }
        built
    }

    /// Rebuild the retained far-ring meshes once streaming has settled.
    ///
    /// Deferred until the streaming queue settles so crossing a chunk border
    /// rebuilds the ring once instead of once per incoming chunk.
    fn rebuild_far_meshes_if_settled(&self) -> bool {
        {
            let mut runtime = self
                .runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !runtime.far_dirty || !runtime.build_queue.is_empty() {
                return false;
            }
            runtime.far_dirty = false;
        }
        let far_chunks: Vec<(i32, i32)> = self
            .runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .far_chunks
            .clone();
        let cached_core_meshes = self.far_core_meshes.get();
        let mut core_meshes = HashMap::with_capacity(far_chunks.len());
        let mut meshes = HashMap::with_capacity(far_chunks.len());
        self.mesh_revision
            .update(|revision| *revision = revision.wrapping_add(1));
        let revision = self.mesh_revision.get();
        for (chunk_x, chunk_z) in far_chunks.iter().copied() {
            let key = (chunk_x, chunk_z);
            let core_mesh = cached_core_meshes.get(&key).cloned().unwrap_or_else(|| {
                Arc::new(self.with_game(|game| mesh_chunk_lod(&game.world, chunk_x, chunk_z)))
            });
            core_meshes.insert(key, Arc::clone(&core_mesh));
            let mut vertices = Vec::new();
            append_lit_vertices(&core_mesh, self.sun_phase.get(), &mut vertices);
            if vertices.is_empty() {
                continue;
            }
            let handle = {
                let mut active = self
                    .chunk_handles
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Some(handle) = active.get(&key) {
                    *handle
                } else {
                    let handle = self
                        .handle_pool
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop()
                        .unwrap_or_else(SgfxMeshHandle::new);
                    active.insert(key, handle);
                    handle
                }
            };
            meshes.insert(key, SgfxMesh::with_handle(handle, revision, vertices));
        }
        self.far_core_meshes.set(Arc::new(core_meshes));

        // Retire handles for chunks that are no longer resident, but retain
        // handles shared by the near ring so a far-to-near transition can
        // update the existing renderer buffer safely.
        let near_chunks = self.near_meshes.get();
        let stale_handles: Vec<(i32, i32)> = {
            let handles = self
                .chunk_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            handles
                .keys()
                .copied()
                .filter(|chunk| !near_chunks.contains_key(chunk) && !meshes.contains_key(chunk))
                .collect()
        };
        if !stale_handles.is_empty() {
            let retired: Vec<SgfxMeshHandle> = {
                let mut handles = self
                    .chunk_handles
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                stale_handles
                    .iter()
                    .filter_map(|chunk| handles.remove(chunk))
                    .collect()
            };
            self.handle_pool
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .extend(retired);
        }
        self.far_meshes.set(Arc::new(meshes));
        true
    }

    /// Build one chunk's retained SGFX meshes from the live world snapshot.
    ///
    /// Terrain supplies local UVs, material identity, and baked sky/block
    /// light plus ambient occlusion. The frontend maps those into the pixel
    /// atlas and writes one interpolated lighting color per vertex.
    fn build_near_chunk(&self, chunk_x: i32, chunk_z: i32) -> bool {
        let key = (chunk_x, chunk_z);
        let core_mesh = Arc::new(self.with_game(|game| mesh_chunk(&game.world, chunk_x, chunk_z)));
        self.near_core_meshes.update(|chunks| {
            let mut next = (**chunks).clone();
            next.insert(key, Arc::clone(&core_mesh));
            *chunks = Arc::new(next);
        });
        self.update_near_mesh_from_core(key, &core_mesh)
    }

    fn update_near_mesh_from_core(&self, key: (i32, i32), core_mesh: &boxcraft_core::Mesh) -> bool {
        let mut vertices = Vec::new();
        append_lit_vertices(core_mesh, self.sun_phase.get(), &mut vertices);
        self.mesh_revision
            .update(|revision| *revision = revision.wrapping_add(1));
        let revision = self.mesh_revision.get();
        let handle = {
            let mut active = self
                .chunk_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(handle) = active.get(&key) {
                *handle
            } else {
                let handle = self
                    .handle_pool
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .pop()
                    .unwrap_or_else(SgfxMeshHandle::new);
                active.insert(key, handle);
                handle
            }
        };
        let mesh = SgfxMesh::with_handle(handle, revision, vertices);
        self.near_meshes.update(|chunks| {
            let mut next = (**chunks).clone();
            next.insert(key, mesh);
            *chunks = Arc::new(next);
        });
        true
    }

    /// Re-bake the resident vertex colors when the moving sun crosses a
    /// lighting step. The expensive voxel propagation stays unchanged; only
    /// the current sky-channel composition is rebuilt.
    fn rebuild_lighting_meshes(&self) -> bool {
        let mut rebuilt = false;
        let core_meshes = self.near_core_meshes.get();
        for (key, core_mesh) in core_meshes.iter() {
            rebuilt |= self.update_near_mesh_from_core(*key, core_mesh);
        }
        self.runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .far_dirty = true;
        rebuilt | self.rebuild_far_meshes_if_settled()
    }

    fn refresh_frame(&self) {
        let (camera, transform) = self.with_game(|game| {
            let camera = game.player.camera();
            let transform = Mat4::perspective_rh_gl(CAMERA_FOV, REFERENCE_ASPECT, 0.05, 128.0)
                .mul_mat4(camera.view_matrix())
                .columns;
            (camera, transform)
        });
        self.frame_revision
            .update(|revision| *revision = revision.wrapping_add(1));
        let mut frame =
            SgfxCanvasFrame::new(self.frame_revision.get(), Color::rgb(0.045, 0.075, 0.13))
                .depth_tested()
                // The SGFX renderer corrects this reference perspective as its canvas resizes.
                .reference_aspect(REFERENCE_ASPECT);
        let far_meshes = self.far_meshes.get();
        for (_, mesh) in far_meshes.iter().filter(|(chunk, mesh)| {
            mesh.triangle_count() > 0
                && chunk_is_visible(camera.position, camera.forward(), **chunk)
        }) {
            frame = frame.draw(
                SgfxCanvasDraw::new(Arc::clone(mesh), transform)
                    .tint(Color::WHITE)
                    .texture(Arc::clone(&self.block_atlas)),
            );
        }
        let near_meshes = self.near_meshes.get();
        for (_, meshes) in near_meshes.iter().filter(|(chunk, mesh)| {
            mesh.triangle_count() > 0
                && chunk_is_visible(camera.position, camera.forward(), **chunk)
        }) {
            frame = frame.draw(
                SgfxCanvasDraw::new(Arc::clone(meshes), transform)
                    .tint(Color::WHITE)
                    .texture(Arc::clone(&self.block_atlas)),
            );
        }
        self.canvas_frame.set(Arc::new(frame));
    }

    fn update_hud(&self) {
        let (position, block, slot) = self.with_game(|game| {
            (
                game.player.position,
                game.player.selected_block,
                game.player.selected_slot,
            )
        });
        self.position.set(format!(
            "Position: {:.1}, {:.1}, {:.1}",
            position.x, position.y, position.z
        ));
        self.selected_block
            .set(format!("{}: {}", slot + 1, block_name(block)));
    }

    fn content(&self) -> impl View + Clone + use<> {
        let capture = self.clone();
        let reset = self.clone();
        let fullscreen = self.clone();
        let canvas_input = self.clone();
        let key_input = self.clone();
        let pointer_delta = self.clone();
        let settings_toggle = self.clone();
        let distance_down = self.clone();
        let distance_up = self.clone();
        let settings_close = self.clone();
        let pointer_locked = self.pointer_lock_applied.get();
        let fullscreen_desired = self.fullscreen_desired.get();
        let settings_open = self.settings_visible.get();
        let render_distance = self.render_distance.get();

        let controls = scarlet_ui::if_view!(
            !pointer_locked,
            hstack! {
                Button::new("Capture pointer").header_style().on_click(move || capture.request_pointer_lock()),
                Button::new("Reset world").header_style().on_click(move || reset.reset_world()),
                Button::new(if settings_open { "Hide settings" } else { "Settings" })
                    .header_style()
                    .on_click(move || settings_toggle.toggle_settings()),
                Button::new(if fullscreen_desired { "Exit fullscreen" } else { "Fullscreen" })
                    .header_style()
                    .on_click(move || fullscreen.toggle_fullscreen()),
            }
            .spacing(6.0),
            Spacer::new().frame_width(0.0)
        );
        let header = HeaderBar::new(
            hstack! {
                Text::new("Boxcraft").font_size(18.0),
                Spacer::new(),
                Text::from_state(self.position.clone()).font_size(12.0),
                Text::from_state(self.fps_text.clone()).font_size(12.0),
                controls,
            }
            .spacing(12.0)
            .padding(10.0),
        );

        let canvas = SgfxCanvas::from_state(
            self.canvas_handle,
            f32::INFINITY,
            f32::INFINITY,
            self.canvas_frame.clone(),
        )
        .placeholder(Color::rgb(0.045, 0.075, 0.13))
        .frame(f32::INFINITY, f32::INFINITY);
        let game_area = zstack! {
            canvas,
            Text::new("+").font_size(28.0).color(Color::rgb(0.95, 0.95, 0.98)),
            scarlet_ui::if_view!(
                settings_open,
                hstack! {
                    Text::new(format!("Render distance: {render_distance} chunks")).font_size(13.0),
                    Button::new("−").on_click(move || distance_down.change_render_distance(-1)),
                    Button::new("+").on_click(move || distance_up.change_render_distance(1)),
                    Button::new("Close").on_click(move || settings_close.toggle_settings()),
                }
                .spacing(10.0)
                .padding(12.0)
                .background(Color::rgba(0.03, 0.04, 0.07, 0.88)),
                Spacer::new()
            ),
        }
        .alignment(Alignment::Center)
        .frame(f32::INFINITY, f32::INFINITY)
        .on_mouse_delta(move |dx, dy| pointer_delta.handle_mouse_delta(dx, dy))
        .on_mouse_button(move |button, pressed| canvas_input.handle_mouse_button(button, pressed));

        let pointer_help = if pointer_locked {
            String::from("Esc to release · Left break · Right place · WASD + Space to move")
        } else {
            String::from("Click to capture pointer · O settings · +/- render distance")
        };
        vstack! {
            header,
            game_area,
            hstack! {
                Text::from_state(self.selected_block.clone()).font_size(13.0),
                Spacer::new(),
                Text::new(pointer_help).font_size(12.0),
                Spacer::new(),
                Text::from_state(self.status.clone()).font_size(12.0),
            }
            .spacing(10.0)
            .padding(10.0)
            .background(Color::rgb(0.03, 0.04, 0.07)),
        }
        .frame(f32::INFINITY, f32::INFINITY)
        .on_key(move |event| key_input.handle_key(event))
    }
}

impl View for BoxcraftApp {
    fn create_element(&self) -> Box<dyn Element> {
        Box::new(ComponentElement::new_with_builder(
            self.clone(),
            build_boxcraft_content,
        ))
    }

    fn listenables(&self) -> Vec<&dyn scarlet_ui::Listenable> {
        vec![
            &self.pointer_lock_applied as &dyn scarlet_ui::Listenable,
            &self.fullscreen_desired as &dyn scarlet_ui::Listenable,
            &self.decorations_hidden as &dyn scarlet_ui::Listenable,
            &self.settings_visible as &dyn scarlet_ui::Listenable,
            &self.render_distance as &dyn scarlet_ui::Listenable,
        ]
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl Application for BoxcraftApp {
    fn on_window_sync(&mut self, _ctx: &WindowContext, window: &mut dyn PlatformWindow) {
        let desired_pointer_lock = self.pointer_lock_desired.get();
        if desired_pointer_lock != self.pointer_lock_applied.get()
            && !self.pointer_lock_pending.get()
        {
            self.pointer_lock_pending.set(true);
            if window.set_pointer_lock(desired_pointer_lock).is_err() {
                self.pointer_lock_pending.set(false);
                self.pointer_lock_desired
                    .set(self.pointer_lock_applied.get());
                self.status.set(String::from(
                    "Pointer capture is unavailable for this window",
                ));
            }
        }

        let desired_fullscreen = self.fullscreen_desired.get();
        if desired_fullscreen != self.fullscreen_applied.get() && !self.fullscreen_pending.get() {
            self.fullscreen_pending.set(true);
            if window.set_fullscreen(desired_fullscreen).is_err() {
                self.fullscreen_pending.set(false);
                self.fullscreen_desired.set(self.fullscreen_applied.get());
                self.status
                    .set(String::from("Fullscreen request was not accepted"));
            }
        }
    }

    fn on_window_pointer_lock_changed(&mut self, _ctx: &WindowContext, locked: bool) {
        self.pointer_lock_pending.set(false);
        self.pointer_lock_applied.set(locked);
        self.pointer_lock_desired.set(locked);
        if locked {
            self.status.set(String::from("Pointer captured"));
        } else {
            self.clear_pressed_keys();
            self.status
                .set(String::from("Pointer released — UI controls are available"));
        }
    }

    fn on_window_fullscreen_changed(&mut self, _ctx: &WindowContext, fullscreen: bool) {
        self.fullscreen_pending.set(false);
        self.fullscreen_applied.set(fullscreen);
        self.fullscreen_desired.set(fullscreen);
        // Match vellum: a real fullscreen window also hides its decorations,
        // otherwise the compositor keeps the title bar and it just maximizes.
        self.decorations_hidden.set(fullscreen);
    }

    fn on_window_resize(&mut self, _ctx: &WindowContext, _width: u32, _height: u32) {
        if self.fullscreen_applied.get() == self.fullscreen_desired.get()
            && self.decorations_hidden.get() != self.fullscreen_desired.get()
        {
            self.decorations_hidden.set(self.fullscreen_desired.get());
        }
        self.refresh_frame();
    }

    fn on_focus_changed(&mut self, _window_id: u32, _app_name: &str, _menu_titles: &str) {
        self.clear_pressed_keys();
    }

    fn on_idle(&mut self) {
        let now = Instant::now();
        let (delta_seconds, fps, sun_phase) = {
            let mut runtime = match self.runtime.lock() {
                Ok(runtime) => runtime,
                Err(poisoned) => poisoned.into_inner(),
            };
            let delta_seconds = now
                .saturating_duration_since(runtime.last_idle)
                .as_secs_f32()
                .min(0.1);
            runtime.last_idle = now;
            runtime.day_seconds += delta_seconds;
            let sunlight_step = (runtime.day_seconds * SUNLIGHT_UPDATES_PER_SECOND) as u64;
            let sun_phase = (sunlight_step != runtime.sunlight_step).then(|| {
                runtime.sunlight_step = sunlight_step;
                (runtime.day_seconds / DAY_LENGTH_SECONDS).rem_euclid(1.0)
            });
            runtime.frames_since_sample = runtime.frames_since_sample.saturating_add(1);
            let elapsed = now.saturating_duration_since(runtime.fps_sample_start);
            let fps = (elapsed.as_millis() >= 500).then(|| {
                let fps = (runtime.frames_since_sample as f64 / elapsed.as_secs_f64())
                    .round()
                    .clamp(0.0, u32::MAX as f64) as u32;
                runtime.frames_since_sample = 0;
                runtime.fps_sample_start = now;
                fps
            });
            (delta_seconds, fps, sun_phase)
        };

        let input = self.keys.get().player_input();
        let previous_camera = self.with_game(|game| game.player.camera());
        self.with_game(|game| {
            game.player.step(&game.world, input, delta_seconds);
            if game.player.position.y < -4.0 {
                game.player.position = game.world.spawn_point();
                game.player.velocity = Vec3::zero();
                game.player.grounded = false;
            }
        });
        let camera_changed = self.with_game(|game| game.player.camera()) != previous_camera;
        let mut frame_changed = camera_changed;
        if let Some(sun_phase) = sun_phase {
            self.sun_phase.set(sun_phase);
            frame_changed |= self.rebuild_lighting_meshes();
        }
        // Stream terrain chunks around the player and drop distant ones.
        frame_changed |= self.refresh_chunk_set();
        frame_changed |= self.process_build_queue(CHUNKS_PER_FRAME);
        frame_changed |= self.rebuild_far_meshes_if_settled();
        if let Some(fps) = fps {
            self.fps.set(fps);
            self.fps_text.set(format!("FPS: {fps}"));
            self.update_hud();
        }
        if frame_changed {
            self.refresh_frame();
        }
    }

    fn scenes(&self) -> impl Scene {
        WindowGroup::new(
            "main",
            Window::new("Boxcraft", self.clone())
                .app_id(APP_ID)
                .decorated(!self.decorations_hidden.get())
                .size(Size::new(WINDOW_WIDTH, WINDOW_HEIGHT))
                .min_size(Size::new(720.0, 480.0))
                .resizable(true),
        )
    }
}

fn build_boxcraft_content(app: &BoxcraftApp) -> Box<dyn View> {
    Box::new(app.content())
}

/// Convert one terrain chunk into a single textured mesh with interpolated
/// vertex lighting. The SGFX texture vertex-color pipeline multiplies the
/// atlas sample by this color at every fragment, so no triangle light buckets
/// or block-sized draw tints are needed.
fn append_lit_vertices(
    core_mesh: &boxcraft_core::Mesh,
    sun_phase: f32,
    vertices: &mut Vec<SgfxCanvasVertex>,
) {
    for triangle in core_mesh.indices.chunks_exact(3) {
        let [first, second, third] = triangle else {
            continue;
        };
        let (Some(first), Some(second), Some(third)) = (
            core_mesh.vertices.get(*first as usize),
            core_mesh.vertices.get(*second as usize),
            core_mesh.vertices.get(*third as usize),
        ) else {
            continue;
        };
        for vertex in [first, second, third] {
            vertices.push(
                SgfxCanvasVertex::new(
                    [vertex.position.x, vertex.position.y, vertex.position.z, 1.0],
                    terrain_vertex_color(vertex, sun_phase),
                )
                .with_tex_coord(atlas_tex_coord(
                    vertex.block,
                    vertex.normal,
                    vertex.atlas_uv,
                )),
            );
        }
    }
}

/// Create the compact RGBA8 atlas used by every visible terrain block.
///
/// The small, procedural tiles deliberately have hard pixel edges: they stay
/// readable at a distance without depending on an external asset pipeline.
fn block_texture_atlas() -> Arc<SgfxTexture> {
    let width = ATLAS_COLUMNS * ATLAS_TILE_SIZE;
    let height = ATLAS_ROWS * ATLAS_TILE_SIZE;
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let tile = (x / ATLAS_TILE_SIZE + (y / ATLAS_TILE_SIZE) * ATLAS_COLUMNS) as usize;
            let color = atlas_pixel(tile, x % ATLAS_TILE_SIZE, y % ATLAS_TILE_SIZE);
            pixels.extend_from_slice(&color);
        }
    }
    SgfxTexture::rgba8(width, height, pixels)
}

fn atlas_pixel(tile: usize, x: u32, y: u32) -> [u8; 4] {
    // Fractal noise evaluated in continuous tile space: several octaves of
    // smooth value noise replace the per-pixel hash that read as harsh
    // mosaic static at a distance.
    let fx = x as f32;
    let fy = y as f32;
    let grain = tile_fbm(tile as u32, fx, fy, 4.0, 4);
    let detail = tile_fbm(tile as u32 ^ 0x5EED, fx, fy, 8.0, 3);
    let clumps = tile_fbm(tile as u32 ^ 0xC0FFEE, fx, fy, 2.0, 2);
    let shade = (grain * 22.0 + detail * 12.0) as i16;
    let mut color = match tile {
        // Grass top: broad clump tinting over fine blade detail.
        0 => {
            let dry = ((clumps - 0.15) / 0.85).clamp(0.0, 1.0);
            [
                66 + dry as i16 as u8 * 0 + (dry * 52.0) as u8,
                138 + (dry * 6.0) as u8,
                52 + (dry * 8.0) as u8,
            ]
        }
        // Grass sides: a turf cap with an organic, noise-driven edge.
        1 => {
            let edge = 5.0 + clumps * 3.0;
            if fy < edge {
                [64 + (grain * 10.0) as u8, 132, 50]
            } else {
                [124, 77, 43]
            }
        }
        // Dirt: pebbly clumps.
        2 => [118 + (clumps * 16.0) as u8, 74 + (clumps * 10.0) as u8, 42],
        // Stone: banded mineral with fracture creases.
        3 => {
            let crease = (detail.abs() * 26.0) as u8;
            [112 - crease, 120 - crease, 127 - crease]
        }
        // Wood bark: stretched vertical grain.
        4 => {
            let streak = tile_fbm(tile as u32, fx * 0.22, fy * 1.6, 3.0, 3);
            let dark = (streak * 30.0) as u8;
            [133 - dark, 85 - dark, 43 - dark]
        }
        // Wood end grain: concentric rings around the tile centre.
        5 => {
            let dx = fx - ATLAS_TILE_SIZE as f32 * 0.5;
            let dy = fy - ATLAS_TILE_SIZE as f32 * 0.5;
            let ring = ((dx * dx + dy * dy).sqrt() * 0.9 + grain * 1.8).sin();
            if ring > 0.55 {
                [112, 67, 34]
            } else {
                [157, 105, 55]
            }
        }
        // Leaves: dense foliage clumping, kept opaque for depth writes.
        6 => {
            let depth = ((clumps + 1.0) * 0.5).clamp(0.0, 1.0);
            [
                (38.0 + depth * 22.0) as u8,
                (96.0 + depth * 44.0) as u8,
                (36.0 + depth * 22.0) as u8,
            ]
        }
        // Sand: soft wind ripples.
        7 => {
            let ripple = ((fy * 0.7 + grain * 6.0).sin() * 8.0) as i16;
            let base = 218 + ripple as u8;
            [base, (base as i16 - 27) as u8, (base as i16 - 92) as u8]
        }
        // Water: gentle layered current.
        8 => {
            let wave = ((fx * 0.35 + fy * 0.2 + grain * 4.0).sin() * 0.5 + 0.5) * 14.0;
            [
                (46.0 + wave) as u8,
                (118.0 + wave) as u8,
                (180.0 + wave * 0.6) as u8,
            ]
        }
        // Snow: bright with sparse blue-grey shadow pockets.
        9 => {
            let pocket = ((clumps + 1.0) * 0.5).clamp(0.0, 1.0);
            [
                (216.0 + pocket * 34.0) as u8,
                (222.0 + pocket * 30.0) as u8,
                (230.0 + pocket * 24.0) as u8,
            ]
        }
        // Torch: a bright ember head over a wooden stick. The thin torch
        // model samples only the tile's narrow centre column.
        10 => {
            let ember = x >= 13 && x < 19 && y < 8;
            let stick = x >= 14 && x < 18 && y >= 8 && y < 23;
            if ember && y < 4 {
                [255, 236, 170]
            } else if ember {
                [244, 168, 66]
            } else if stick {
                [124, 82, 44]
            } else {
                [38, 30, 24]
            }
        }
        _ => [255, 0, 255],
    };
    for component in &mut color {
        *component = ((i16::from(*component) + shade).clamp(0, 255)) as u8;
    }
    [color[0], color[1], color[2], 255]
}

/// Smooth multi-octave value noise wrapped inside one atlas tile.
fn tile_fbm(seed: u32, x: f32, y: f32, wavelength: f32, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut total = 0.0;
    let mut scale = wavelength;
    for octave in 0..octaves {
        sum +=
            tile_value_noise(seed.wrapping_add(octave.wrapping_mul(101)), x, y, scale) * amplitude;
        total += amplitude;
        amplitude *= 0.5;
        scale *= 0.5;
    }
    sum / total
}

fn tile_value_noise(seed: u32, x: f32, y: f32, wavelength: f32) -> f32 {
    let tile = ATLAS_TILE_SIZE as f32;
    // Coordinates wrap around the tile so textures tile seamlessly.
    let gx = (x / wavelength).floor();
    let gy = (y / wavelength).floor();
    let fx = x / wavelength - gx;
    let fy = y / wavelength - gy;
    let blend_x = fx * fx * (3.0 - 2.0 * fx);
    let blend_y = fy * fy * (3.0 - 2.0 * fy);
    let sample = |ix: f32, iy: f32| {
        let wrapped_x = ix.rem_euclid(tile / wavelength);
        let wrapped_y = iy.rem_euclid(tile / wavelength);
        let bits = pixel_noise(
            seed,
            (wrapped_x * wavelength) as u32 % ATLAS_TILE_SIZE,
            (wrapped_y * wavelength) as u32 % ATLAS_TILE_SIZE,
        );
        bits as f32 / u32::MAX as f32 * 2.0 - 1.0
    };
    let lerp = |from: f32, to: f32, amount: f32| from + (to - from) * amount;
    let north = lerp(sample(gx, gy), sample(gx + 1.0, gy), blend_x);
    let south = lerp(sample(gx, gy + 1.0), sample(gx + 1.0, gy + 1.0), blend_x);
    lerp(north, south, blend_y)
}

fn pixel_noise(tile: u32, x: u32, y: u32) -> u32 {
    let mut value = tile.wrapping_mul(0x9E37_79B9) ^ x.wrapping_mul(0x85EB_CA6B);
    value ^= y.wrapping_mul(0xC2B2_AE35);
    value ^= value >> 16;
    value.wrapping_mul(0x7FEB_352D) ^ (value >> 15)
}

/// Map a core atlas coordinate into this atlas, preserving its local UV.
///
/// The core owns the material choice and emits a normalized coordinate in the
/// same four-column atlas layout. Decoding through the material keeps a tiny
/// half-texel inset around every tile, preventing sampling from an adjacent
/// block texture at a shared edge. Atlas pixels are authored top-to-bottom,
/// while SGFX samples each tile's V coordinate in the opposite direction, so
/// only the local V is flipped. Keeping the tile row fixed is important: a
/// whole-atlas flip would turn the grass row into the water row.
fn atlas_tex_coord(block: Block, normal: Vec3, atlas_uv: [f32; 2]) -> [f32; 2] {
    let tile = block_texture_tile(block, normal);
    let tile_x = tile % ATLAS_COLUMNS;
    let tile_y = tile / ATLAS_COLUMNS;
    let inset = 0.5 / ATLAS_TILE_SIZE as f32;
    let local_u = (atlas_uv[0] * ATLAS_COLUMNS as f32 - tile_x as f32).clamp(0.0, 1.0);
    let local_v = (atlas_uv[1] * ATLAS_ROWS as f32 - tile_y as f32).clamp(0.0, 1.0);
    let mapped_u =
        (tile_x as f32 + local_u.mul_add(1.0 - inset * 2.0, inset)) / ATLAS_COLUMNS as f32;
    let flipped_local_v = 1.0 - local_v;
    let mapped_v =
        (tile_y as f32 + flipped_local_v.mul_add(1.0 - inset * 2.0, inset)) / ATLAS_ROWS as f32;
    [mapped_u, mapped_v]
}

fn block_texture_tile(block: Block, normal: Vec3) -> u32 {
    match block {
        Block::Grass if normal.y > 0.5 => 0,
        Block::Grass if normal.y < -0.5 => 2,
        Block::Grass => 1,
        Block::Dirt => 2,
        Block::Stone => 3,
        Block::Wood if normal.y.abs() > 0.5 => 5,
        Block::Wood => 4,
        Block::Leaves => 6,
        Block::Sand => 7,
        Block::Water => 8,
        Block::Snow => 9,
        Block::Torch => 10,
        Block::Air => 0,
    }
}

/// Compose the two Minecraft-style light channels into a warm RGB vertex
/// multiplier. Sky light follows the day cycle; block light remains warm and
/// independent, so a torch still illuminates an enclosed room at night.
fn terrain_vertex_color(vertex: &boxcraft_core::Vertex, sun_phase: f32) -> [f32; 4] {
    let daylight = sunlight_daylight(sun_phase);
    let sky = vertex.light.clamp(0.0, 1.0) * daylight;
    let torch = vertex.torch_light.clamp(0.0, 1.0);
    let ao = (0.68 + vertex.ambient_occlusion.clamp(0.0, 1.0) * 0.32).clamp(0.0, 1.0);
    [
        (sky + torch).clamp(0.0, 1.0) * ao,
        (sky * 0.94 + torch * 0.68).clamp(0.0, 1.0) * ao,
        (sky * 0.88 + torch * 0.32).clamp(0.0, 1.0) * ao,
        1.0,
    ]
}

/// Return the normalized directional sun vector for a day-cycle phase.
fn sunlight_direction(sun_phase: f32) -> Vec3 {
    let sun_angle = sun_phase * core::f32::consts::TAU;
    Vec3::new(
        sun_angle.cos() * 0.62,
        sun_angle.sin(),
        sun_angle.sin() * 0.42,
    )
    .normalized()
}

/// Return the sun's horizon-aware daylight contribution for a phase.
fn sunlight_daylight(sun_phase: f32) -> f32 {
    (sunlight_direction(sun_phase).y * 0.5 + 0.5).clamp(0.08, 1.0)
}

fn block_name(block: Block) -> &'static str {
    match block {
        Block::Air => "Air",
        Block::Grass => "Grass",
        Block::Dirt => "Dirt",
        Block::Stone => "Stone",
        Block::Wood => "Wood",
        Block::Leaves => "Leaves",
        Block::Sand => "Sand",
        Block::Water => "Water",
        Block::Snow => "Snow",
        Block::Torch => "Torch",
    }
}

/// Return whether a horizontal chunk can intersect the camera's view cone.
///
/// This is intentionally conservative: the chunk's diagonal plus a small
/// margin is treated as a circle, so a face at the edge of the viewport is not
/// clipped just because its chunk centre is outside the exact FOV.
fn chunk_is_visible(camera_position: Vec3, camera_forward: Vec3, chunk: (i32, i32)) -> bool {
    let center = Vec3::new(
        chunk.0 as f32 * CHUNK_SIZE as f32 + CHUNK_SIZE as f32 * 0.5,
        camera_position.y,
        chunk.1 as f32 * CHUNK_SIZE as f32 + CHUNK_SIZE as f32 * 0.5,
    );
    let to_chunk = center - camera_position;
    let distance = (to_chunk.x * to_chunk.x + to_chunk.z * to_chunk.z).sqrt();
    let horizontal_forward = Vec3::new(camera_forward.x, 0.0, camera_forward.z).normalized();
    if distance <= f32::EPSILON || horizontal_forward.length() <= f32::EPSILON {
        return true;
    }

    let direction = Vec3::new(to_chunk.x / distance, 0.0, to_chunk.z / distance);
    let chunk_radius = CHUNK_SIZE as f32 * core::f32::consts::SQRT_2 * 0.5 + 2.0;
    let angular_margin = (chunk_radius / distance).clamp(0.0, 1.0).asin();
    let half_fov = CAMERA_FOV * 0.5 + angular_margin + 0.08;
    direction.dot(horizontal_forward) >= half_fov.min(core::f32::consts::PI).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_v_is_flipped_without_swapping_atlas_rows() {
        let top = atlas_tex_coord(Block::Grass, Vec3::new(0.0, 1.0, 0.0), [0.0, 0.0]);
        let bottom = atlas_tex_coord(Block::Grass, Vec3::new(0.0, 1.0, 0.0), [0.0, 1.0]);
        let water = atlas_tex_coord(Block::Water, Vec3::new(0.0, 1.0, 0.0), [0.0, 2.0]);

        assert!(top[1] > bottom[1]);
        assert!(top[1] < 1.0 / ATLAS_ROWS as f32);
        assert!(water[1] > 2.0 / ATLAS_ROWS as f32);
    }

    #[test]
    fn terrain_vertex_color_keeps_torch_light_at_night() {
        let dark = boxcraft_core::Vertex {
            light: 0.0,
            torch_light: 0.0,
            ..test_vertex()
        };
        let torch_lit = boxcraft_core::Vertex {
            light: 0.0,
            torch_light: 0.9,
            ..test_vertex()
        };
        let dark_color = terrain_vertex_color(&dark, 0.5);
        let torch_color = terrain_vertex_color(&torch_lit, 0.5);
        assert!(dark_color[0] < 0.01);
        assert!(torch_color[0] > torch_color[1]);
        assert!(torch_color[0] > dark_color[0] + 0.4);
    }

    fn test_vertex() -> boxcraft_core::Vertex {
        boxcraft_core::Vertex {
            position: Vec3::zero(),
            normal: Vec3::new(0.0, 1.0, 0.0),
            color: [1.0; 4],
            block: Block::Grass,
            atlas_uv: [0.0, 0.0],
            ambient_occlusion: 1.0,
            light: 0.0,
            torch_light: 0.0,
        }
    }
}
