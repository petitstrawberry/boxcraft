//! Cross-platform ScarletUI frontend for Boxcraft.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(target_os = "scarlet")]
use scarlet_sys::{Syscall, syscall3};

use boxcraft_core::{
    Block, CHUNK_SIZE, DEFAULT_WORLD_HEIGHT, Game, IVec3, LightUpdate, Mat4, PlayerInput, Vec3,
    VisibleSpace, World, mesh_chunk,
};
use scarlet_ui::prelude::*;
use scarlet_ui::{
    ApplicationRunExt, ComponentElement, HeaderBar, KeyCode, KeyEvent, MouseButton, PlatformWindow,
    SgfxCanvas, SgfxCanvasDraw, SgfxCanvasFrame, SgfxCanvasHandle, SgfxCanvasVertex, SgfxMesh,
    SgfxMeshHandle, SgfxTexture, hstack, vstack, zstack,
};

use crate::mesh_worker::{
    FarMeshResult, LightEdit, MeshJob, MeshResult, MeshWorkers, WORKER_COUNT,
};

const APP_ID: &str = "org.scarlet-os.boxcraft";
const WINDOW_WIDTH: f32 = 1100.0;
const WINDOW_HEIGHT: f32 = 760.0;
const REFERENCE_ASPECT: f32 = 16.0 / 9.0;
const CAMERA_FOV: f32 = 70.0_f32.to_radians();
const REACH: f32 = 6.0;
const LOOK_SENSITIVITY: f32 = 0.003;
const ATLAS_TILE_SIZE: u32 = 32;
const ATLAS_COLUMNS: u32 = 4;
const ATLAS_ROWS: u32 = 4;
/// One complete sunrise-to-sunrise cycle, matching Minecraft's 20-minute day.
const DAY_LENGTH_SECONDS: f32 = 1_200.0;
const SUNLIGHT_UPDATES_PER_SECOND: f32 = 4.0;
/// Keep both workers busy without allowing stale movement jobs to pile up.
const MAX_IN_FLIGHT_NEAR_JOBS: usize = WORKER_COUNT * 2;
const MAX_MESH_RESULTS_PER_IDLE: usize = 2;
/// Cap CPU-side vertex expansion and SGFX mesh replacement per idle tick.
/// One oversized result is still allowed through so a single chunk cannot
/// starve forever, but a burst of streamed chunks is spread over later ticks.
const MAX_MESH_UPLOAD_VERTICES_PER_IDLE: usize = 60_000;
/// Default and inclusive bounds for the configurable render distance.
const DEFAULT_RENDER_DISTANCE: i32 = 5;
const MIN_RENDER_DISTANCE: i32 = 1;
const MAX_RENDER_DISTANCE: i32 = 12;
/// Chunks within this Chebyshev radius keep individual meshes for edits. Each
/// chunk now has one textured draw; illumination is carried by its vertices.
const NEAR_CHUNK_RADIUS: i32 = 2;
/// Far meshes combine a small tile of chunks to keep SGFX draw-state commands
/// below its opaque command-stream limit while retaining coarse culling.
const FAR_MESH_GROUP_SIZE: i32 = 4;
const MAX_FAR_GROUPS_PER_JOB: usize = 4;

static NEXT_WORLD_SEED: AtomicU64 = AtomicU64::new(0xB0CA_FE00_2026_0001);

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
    present_sample_start: Instant,
    presented_since_sample: u32,
    day_seconds: f32,
    sunlight_step: u64,
    build_queue: VecDeque<(i32, i32)>,
    queued_chunks: HashSet<(i32, i32)>,
    in_flight_chunks: HashMap<(i32, i32), u64>,
    next_mesh_job_id: u64,
    terrain_revision: u64,
    player_chunk: (i32, i32),
    far_dirty: bool,
    far_chunks: Vec<(i32, i32)>,
    far_pending_chunks: Vec<(i32, i32)>,
    far_generation: u64,
    far_job_in_flight: Option<u64>,
    edit_job_in_flight: Option<u64>,
    pending_light_edits: Vec<LightEdit>,
}

impl Runtime {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            last_idle: now,
            present_sample_start: now,
            presented_since_sample: 0,
            day_seconds: 0.0,
            sunlight_step: 0,
            build_queue: VecDeque::new(),
            queued_chunks: HashSet::new(),
            in_flight_chunks: HashMap::new(),
            next_mesh_job_id: 0,
            terrain_revision: 0,
            player_chunk: (i32::MAX, i32::MAX),
            far_dirty: false,
            far_chunks: Vec::new(),
            far_pending_chunks: Vec::new(),
            far_generation: 0,
            far_job_in_flight: None,
            edit_job_in_flight: None,
            pending_light_edits: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct EditRequest {
    position: IVec3,
    block: Block,
    topology_changed: bool,
}

#[derive(Clone, Copy, Debug)]
struct MeshBounds {
    min: Vec3,
    max: Vec3,
}

impl MeshBounds {
    fn fallback(origin_x: i32, origin_z: i32, span: i32) -> Self {
        let min_x = origin_x as f32 * CHUNK_SIZE as f32;
        let min_z = origin_z as f32 * CHUNK_SIZE as f32;
        let max_x = (origin_x + span) as f32 * CHUNK_SIZE as f32;
        let max_z = (origin_z + span) as f32 * CHUNK_SIZE as f32;
        Self {
            min: Vec3::new(min_x, 0.0, min_z),
            max: Vec3::new(max_x, DEFAULT_WORLD_HEIGHT as f32, max_z),
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            min: Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }
}

#[derive(Debug)]
struct TerrainMesh {
    gpu: Arc<SgfxMesh>,
    has_block_light: bool,
    bounds: MeshBounds,
}

impl TerrainMesh {
    fn triangle_count(&self) -> usize {
        self.gpu.triangle_count()
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
    near_meshes: State<Arc<HashMap<(i32, i32), Arc<TerrainMesh>>>>,
    presented_near_meshes: State<Arc<HashMap<(i32, i32), Arc<TerrainMesh>>>>,
    far_core_meshes: State<Arc<HashMap<(i32, i32), Arc<boxcraft_core::Mesh>>>>,
    far_meshes: State<Arc<HashMap<(i32, i32), Arc<TerrainMesh>>>>,
    presented_far_meshes: State<Arc<HashMap<(i32, i32), Arc<TerrainMesh>>>>,
    mesh_revision: State<u64>,
    frame_revision: State<u64>,
    sun_phase: State<f32>,
    fps_text: State<String>,
    position: State<String>,
    selected_block: State<String>,
    status: State<String>,
    canvas_handle: SgfxCanvasHandle,
    chunk_handles: Arc<Mutex<HashMap<(i32, i32), SgfxMeshHandle>>>,
    far_handles: Arc<Mutex<HashMap<(i32, i32), SgfxMeshHandle>>>,
    handle_pool: Arc<Mutex<Vec<SgfxMeshHandle>>>,
    visible_space: Arc<Mutex<Option<Arc<VisibleSpace>>>>,
    block_atlas: Arc<SgfxTexture>,
    runtime: Arc<Mutex<Runtime>>,
    mesh_world: Arc<Mutex<Arc<World>>>,
    pending_mouse_dx: Arc<AtomicI32>,
    pending_mouse_dy: Arc<AtomicI32>,
    mesh_workers: MeshWorkers,
}

impl BoxcraftApp {
    fn new() -> Self {
        let seed = random_world_seed();
        let initial_game = Game::generated(seed);
        let mesh_world = Arc::new(Mutex::new(Arc::clone(&initial_game.world)));
        let initial_frame = Arc::new(
            SgfxCanvasFrame::new(0, sky_color(0.0))
                .depth_tested()
                .reference_aspect(REFERENCE_ASPECT),
        );
        let app = Self {
            game: State::new(StateId::new(1), Arc::new(Mutex::new(initial_game))),
            keys: State::new(StateId::new(2), PressedKeys::default()),
            seed: State::new(StateId::new(3), seed),
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
                Arc::new(HashMap::<(i32, i32), Arc<TerrainMesh>>::new()),
            ),
            presented_near_meshes: State::new(
                StateId::new(27),
                Arc::new(HashMap::<(i32, i32), Arc<TerrainMesh>>::new()),
            ),
            far_core_meshes: State::new(
                StateId::new(26),
                Arc::new(HashMap::<(i32, i32), Arc<boxcraft_core::Mesh>>::new()),
            ),
            far_meshes: State::new(
                StateId::new(24),
                Arc::new(HashMap::<(i32, i32), Arc<TerrainMesh>>::new()),
            ),
            presented_far_meshes: State::new(
                StateId::new(28),
                Arc::new(HashMap::<(i32, i32), Arc<TerrainMesh>>::new()),
            ),
            mesh_revision: State::new(StateId::new(12), 0),
            frame_revision: State::new(StateId::new(13), 0),
            sun_phase: State::new(StateId::new(20), 0.0),
            fps_text: State::new(StateId::new(15), String::from("FPS: 0")),
            position: State::new(StateId::new(16), String::from("Position: loading")),
            selected_block: State::new(StateId::new(17), String::from("1: Grass")),
            status: State::new(
                StateId::new(18),
                String::from("Click the terrain to capture the pointer"),
            ),
            canvas_handle: SgfxCanvasHandle::new(),
            chunk_handles: Arc::new(Mutex::new(HashMap::new())),
            far_handles: Arc::new(Mutex::new(HashMap::new())),
            handle_pool: Arc::new(Mutex::new(Vec::new())),
            visible_space: Arc::new(Mutex::new(None)),
            block_atlas: block_texture_atlas(),
            runtime: Arc::new(Mutex::new(Runtime::new())),
            mesh_world,
            pending_mouse_dx: Arc::new(AtomicI32::new(0)),
            pending_mouse_dy: Arc::new(AtomicI32::new(0)),
            mesh_workers: MeshWorkers::new(),
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

    fn clear_pending_mouse_delta(&self) {
        self.pending_mouse_dx.store(0, Ordering::Relaxed);
        self.pending_mouse_dy.store(0, Ordering::Relaxed);
    }

    fn take_pending_mouse_delta(&self) -> (i32, i32) {
        (
            self.pending_mouse_dx.swap(0, Ordering::Relaxed),
            self.pending_mouse_dy.swap(0, Ordering::Relaxed),
        )
    }

    fn request_pointer_lock(&self) {
        self.pointer_lock_desired.set(true);
        self.status.set(String::from("Requesting pointer capture…"));
    }

    fn release_pointer_lock(&self) {
        self.pointer_lock_desired.set(false);
        self.clear_pressed_keys();
        self.clear_pending_mouse_delta();
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
        let seed = random_world_seed();
        self.seed.set(seed);
        let game = Game::generated(seed);
        *self
            .mesh_world
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Arc::clone(&game.world);
        self.game.set(Arc::new(Mutex::new(game)));
        self.clear_pressed_keys();
        self.clear_pending_mouse_delta();
        let retired: Vec<SgfxMeshHandle> = {
            let mut handles = self
                .chunk_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            handles.drain().map(|(_, handles)| handles).collect()
        };
        let retired_far: Vec<SgfxMeshHandle> = {
            let mut handles = self
                .far_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            handles.drain().map(|(_, handle)| handle).collect()
        };
        self.handle_pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .extend(retired.into_iter().chain(retired_far));
        self.near_meshes
            .update(|chunks| *chunks = Arc::new(HashMap::<(i32, i32), Arc<TerrainMesh>>::new()));
        self.presented_near_meshes
            .set(Arc::new(HashMap::<(i32, i32), Arc<TerrainMesh>>::new()));
        self.near_core_meshes.set(Arc::new(
            HashMap::<(i32, i32), Arc<boxcraft_core::Mesh>>::new(),
        ));
        self.far_core_meshes.set(Arc::new(
            HashMap::<(i32, i32), Arc<boxcraft_core::Mesh>>::new(),
        ));
        self.far_meshes
            .set(Arc::new(HashMap::<(i32, i32), Arc<TerrainMesh>>::new()));
        self.presented_far_meshes
            .set(Arc::new(HashMap::<(i32, i32), Arc<TerrainMesh>>::new()));
        *self
            .visible_space
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        {
            let mut runtime = self
                .runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            runtime.build_queue.clear();
            runtime.queued_chunks.clear();
            runtime.in_flight_chunks.clear();
            runtime.terrain_revision = runtime.terrain_revision.wrapping_add(1);
            runtime.player_chunk = (i32::MAX, i32::MAX);
            runtime.far_dirty = false;
            runtime.far_chunks.clear();
            runtime.far_pending_chunks.clear();
            runtime.far_generation = runtime.far_generation.wrapping_add(1);
            runtime.far_job_in_flight = None;
            runtime.edit_job_in_flight = None;
            runtime.pending_light_edits.clear();
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
        // Relative-motion events can arrive several times per frame. Keep
        // input handling cheap and let on_idle apply one combined rotation,
        // just like keyboard movement. Saturation prevents a stuck device
        // from wrapping the accumulated delta.
        let _ =
            self.pending_mouse_dx
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |pending| {
                    Some(pending.saturating_add(dx))
                });
        let _ =
            self.pending_mouse_dy
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |pending| {
                    Some(pending.saturating_add(dy))
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

        let request = self.with_game(|game| match button {
            MouseButton::Left => {
                let hit = game.world.raycast(
                    game.player.camera().position,
                    game.player.camera().forward(),
                    REACH,
                )?;
                Some(EditRequest {
                    position: hit.position,
                    block: Block::Air,
                    topology_changed: game.world.block(hit.position).is_some_and(Block::occludes),
                })
            }
            MouseButton::Right => {
                let position = game.player.placement_target(&game.world, REACH)?;
                Some(EditRequest {
                    position,
                    block: game.player.selected_block,
                    topology_changed: game.player.selected_block.occludes(),
                })
            }
            MouseButton::Middle => None,
        });
        if let Some(request) = request {
            self.submit_edit(request);
        }
        true
    }

    /// Apply geometry immediately, then replace the pending background light
    /// snapshot so rendering never waits for propagation to finish.
    fn submit_edit(&self, request: EditRequest) {
        let current_world = Arc::clone(
            &self
                .mesh_world
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        if current_world.block(request.position) == Some(request.block) {
            return;
        }
        let mut next_world = (*current_world).clone();
        if !next_world.set(request.position, request.block) {
            return;
        }
        let next_world = Arc::new(next_world);
        *self
            .mesh_world
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Arc::clone(&next_world);
        self.with_game(|game| game.world = Arc::clone(&next_world));
        self.rebuild_immediate_edit_chunks(&next_world, request.position);

        let edit = LightEdit {
            position: request.position,
            topology_changed: request.topology_changed,
        };
        let (id, terrain_revision, edits) = {
            let mut runtime = self
                .runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            runtime.next_mesh_job_id = runtime.next_mesh_job_id.wrapping_add(1);
            let id = runtime.next_mesh_job_id;
            runtime.terrain_revision = runtime.terrain_revision.wrapping_add(1);
            runtime.build_queue.clear();
            runtime.queued_chunks.clear();
            runtime.in_flight_chunks.clear();
            runtime.far_job_in_flight = None;
            runtime.edit_job_in_flight = Some(id);
            runtime.pending_light_edits.push(edit);
            (
                id,
                runtime.terrain_revision,
                runtime.pending_light_edits.clone(),
            )
        };
        self.mesh_workers.submit_edit(MeshJob::Edit {
            id,
            terrain_revision,
            world: next_world,
            edits,
        });
        self.refresh_frame();
        self.status
            .set(String::from("Terrain changed · updating light…"));
    }

    /// Rebuild the edited near chunks synchronously using the previous light
    /// field. The background result replaces their lighting afterwards.
    fn rebuild_immediate_edit_chunks(&self, world: &World, position: IVec3) {
        let mut affected = HashSet::new();
        insert_chunks_for_voxel_bounds(
            &mut affected,
            position.x - 1,
            position.x + 1,
            position.z - 1,
            position.z + 1,
        );
        let resident = self.near_meshes.get();
        let rebuilt: HashMap<(i32, i32), Arc<boxcraft_core::Mesh>> = affected
            .into_iter()
            .filter(|chunk| resident.contains_key(chunk))
            .map(|chunk| (chunk, Arc::new(mesh_chunk(world, chunk.0, chunk.1))))
            .collect();
        if rebuilt.is_empty() {
            return;
        }
        self.near_core_meshes.update(|meshes| {
            let mut next = (**meshes).clone();
            for (chunk, mesh) in &rebuilt {
                next.insert(*chunk, Arc::clone(mesh));
            }
            *meshes = Arc::new(next);
        });
        for (chunk, mesh) in &rebuilt {
            self.update_near_mesh_from_core(*chunk, mesh);
        }
        let rendered = self.near_meshes.get();
        self.presented_near_meshes.update(|meshes| {
            let mut next = (**meshes).clone();
            for chunk in rebuilt.keys() {
                if let Some(mesh) = rendered.get(chunk) {
                    next.insert(*chunk, Arc::clone(mesh));
                }
            }
            *meshes = Arc::new(next);
        });
    }

    /// Rebuild every resident chunk touched by changed geometry or light.
    fn rebuild_edited_chunks(&self, edits: &[LightEdit], light_update: LightUpdate) {
        let mut affected = HashSet::new();
        for edit in edits {
            insert_chunks_for_voxel_bounds(
                &mut affected,
                edit.position.x - 1,
                edit.position.x + 1,
                edit.position.z - 1,
                edit.position.z + 1,
            );
        }
        if let Some((min_x, max_x, min_z, max_z)) = light_update.horizontal_bounds() {
            // Smooth vertex light samples cells on both sides of a chunk edge.
            insert_chunks_for_voxel_bounds(
                &mut affected,
                min_x - 1,
                max_x + 1,
                min_z - 1,
                max_z + 1,
            );
        }

        let resident_near = self.near_meshes.get();
        let mut runtime = self
            .runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for chunk in affected
            .iter()
            .copied()
            .filter(|chunk| resident_near.contains_key(chunk))
        {
            if runtime.queued_chunks.insert(chunk) {
                runtime.build_queue.push_back(chunk);
            }
        }
        drop(runtime);

        let far_chunks: HashSet<(i32, i32)> = self
            .runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .far_chunks
            .iter()
            .copied()
            .collect();
        let affected_far: Vec<(i32, i32)> = affected
            .iter()
            .copied()
            .filter(|chunk| far_chunks.contains(chunk))
            .collect();
        let far_dirty = if edits.iter().any(|edit| edit.topology_changed) {
            // Opening or sealing space can change which connected cave
            // component is visible, so topology edits invalidate the LOD map.
            self.far_core_meshes.set(Arc::new(
                HashMap::<(i32, i32), Arc<boxcraft_core::Mesh>>::new(),
            ));
            *self
                .visible_space
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
            !far_chunks.is_empty()
        } else if affected_far.is_empty() {
            false
        } else {
            self.far_core_meshes.update(|meshes| {
                let mut next = (**meshes).clone();
                for chunk in &affected_far {
                    next.remove(chunk);
                }
                *meshes = Arc::new(next);
            });
            true
        };
        if far_dirty {
            let mut runtime = self
                .runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // Rebuild complete far groups in priority order. Keeping this
            // queue explicit lets a lighting edit update the visible part of
            // the ring first instead of waiting for every distant chunk.
            runtime.far_pending_chunks = runtime.far_chunks.clone();
            runtime.far_dirty = true;
            runtime.far_generation = runtime.far_generation.wrapping_add(1);
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
        let (player_chunk, camera_position, camera_forward) = self.with_game(|game| {
            let position = game.player.position;
            let camera = game.player.camera();
            (
                (
                    (position.x as i32).div_euclid(CHUNK_SIZE),
                    (position.z as i32).div_euclid(CHUNK_SIZE),
                ),
                camera.position,
                camera.forward(),
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
        far.sort_by(|left, right| {
            far_group_stream_priority(*left, player_chunk, camera_position, camera_forward)
                .partial_cmp(&far_group_stream_priority(
                    *right,
                    player_chunk,
                    camera_position,
                    camera_forward,
                ))
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| {
                    chunk_stream_priority(*left, player_chunk, camera_position, camera_forward)
                        .partial_cmp(&chunk_stream_priority(
                            *right,
                            player_chunk,
                            camera_position,
                            camera_forward,
                        ))
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
        });

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

        let far_set: HashSet<(i32, i32)> = far.iter().copied().collect();
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
        let current_far_set: HashSet<(i32, i32)> = runtime.far_chunks.iter().copied().collect();
        if current_far_set != far_set {
            runtime.far_chunks = far;
            runtime.far_pending_chunks = runtime.far_chunks.clone();
            runtime.far_dirty = !runtime.far_pending_chunks.is_empty();
            runtime.far_generation = runtime.far_generation.wrapping_add(1);
            self.far_core_meshes.update(|meshes| {
                let mut next = (**meshes).clone();
                next.retain(|chunk, _| far_set.contains(chunk));
                *meshes = Arc::new(next);
            });
            self.far_meshes
                .set(Arc::new(HashMap::<(i32, i32), Arc<TerrainMesh>>::new()));
            // Keep the currently presented far ring alive while the incoming
            // near chunks are streamed. The completed near and far sets are
            // handed to the canvas together, so a chunk-border crossing never
            // presents an empty far ring or overlapping LODs.
            changed = true;
        } else {
            // A camera turn changes priority, not residency. Reorder only the
            // not-yet-built queue and keep the current generation/meshes
            // intact; otherwise looking around would restart the whole ring.
            let pending: HashSet<(i32, i32)> = runtime.far_pending_chunks.iter().copied().collect();
            runtime.far_chunks = far;
            runtime.far_pending_chunks = runtime
                .far_chunks
                .iter()
                .copied()
                .filter(|chunk| pending.contains(chunk))
                .collect();
        }
        let mut missing: Vec<(i32, i32)> = desired
            .into_iter()
            .filter(|chunk| {
                near.contains(chunk)
                    && !self.near_meshes.get().contains_key(chunk)
                    && !runtime.queued_chunks.contains(chunk)
            })
            .collect();
        missing.sort_by(|left, right| {
            chunk_stream_priority(*left, player_chunk, camera_position, camera_forward)
                .partial_cmp(&chunk_stream_priority(
                    *right,
                    player_chunk,
                    camera_position,
                    camera_forward,
                ))
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        for chunk in missing {
            runtime.queued_chunks.insert(chunk);
            runtime.build_queue.push_back(chunk);
        }
        changed
    }

    /// Fill the bounded worker queue with pending near chunks.
    fn dispatch_near_mesh_jobs(&self) {
        let world = Arc::clone(
            &self
                .mesh_world
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        let jobs = {
            let mut runtime = self
                .runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if runtime.edit_job_in_flight.is_some() {
                return;
            }
            let available = MAX_IN_FLIGHT_NEAR_JOBS.saturating_sub(runtime.in_flight_chunks.len());
            let mut jobs = Vec::with_capacity(available);
            for _ in 0..available {
                let Some(chunk) = runtime.build_queue.pop_front() else {
                    break;
                };
                if !runtime.queued_chunks.contains(&chunk) {
                    continue;
                }
                runtime.next_mesh_job_id = runtime.next_mesh_job_id.wrapping_add(1);
                let id = runtime.next_mesh_job_id;
                let terrain_revision = runtime.terrain_revision;
                runtime.in_flight_chunks.insert(chunk, id);
                jobs.push(MeshJob::Near {
                    id,
                    terrain_revision,
                    chunk,
                    world: Arc::clone(&world),
                });
            }
            jobs
        };
        for job in jobs {
            self.mesh_workers.submit(job);
        }
    }

    /// Start one far-ring rebuild after all incoming near chunks are complete.
    fn dispatch_far_mesh_job_if_settled(&self) {
        let request = {
            let mut runtime = self
                .runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if runtime.edit_job_in_flight.is_some()
                || !runtime.far_dirty
                || runtime.far_pending_chunks.is_empty()
                || !runtime.build_queue.is_empty()
                || !runtime.in_flight_chunks.is_empty()
                || runtime.far_job_in_flight.is_some()
            {
                return;
            }
            let mut groups = HashSet::new();
            for chunk in &runtime.far_pending_chunks {
                groups.insert(far_group_for_chunk(*chunk));
                if groups.len() >= MAX_FAR_GROUPS_PER_JOB {
                    break;
                }
            }
            let batch: Vec<(i32, i32)> = runtime
                .far_pending_chunks
                .iter()
                .copied()
                .filter(|chunk| groups.contains(&far_group_for_chunk(*chunk)))
                .collect();
            if batch.is_empty() {
                return;
            }
            runtime.next_mesh_job_id = runtime.next_mesh_job_id.wrapping_add(1);
            let id = runtime.next_mesh_job_id;
            runtime.far_job_in_flight = Some(id);
            (id, runtime.terrain_revision, runtime.far_generation, batch)
        };
        let camera = self.with_game(|game| game.player.camera().position);
        let viewer = IVec3::new(
            camera.x.floor() as i32,
            camera.y.floor() as i32,
            camera.z.floor() as i32,
        );
        let world = Arc::clone(
            &self
                .mesh_world
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        let visible_space = self
            .visible_space
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        self.mesh_workers.submit(MeshJob::Far {
            id: request.0,
            terrain_revision: request.1,
            far_generation: request.2,
            chunks: request.3,
            world,
            viewer,
            visible_space,
            cached_meshes: self.far_core_meshes.get(),
        });
    }

    /// Apply completed CPU meshes on the UI thread and discard stale jobs.
    fn process_mesh_results(&self) -> bool {
        let mut changed = false;
        let mut uploaded_vertices: usize = 0;
        for _ in 0..MAX_MESH_RESULTS_PER_IDLE {
            let Some(result) = self.mesh_workers.try_recv() else {
                break;
            };
            match result {
                MeshResult::Near(result) => {
                    let estimated_vertices = result.mesh.indices.len();
                    let current = self
                        .runtime
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .in_flight_chunks
                        .get(&result.chunk)
                        .copied()
                        == Some(result.id);
                    if current
                        && uploaded_vertices > 0
                        && uploaded_vertices.saturating_add(estimated_vertices)
                            > MAX_MESH_UPLOAD_VERTICES_PER_IDLE
                    {
                        self.mesh_workers.defer(MeshResult::Near(result));
                        break;
                    }
                    if current {
                        uploaded_vertices = uploaded_vertices.saturating_add(estimated_vertices);
                    }
                    let accepted = {
                        let mut runtime = self
                            .runtime
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        let current = runtime.in_flight_chunks.get(&result.chunk).copied();
                        if current != Some(result.id) {
                            false
                        } else {
                            runtime.in_flight_chunks.remove(&result.chunk);
                            runtime.queued_chunks.remove(&result.chunk);
                            result.terrain_revision == runtime.terrain_revision
                                && chunk_within_near_radius(
                                    result.chunk,
                                    runtime.player_chunk,
                                    self.render_distance.get(),
                                )
                        }
                    };
                    if accepted {
                        self.near_core_meshes.update(|chunks| {
                            let mut next = (**chunks).clone();
                            next.insert(result.chunk, Arc::clone(&result.mesh));
                            *chunks = Arc::new(next);
                        });
                        changed |= self.update_near_mesh_from_core(result.chunk, &result.mesh);
                    }
                }
                MeshResult::Far(result) => {
                    let estimated_vertices = result
                        .meshes
                        .values()
                        .map(|mesh| mesh.indices.len())
                        .sum::<usize>();
                    let current = self
                        .runtime
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .far_job_in_flight
                        == Some(result.id);
                    if current
                        && uploaded_vertices > 0
                        && uploaded_vertices.saturating_add(estimated_vertices)
                            > MAX_MESH_UPLOAD_VERTICES_PER_IDLE
                    {
                        self.mesh_workers.defer(MeshResult::Far(result));
                        break;
                    }
                    if current {
                        uploaded_vertices = uploaded_vertices.saturating_add(estimated_vertices);
                    }
                    let accepted = {
                        let mut runtime = self
                            .runtime
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        if runtime.far_job_in_flight != Some(result.id) {
                            false
                        } else {
                            runtime.far_job_in_flight = None;
                            let result_is_current = result.terrain_revision
                                == runtime.terrain_revision
                                && result.far_generation == runtime.far_generation
                                && runtime.build_queue.is_empty()
                                && runtime.in_flight_chunks.is_empty();
                            let result_is_resident = result
                                .chunks
                                .iter()
                                .all(|chunk| runtime.far_chunks.contains(chunk));
                            if result_is_current && result_is_resident {
                                for chunk in &result.chunks {
                                    runtime
                                        .far_pending_chunks
                                        .retain(|pending| pending != chunk);
                                }
                                runtime.far_dirty = !runtime.far_pending_chunks.is_empty();
                            }
                            result_is_current && result_is_resident
                        }
                    };
                    if accepted {
                        self.install_far_mesh_result(result);
                        changed = true;
                    }
                }
                MeshResult::Edit(result) => {
                    let accepted = {
                        let mut runtime = self
                            .runtime
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        if runtime.edit_job_in_flight != Some(result.id)
                            || result.terrain_revision != runtime.terrain_revision
                        {
                            false
                        } else {
                            runtime.edit_job_in_flight = None;
                            runtime.pending_light_edits.clear();
                            true
                        }
                    };
                    if accepted {
                        *self
                            .mesh_world
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                            Arc::clone(&result.world);
                        self.with_game(|game| game.world = Arc::clone(&result.world));
                        self.rebuild_edited_chunks(&result.edits, result.light_update);
                        self.status.set(String::from("Terrain updated"));
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    fn install_far_mesh_result(&self, result: FarMeshResult) {
        *self
            .visible_space
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(Arc::clone(&result.visible_space));
        let batch_meshes = result.meshes;
        let changed_groups: HashSet<(i32, i32)> = batch_meshes
            .keys()
            .copied()
            .map(far_group_for_chunk)
            .collect();
        self.far_core_meshes.update(|meshes| {
            let mut next = (**meshes).clone();
            for (chunk, core_mesh) in &batch_meshes {
                next.insert(*chunk, Arc::clone(core_mesh));
            }
            *meshes = Arc::new(next);
        });

        // Only rebuild the group meshes touched by this batch. The remaining
        // far groups already contain valid GPU handles and can stay resident.
        let all_core_meshes = self.far_core_meshes.get();
        self.mesh_revision
            .update(|revision| *revision = revision.wrapping_add(1));
        let revision = self.mesh_revision.get();
        let mut rebuilt = HashMap::with_capacity(changed_groups.len());
        for group in changed_groups.iter().copied() {
            let core_meshes: Vec<Arc<boxcraft_core::Mesh>> = all_core_meshes
                .iter()
                .filter(|(chunk, _)| far_group_for_chunk(**chunk) == group)
                .map(|(_, core_mesh)| Arc::clone(core_mesh))
                .collect();
            if core_meshes.is_empty() {
                continue;
            }
            let bounds = mesh_bounds_for_cores(
                &core_meshes,
                MeshBounds::fallback(
                    group.0 * FAR_MESH_GROUP_SIZE,
                    group.1 * FAR_MESH_GROUP_SIZE,
                    FAR_MESH_GROUP_SIZE,
                ),
            );
            let has_block_light = core_meshes
                .iter()
                .any(|core_mesh| mesh_has_block_light(core_mesh));
            let daylight = if has_block_light {
                sunlight_daylight(self.sun_phase.get())
            } else {
                1.0
            };
            let mut vertices = Vec::new();
            for core_mesh in core_meshes {
                append_lit_vertices(&core_mesh, daylight, &mut vertices);
            }
            if vertices.is_empty() {
                continue;
            }
            let handle = {
                let mut active = self
                    .far_handles
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Some(handle) = active.get(&group) {
                    *handle
                } else {
                    let handle = self
                        .handle_pool
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pop()
                        .unwrap_or_else(SgfxMeshHandle::new);
                    active.insert(group, handle);
                    handle
                }
            };
            rebuilt.insert(
                group,
                Arc::new(TerrainMesh {
                    gpu: SgfxMesh::with_handle(handle, revision, vertices),
                    has_block_light,
                    bounds,
                }),
            );
        }

        let desired_groups: HashSet<(i32, i32)> = self
            .runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .far_chunks
            .iter()
            .copied()
            .map(far_group_for_chunk)
            .collect();
        self.far_meshes.update(|meshes| {
            let mut next = (**meshes).clone();
            for group in &changed_groups {
                if let Some(mesh) = rebuilt.get(group) {
                    next.insert(*group, Arc::clone(mesh));
                } else {
                    next.remove(group);
                }
            }
            next.retain(|group, _| desired_groups.contains(group));
            *meshes = Arc::new(next);
        });

        // Retire handles for groups that no longer have resident geometry.
        let resident_groups = self.far_meshes.get();
        let stale_handles: Vec<(i32, i32)> = {
            let handles = self
                .far_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            handles
                .keys()
                .copied()
                .filter(|group| !resident_groups.contains_key(group))
                .collect()
        };
        if !stale_handles.is_empty() {
            let retired: Vec<SgfxMeshHandle> = {
                let mut handles = self
                    .far_handles
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
    }

    /// Keep the last complete canvas frame visible during a far-ring handoff.
    ///
    /// Near chunks are intentionally built over several idle ticks. Once a
    /// far ring has been presented, publishing those intermediate near maps
    /// would either expose holes or overlap the retained far LOD. Waiting only
    /// affects presentation; chunk meshing still advances every idle tick.
    fn terrain_handoff_pending(&self) -> bool {
        let far_dirty = self
            .runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .far_dirty;
        should_hold_presented_terrain(far_dirty, !self.presented_far_meshes.get().is_empty())
    }

    fn refresh_frame_if_terrain_ready(&self) {
        if !self.terrain_handoff_pending() {
            self.presented_near_meshes.set(self.near_meshes.get());
            self.presented_far_meshes.set(self.far_meshes.get());
        }
        // Even while workers prepare the next terrain snapshot, keep camera
        // motion smooth by redrawing the last complete snapshot.
        self.refresh_frame();
    }

    fn update_near_mesh_from_core(&self, key: (i32, i32), core_mesh: &boxcraft_core::Mesh) -> bool {
        let has_block_light = mesh_has_block_light(core_mesh);
        let daylight = if has_block_light {
            sunlight_daylight(self.sun_phase.get())
        } else {
            1.0
        };
        let mut vertices = Vec::new();
        append_lit_vertices(core_mesh, daylight, &mut vertices);
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
        let mesh = Arc::new(TerrainMesh {
            gpu: SgfxMesh::with_handle(handle, revision, vertices),
            has_block_light,
            bounds: mesh_bounds_for_core(core_mesh, key.0, key.1, 1),
        });
        self.near_meshes.update(|chunks| {
            let mut next = (**chunks).clone();
            next.insert(key, mesh);
            *chunks = Arc::new(next);
        });
        true
    }

    /// Re-bake only meshes where warm block light must remain independent of
    /// the moving sun. Sky-only meshes keep immutable vertex data and apply
    /// daylight through the draw uniform instead.
    fn rebuild_lighting_meshes(&self) -> bool {
        let mut rebuilt = false;
        let core_meshes = self.near_core_meshes.get();
        for (key, core_mesh) in core_meshes
            .iter()
            .filter(|(_, core_mesh)| mesh_has_block_light(core_mesh))
        {
            rebuilt |= self.update_near_mesh_from_core(*key, core_mesh);
        }
        rebuilt | self.rebuild_block_lit_far_meshes()
    }

    fn rebuild_block_lit_far_meshes(&self) -> bool {
        let mut groups: HashMap<(i32, i32), Vec<Arc<boxcraft_core::Mesh>>> = HashMap::new();
        for (chunk, core_mesh) in self.far_core_meshes.get().iter() {
            let group = (
                chunk.0.div_euclid(FAR_MESH_GROUP_SIZE),
                chunk.1.div_euclid(FAR_MESH_GROUP_SIZE),
            );
            groups.entry(group).or_default().push(Arc::clone(core_mesh));
        }
        groups.retain(|_, meshes| meshes.iter().any(|mesh| mesh_has_block_light(mesh)));
        if groups.is_empty() {
            return false;
        }

        self.mesh_revision
            .update(|revision| *revision = revision.wrapping_add(1));
        let revision = self.mesh_revision.get();
        let daylight = sunlight_daylight(self.sun_phase.get());
        let mut rebuilt = HashMap::with_capacity(groups.len());
        for (group, core_meshes) in groups {
            let bounds = mesh_bounds_for_cores(
                &core_meshes,
                MeshBounds::fallback(
                    group.0 * FAR_MESH_GROUP_SIZE,
                    group.1 * FAR_MESH_GROUP_SIZE,
                    FAR_MESH_GROUP_SIZE,
                ),
            );
            let mut vertices = Vec::new();
            for core_mesh in core_meshes {
                append_lit_vertices(&core_mesh, daylight, &mut vertices);
            }
            if vertices.is_empty() {
                continue;
            }
            let Some(handle) = self
                .far_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&group)
                .copied()
            else {
                continue;
            };
            rebuilt.insert(
                group,
                Arc::new(TerrainMesh {
                    gpu: SgfxMesh::with_handle(handle, revision, vertices),
                    has_block_light: true,
                    bounds,
                }),
            );
        }
        if rebuilt.is_empty() {
            return false;
        }
        self.far_meshes.update(|meshes| {
            let mut next = (**meshes).clone();
            next.extend(rebuilt);
            *meshes = Arc::new(next);
        });
        true
    }

    fn refresh_frame(&self) {
        let far_plane = camera_far_plane(self.render_distance.get());
        let (camera, transform) = self.with_game(|game| {
            let camera = game.player.camera();
            let transform = Mat4::perspective_rh_gl(CAMERA_FOV, REFERENCE_ASPECT, 0.05, far_plane)
                .mul_mat4(camera.view_matrix())
                .columns;
            (camera, transform)
        });
        self.frame_revision
            .update(|revision| *revision = revision.wrapping_add(1));
        let sun_phase = self.sun_phase.get();
        let mut frame = SgfxCanvasFrame::new(self.frame_revision.get(), sky_color(sun_phase))
            .depth_tested()
            // The SGFX renderer corrects this reference perspective as its canvas resizes.
            .reference_aspect(REFERENCE_ASPECT);
        let daylight = sunlight_daylight(sun_phase);
        let far_meshes = self.presented_far_meshes.get();
        let near_meshes = self.presented_near_meshes.get();
        let mut candidates: Vec<Arc<TerrainMesh>> = near_meshes
            .values()
            .chain(far_meshes.values())
            .filter(|mesh| {
                mesh.triangle_count() > 0
                    && chunk_is_visible(camera.position, camera.forward(), mesh.bounds, far_plane)
            })
            .cloned()
            .collect();
        candidates.sort_by(|left, right| {
            mesh_distance_sq(left.bounds, camera.position)
                .partial_cmp(&mesh_distance_sq(right.bounds, camera.position))
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        for mesh in candidates {
            let tint = if mesh.has_block_light {
                Color::WHITE
            } else {
                Color::rgb(daylight, daylight, daylight)
            };
            frame = frame.draw(
                SgfxCanvasDraw::new(Arc::clone(&mesh.gpu), transform)
                    .tint(tint)
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
        .placeholder(sky_color(self.sun_phase.get()))
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

        vstack! {
            header,
            game_area,
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
            self.clear_pending_mouse_delta();
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
        self.refresh_frame_if_terrain_ready();
    }

    fn on_focus_changed(&mut self, _window_id: u32, _app_name: &str, _menu_titles: &str) {
        self.clear_pressed_keys();
        self.clear_pending_mouse_delta();
    }

    fn on_idle(&mut self) {
        let now = Instant::now();
        let (delta_seconds, sun_phase) = {
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
            (delta_seconds, sun_phase)
        };

        let input = self.keys.get().player_input();
        let previous_camera = self.with_game(|game| game.player.camera());
        let (mouse_dx, mouse_dy) = self.take_pending_mouse_delta();
        if mouse_dx != 0 || mouse_dy != 0 {
            self.with_game(|game| {
                game.player.look(
                    mouse_dx as f32 * LOOK_SENSITIVITY,
                    -(mouse_dy as f32) * LOOK_SENSITIVITY,
                );
            });
        }
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
            // Sky-only meshes change through draw uniforms, so the frame still
            // needs a cheap command refresh even when no vertex data changed.
            frame_changed = true;
            frame_changed |= self.rebuild_lighting_meshes();
        }
        // Stream terrain chunks around the player and drop distant ones.
        frame_changed |= self.process_mesh_results();
        frame_changed |= self.refresh_chunk_set();
        self.dispatch_near_mesh_jobs();
        self.dispatch_far_mesh_job_if_settled();
        if frame_changed {
            self.refresh_frame_if_terrain_ready();
        }
    }

    fn on_frame_presented(&mut self, _ctx: &WindowContext) {
        let now = Instant::now();
        let sample = {
            let mut runtime = self
                .runtime
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            runtime.presented_since_sample = runtime.presented_since_sample.saturating_add(1);
            let elapsed = now.saturating_duration_since(runtime.present_sample_start);
            (elapsed.as_millis() >= 750).then(|| {
                let fps = runtime.presented_since_sample as f64 / elapsed.as_secs_f64();
                runtime.presented_since_sample = 0;
                runtime.present_sample_start = now;
                fps
            })
        };
        if let Some(fps) = sample {
            let frame_ms = if fps > 0.0 { 1_000.0 / fps } else { 0.0 };
            self.fps_text
                .set(format!("FPS: {fps:.1} · {frame_ms:.1} ms"));
            self.update_hud();
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

/// Produce a different terrain seed for every launch and reset.
///
/// World seeds are not security-sensitive, so the wall clock is combined with
/// a process-local sequence and an address-derived value. The latter keeps the
/// fallback useful on guests without an initialized RTC as well.
fn random_world_seed() -> u64 {
    let os_entropy = platform_entropy_u64();
    let clock = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let sequence = NEXT_WORLD_SEED.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    let address = (&clock as *const u64 as usize) as u64;
    let mut seed = os_entropy ^ clock ^ sequence.rotate_left(17) ^ address.rotate_right(11);
    seed = (seed ^ (seed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    seed = (seed ^ (seed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let seed = seed ^ (seed >> 31);
    if seed == 0 {
        0xB0CA_FE00_2026_0001
    } else {
        seed
    }
}

#[cfg(target_os = "scarlet")]
fn platform_entropy_u64() -> u64 {
    let mut bytes = [0_u8; 8];
    let result = syscall3(
        Syscall::GetRandom,
        bytes.as_mut_ptr() as usize,
        bytes.len(),
        0,
    );
    if result == bytes.len() {
        u64::from_le_bytes(bytes)
    } else {
        0
    }
}

#[cfg(not(target_os = "scarlet"))]
fn platform_entropy_u64() -> u64 {
    0
}

/// Convert one terrain chunk into a single textured mesh with interpolated
/// vertex lighting. The SGFX texture vertex-color pipeline multiplies the
/// atlas sample by this color at every fragment, so no triangle light buckets
/// or block-sized draw tints are needed.
fn append_lit_vertices(
    core_mesh: &boxcraft_core::Mesh,
    daylight: f32,
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
                    terrain_vertex_color(vertex, daylight),
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

fn mesh_has_block_light(mesh: &boxcraft_core::Mesh) -> bool {
    mesh.vertices
        .iter()
        .any(|vertex| vertex.torch_light > f32::EPSILON)
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
            let base = (218 + ripple).clamp(0, u8::MAX as i16);
            [base as u8, (base - 27) as u8, (base - 92) as u8]
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
fn terrain_vertex_color(vertex: &boxcraft_core::Vertex, daylight: f32) -> [f32; 4] {
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

/// Return a Rayleigh-inspired sky clear color for the current day-cycle phase.
///
/// The canvas has no sky geometry, so changing its clear color gives the day
/// cycle a continuous night, blue-hour, and daytime transition without adding
/// any vertices or draw calls. The clear color intentionally represents the
/// zenith-facing sky: a warm horizon color would tint the whole screen red or
/// orange because this canvas has no vertical atmospheric gradient.
fn sky_color(sun_phase: f32) -> Color {
    let sun_elevation = sunlight_direction(sun_phase).y;
    let night = Color::rgb_f32(0.008, 0.018, 0.055);
    let blue_hour = Color::rgb_f32(0.11, 0.19, 0.36);
    let day = Color::rgb_f32(0.35, 0.59, 0.86);

    let night_to_blue_hour = smoothstep(-0.34, -0.08, sun_elevation);
    let blue_hour_to_day = smoothstep(-0.08, 0.36, sun_elevation);
    let twilight = mix_color(night, blue_hour, night_to_blue_hour);
    mix_color(twilight, day, blue_hour_to_day)
}

fn mix_color(from: Color, to: Color, amount: f32) -> Color {
    let amount = amount.clamp(0.0, 1.0);
    Color::rgb_f32(
        from.r + (to.r - from.r) * amount,
        from.g + (to.g - from.g) * amount,
        from.b + (to.b - from.b) * amount,
    )
}

fn smoothstep(edge_start: f32, edge_end: f32, value: f32) -> f32 {
    let amount = ((value - edge_start) / (edge_end - edge_start)).clamp(0.0, 1.0);
    amount * amount * (3.0 - 2.0 * amount)
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

fn insert_chunks_for_voxel_bounds(
    chunks: &mut HashSet<(i32, i32)>,
    min_x: i32,
    max_x: i32,
    min_z: i32,
    max_z: i32,
) {
    let min_chunk_x = min_x.div_euclid(CHUNK_SIZE);
    let max_chunk_x = max_x.div_euclid(CHUNK_SIZE);
    let min_chunk_z = min_z.div_euclid(CHUNK_SIZE);
    let max_chunk_z = max_z.div_euclid(CHUNK_SIZE);
    for chunk_z in min_chunk_z..=max_chunk_z {
        for chunk_x in min_chunk_x..=max_chunk_x {
            chunks.insert((chunk_x, chunk_z));
        }
    }
}

fn chunk_within_near_radius(
    chunk: (i32, i32),
    player_chunk: (i32, i32),
    render_distance: i32,
) -> bool {
    let radius = render_distance.min(NEAR_CHUNK_RADIUS);
    (chunk.0 - player_chunk.0).abs() <= radius && (chunk.1 - player_chunk.1).abs() <= radius
}

fn chunk_stream_priority(
    chunk: (i32, i32),
    player_chunk: (i32, i32),
    camera_position: Vec3,
    camera_forward: Vec3,
) -> f32 {
    let center = Vec3::new(
        chunk.0 as f32 * CHUNK_SIZE as f32 + CHUNK_SIZE as f32 * 0.5,
        camera_position.y,
        chunk.1 as f32 * CHUNK_SIZE as f32 + CHUNK_SIZE as f32 * 0.5,
    );
    let offset = center - camera_position;
    let distance = offset.length().max(CHUNK_SIZE as f32);
    let horizontal_offset = Vec3::new(offset.x, 0.0, offset.z);
    let horizontal_forward = Vec3::new(camera_forward.x, 0.0, camera_forward.z).normalized();
    let alignment = if horizontal_offset.length() <= f32::EPSILON
        || horizontal_forward.length() <= f32::EPSILON
    {
        1.0
    } else {
        horizontal_offset
            .normalized()
            .dot(horizontal_forward)
            .clamp(-1.0, 1.0)
    };
    let chebyshev_distance = (chunk.0 - player_chunk.0)
        .abs()
        .max((chunk.1 - player_chunk.1).abs()) as f32;
    // Keep nearby chunks ahead of distant ones, then prefer the half-space
    // the camera faces. This reduces turn-time stalls without creating a
    // directional hole around the player.
    chebyshev_distance * 16.0 + distance * (1.0 + (1.0 - alignment.max(0.0)) * 0.75)
}

fn far_group_for_chunk(chunk: (i32, i32)) -> (i32, i32) {
    (
        chunk.0.div_euclid(FAR_MESH_GROUP_SIZE),
        chunk.1.div_euclid(FAR_MESH_GROUP_SIZE),
    )
}

fn far_group_stream_priority(
    chunk: (i32, i32),
    player_chunk: (i32, i32),
    camera_position: Vec3,
    camera_forward: Vec3,
) -> f32 {
    let group = far_group_for_chunk(chunk);
    chunk_stream_priority(
        (group.0 * FAR_MESH_GROUP_SIZE, group.1 * FAR_MESH_GROUP_SIZE),
        player_chunk,
        camera_position,
        camera_forward,
    )
}

fn should_hold_presented_terrain(far_dirty: bool, has_presented_far_ring: bool) -> bool {
    far_dirty && has_presented_far_ring
}

fn mesh_bounds_for_core(
    mesh: &boxcraft_core::Mesh,
    origin_x: i32,
    origin_z: i32,
    span: i32,
) -> MeshBounds {
    let Some(first) = mesh.vertices.first() else {
        return MeshBounds::fallback(origin_x, origin_z, span);
    };
    let mut bounds = MeshBounds {
        min: first.position,
        max: first.position,
    };
    for vertex in &mesh.vertices[1..] {
        bounds = bounds.union(MeshBounds {
            min: vertex.position,
            max: vertex.position,
        });
    }
    bounds
}

fn mesh_bounds_for_cores(meshes: &[Arc<boxcraft_core::Mesh>], fallback: MeshBounds) -> MeshBounds {
    meshes
        .iter()
        .filter_map(|mesh| {
            mesh.vertices.first().map(|first| {
                let mut bounds = MeshBounds {
                    min: first.position,
                    max: first.position,
                };
                for vertex in &mesh.vertices[1..] {
                    bounds = bounds.union(MeshBounds {
                        min: vertex.position,
                        max: vertex.position,
                    });
                }
                bounds
            })
        })
        .reduce(MeshBounds::union)
        .unwrap_or(fallback)
}

fn camera_far_plane(render_distance: i32) -> f32 {
    let diameter = (render_distance.max(1) + 1) as f32 * CHUNK_SIZE as f32;
    (diameter * core::f32::consts::SQRT_2 + CHUNK_SIZE as f32).max(128.0)
}

fn mesh_distance_sq(bounds: MeshBounds, camera_position: Vec3) -> f32 {
    let offset = (bounds.min + bounds.max) * 0.5 - camera_position;
    offset.dot(offset)
}

/// Return whether a mesh's world-space AABB can intersect the camera frustum.
///
/// The old test used only a horizontal chunk centre and therefore ignored
/// camera pitch, terrain height, and the actual mesh extents. This conservative
/// AABB/frustum test uses the same vertical FOV and reference aspect as the
/// projection, so rotating the camera and the visible chunk set agree.
fn chunk_is_visible(
    camera_position: Vec3,
    camera_forward: Vec3,
    bounds: MeshBounds,
    far_plane: f32,
) -> bool {
    let forward = camera_forward.normalized();
    if forward.length() <= f32::EPSILON {
        return true;
    }
    let right = Vec3::new(-forward.z, 0.0, forward.x).normalized();
    if right.length() <= f32::EPSILON {
        return true;
    }
    let up = right.cross(forward).normalized();
    let center = (bounds.min + bounds.max) * 0.5;
    let extents = (bounds.max - bounds.min) * 0.5;
    let relative = center - camera_position;
    let depth = relative.dot(forward);
    let depth_radius =
        extents.x * forward.x.abs() + extents.y * forward.y.abs() + extents.z * forward.z.abs();
    if depth + depth_radius < 0.05 || depth - depth_radius > far_plane {
        return false;
    }

    let horizontal_tangent = (CAMERA_FOV * 0.5).tan() * REFERENCE_ASPECT;
    let vertical_tangent = (CAMERA_FOV * 0.5).tan();
    let horizontal_distance = relative.dot(right).abs();
    let horizontal_radius =
        extents.x * right.x.abs() + extents.y * right.y.abs() + extents.z * right.z.abs();
    let vertical_distance = relative.dot(up).abs();
    let vertical_radius = extents.x * up.x.abs() + extents.y * up.y.abs() + extents.z * up.z.abs();
    let forward_for_fov = depth.max(0.0);
    horizontal_distance
        <= forward_for_fov * horizontal_tangent
            + horizontal_radius
            + depth_radius * horizontal_tangent
        && vertical_distance
            <= forward_for_fov * vertical_tangent
                + vertical_radius
                + depth_radius * vertical_tangent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_far_ring_stays_presented_until_the_replacement_is_ready() {
        assert!(should_hold_presented_terrain(true, true));
        assert!(!should_hold_presented_terrain(false, true));
        assert!(!should_hold_presented_terrain(true, false));
    }

    #[test]
    fn near_radius_rejects_stale_worker_results_after_movement() {
        assert!(chunk_within_near_radius((8, 7), (6, 6), 3));
        assert!(!chunk_within_near_radius((9, 7), (6, 6), 3));
    }

    #[test]
    fn chunk_containing_the_camera_is_never_angle_culled() {
        let camera = Vec3::new(8.0, 10.0, 8.0);
        let away_from_center = Vec3::new(-1.0, 0.0, -1.0).normalized();
        let bounds = MeshBounds::fallback(0, 0, 1);
        assert!(chunk_is_visible(camera, away_from_center, bounds, 128.0));
    }

    #[test]
    fn chunk_culling_uses_the_camera_vertical_fov() {
        let camera = Vec3::zero();
        let forward = Vec3::new(0.0, 0.0, -1.0);
        let centered = MeshBounds {
            min: Vec3::new(-1.0, -1.0, -21.0),
            max: Vec3::new(1.0, 1.0, -19.0),
        };
        let high = MeshBounds {
            min: Vec3::new(-1.0, 20.0, -21.0),
            max: Vec3::new(1.0, 22.0, -19.0),
        };
        assert!(chunk_is_visible(camera, forward, centered, 128.0));
        assert!(!chunk_is_visible(camera, forward, high, 128.0));
    }

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

    #[test]
    fn sky_only_mesh_can_apply_daylight_as_a_draw_tint() {
        let sky_lit = boxcraft_core::Vertex {
            light: 0.83,
            torch_light: 0.0,
            ambient_occlusion: 0.72,
            ..test_vertex()
        };
        let daylight = 0.31;
        let baked_day = terrain_vertex_color(&sky_lit, 1.0);
        let direct = terrain_vertex_color(&sky_lit, daylight);
        for channel in 0..3 {
            assert!((baked_day[channel] * daylight - direct[channel]).abs() < 1.0e-6);
        }
    }

    #[test]
    fn sky_color_follows_sunrise_day_sunset_and_night() {
        let sunrise = sky_color(0.0);
        let day = sky_color(0.25);
        let sunset = sky_color(0.5);
        let night = sky_color(0.75);

        assert!(sunrise.b > sunrise.r);
        assert!(sunset.b > sunset.r);
        assert!(day.b > day.r);
        assert!(day.b > night.b + 0.8);
        assert!(night.r < 0.02);
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
