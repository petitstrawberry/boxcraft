//! ScarletUI frontend for Boxcraft.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use boxcraft_core::{Block, Game, Mat4, Mesh, PlayerInput, Vec3, mesh_world};
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
const ATLAS_TILE_SIZE: u32 = 16;
const ATLAS_COLUMNS: u32 = 4;
const ATLAS_ROWS: u32 = 3;
const DAY_LENGTH_SECONDS: f32 = 150.0;
const SUNLIGHT_UPDATES_PER_SECOND: f32 = 8.0;
/// Number of retained textured terrain draws, from deepest shadow to sunlight.
const TERRAIN_LIGHT_BUCKETS: usize = 4;

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
    canvas_frame: State<Arc<SgfxCanvasFrame>>,
    terrain_mesh: State<Arc<Mesh>>,
    terrain_meshes: State<Arc<[Option<Arc<SgfxMesh>>; TERRAIN_LIGHT_BUCKETS]>>,
    mesh_revision: State<u64>,
    frame_revision: State<u64>,
    sun_phase: State<f32>,
    fps: State<u32>,
    fps_text: State<String>,
    position: State<String>,
    selected_block: State<String>,
    status: State<String>,
    canvas_handle: SgfxCanvasHandle,
    mesh_handles: [SgfxMeshHandle; TERRAIN_LIGHT_BUCKETS],
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
            canvas_frame: State::new(StateId::new(10), initial_frame),
            terrain_mesh: State::new(StateId::new(19), Arc::new(Mesh::default())),
            terrain_meshes: State::new(StateId::new(11), Arc::new(std::array::from_fn(|_| None))),
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
            mesh_handles: std::array::from_fn(|_| SgfxMeshHandle::new()),
            block_atlas: block_texture_atlas(),
            runtime: Arc::new(Mutex::new(Runtime::new())),
        };
        app.rebuild_mesh();
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

    fn reset_world(&self) {
        self.seed.update(|seed| *seed = seed.wrapping_add(1));
        let seed = self.seed.get();
        self.game.set(Arc::new(Mutex::new(Game::generated(seed))));
        self.clear_pressed_keys();
        self.rebuild_mesh();
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
                    self.release_pointer_lock();
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
                    self.select_block(6, Block::Air);
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

        let changed = self.with_game(|game| match button {
            MouseButton::Left => game.player.break_block(&mut game.world, REACH).is_some(),
            MouseButton::Right => game.player.place_block(&mut game.world, REACH).is_some(),
            MouseButton::Middle => false,
        });
        if changed {
            self.rebuild_mesh();
        }
        true
    }

    fn rebuild_mesh(&self) {
        self.terrain_mesh
            .set(Arc::new(self.with_game(|game| mesh_world(&game.world))));
        self.rebuild_render_mesh();
        self.refresh_frame();
    }

    /// Rebuild the retained SGFX mesh from the latest terrain snapshot.
    ///
    /// Terrain supplies local UVs, material identity, and ambient-occlusion
    /// light. The frontend maps those values into the pixel atlas, then groups
    /// complete triangles by their moving-sun illumination. Textured SGFX draws
    /// use each group's uniform tint because that pipeline does not consume
    /// per-vertex colors.
    fn rebuild_render_mesh(&self) {
        let core_mesh = self.terrain_mesh.get();
        let sun_phase = self.sun_phase.get();
        let mut bucket_vertices: [Vec<SgfxCanvasVertex>; TERRAIN_LIGHT_BUCKETS] =
            std::array::from_fn(|_| Vec::new());

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
            let bucket = terrain_triangle_light_bucket([first, second, third], sun_phase);
            let vertices = &mut bucket_vertices[bucket];
            for vertex in [first, second, third] {
                vertices.push(
                    // Textured SGFX currently ignores vertex color. Keep this
                    // neutral so the bucket draw tint remains the single
                    // source of textured terrain illumination.
                    SgfxCanvasVertex::new(
                        [vertex.position.x, vertex.position.y, vertex.position.z, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    )
                    .with_tex_coord(atlas_tex_coord(
                        vertex.block,
                        vertex.normal,
                        vertex.atlas_uv,
                    )),
                );
            }
        }
        self.mesh_revision
            .update(|revision| *revision = revision.wrapping_add(1));
        let revision = self.mesh_revision.get();
        let meshes = std::array::from_fn(|bucket| {
            let vertices = core::mem::take(&mut bucket_vertices[bucket]);
            (!vertices.is_empty())
                .then(|| SgfxMesh::with_handle(self.mesh_handles[bucket], revision, vertices))
        });
        self.terrain_meshes.set(Arc::new(meshes));
    }

    fn refresh_frame(&self) {
        let transform = self.with_game(|game| {
            let camera = game.player.camera();
            Mat4::perspective_rh_gl(CAMERA_FOV, REFERENCE_ASPECT, 0.05, 128.0)
                .mul_mat4(camera.view_matrix())
                .columns
        });
        self.frame_revision
            .update(|revision| *revision = revision.wrapping_add(1));
        let mut frame =
            SgfxCanvasFrame::new(self.frame_revision.get(), Color::rgb(0.045, 0.075, 0.13))
                .depth_tested()
                // The SGFX renderer corrects this reference perspective as its canvas resizes.
                .reference_aspect(REFERENCE_ASPECT);
        let terrain_meshes = self.terrain_meshes.get();
        let sun_phase = self.sun_phase.get();
        for (bucket, mesh) in terrain_meshes.iter().enumerate() {
            if let Some(mesh) = mesh {
                frame = frame.draw(
                    SgfxCanvasDraw::new(Arc::clone(mesh), transform)
                        .tint(terrain_bucket_tint(bucket, sun_phase))
                        .texture(Arc::clone(&self.block_atlas)),
                );
            }
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
        let pointer_locked = self.pointer_lock_applied.get();
        let fullscreen_desired = self.fullscreen_desired.get();

        let controls = scarlet_ui::if_view!(
            !pointer_locked,
            hstack! {
                Button::new("Capture pointer").header_style().on_click(move || capture.request_pointer_lock()),
                Button::new("Reset world").header_style().on_click(move || reset.reset_world()),
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
        }
        .alignment(Alignment::Center)
        .frame(f32::INFINITY, f32::INFINITY)
        .on_mouse_delta(move |dx, dy| pointer_delta.handle_mouse_delta(dx, dy))
        .on_mouse_button(move |button, pressed| canvas_input.handle_mouse_button(button, pressed));

        let pointer_help = if pointer_locked {
            String::from("Esc to release · Left break · Right place · WASD + Space to move")
        } else {
            String::from("Click to capture pointer · UI controls are available while unlocked")
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
    }

    fn on_window_resize(&mut self, _ctx: &WindowContext, _width: u32, _height: u32) {
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
        self.with_game(|game| {
            game.player.step(&game.world, input, delta_seconds);
            if game.player.position.y < -4.0 {
                game.player.position = game.world.spawn_point();
                game.player.velocity = Vec3::zero();
                game.player.grounded = false;
            }
        });
        if let Some(sun_phase) = sun_phase {
            self.sun_phase.set(sun_phase);
            self.rebuild_render_mesh();
        }
        if let Some(fps) = fps {
            self.fps.set(fps);
            self.fps_text.set(format!("FPS: {fps}"));
            self.update_hud();
        }
        self.refresh_frame();
    }

    fn scenes(&self) -> impl Scene {
        WindowGroup::new(
            "main",
            Window::new("Boxcraft", self.clone())
                .app_id(APP_ID)
                .size(Size::new(WINDOW_WIDTH, WINDOW_HEIGHT))
                .min_size(Size::new(720.0, 480.0))
                .resizable(true),
        )
    }
}

fn build_boxcraft_content(app: &BoxcraftApp) -> Box<dyn View> {
    Box::new(app.content())
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
    let noise = pixel_noise(tile as u32, x, y);
    let shade = match noise & 0b11 {
        0 => -12,
        1 => -4,
        2 => 4,
        _ => 10,
    };
    let mut color = match tile {
        // Grass top: rich blades with sparse dry speckles.
        0 if noise % 19 == 0 => [118, 142, 56],
        0 => [70, 143, 56],
        // Grass sides have a short turf cap over soil.
        1 if y < 4 || (y == 4 && noise % 3 != 0) => [68, 140, 55],
        1 => [124, 77, 43],
        // Dirt.
        2 => [126, 78, 44],
        // Stone, with occasional darker fracture pixels.
        3 if (x + y * 3) % 11 == 0 || noise % 23 == 0 => [87, 94, 102],
        3 => [119, 126, 132],
        // Wood bark: vertical dark grain.
        4 if x % 5 == 0 || (x + y) % 13 == 0 => [76, 48, 28],
        4 => [133, 85, 43],
        // Wood end grain: a simple square ring.
        5 if x
            .min(y)
            .min(ATLAS_TILE_SIZE - 1 - x)
            .min(ATLAS_TILE_SIZE - 1 - y)
            % 4
            == 0 =>
        {
            [95, 57, 30]
        }
        5 => [157, 105, 55],
        // Leaves: dense green clusters, kept opaque for stable depth writes.
        6 if noise % 7 == 0 => [45, 96, 45],
        6 => [53, 126, 54],
        // Sand.
        7 if noise % 13 == 0 => [202, 175, 105],
        7 => [222, 194, 121],
        // Water: a blue tile with a subtle horizontal current.
        8 if (x + y * 2) % 9 < 2 => [71, 156, 203],
        8 => [48, 121, 183],
        _ => [255, 0, 255],
    };
    for component in &mut color {
        *component = ((i16::from(*component) + shade).clamp(0, 255)) as u8;
    }
    [color[0], color[1], color[2], 255]
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
        Block::Air => 0,
    }
}

/// Place a complete triangle in a stable coarse illumination bucket.
///
/// Textured SGFX draws apply one tint uniform per draw, rather than a color
/// per vertex. Averaging the three source corners makes each triangle's shadow
/// band stable while retaining both baked AO and moving sun direction.
fn terrain_triangle_light_bucket(vertices: [&boxcraft_core::Vertex; 3], sun_phase: f32) -> usize {
    let brightness = vertices
        .into_iter()
        .map(|vertex| {
            sunlight_brightness(
                vertex.normal,
                vertex.light,
                vertex.ambient_occlusion,
                sun_phase,
            )
        })
        .sum::<f32>()
        / 3.0;
    terrain_light_bucket(brightness)
}

/// Convert normalized illumination to a bounded terrain-draw bucket.
fn terrain_light_bucket(brightness: f32) -> usize {
    ((brightness.clamp(0.0, 1.0) * TERRAIN_LIGHT_BUCKETS as f32) as usize)
        .min(TERRAIN_LIGHT_BUCKETS - 1)
}

/// Build the uniform tint used for one illuminated terrain bucket.
///
/// The color temperature follows the moving sun as well as the bucket's
/// brightness. This is deliberately a draw tint so it affects textured pixels
/// even though SGFX's textured pipeline ignores `SgfxCanvasVertex::color`.
fn terrain_bucket_tint(bucket: usize, sun_phase: f32) -> Color {
    let brightness =
        ((bucket.min(TERRAIN_LIGHT_BUCKETS - 1) as f32) + 0.5) / TERRAIN_LIGHT_BUCKETS as f32;
    let daylight = sunlight_daylight(sun_phase);
    Color::rgb(
        brightness,
        brightness * (0.86 + daylight * 0.14),
        brightness * (0.69 + daylight * 0.31),
    )
}

/// Combine a moving directional sun with the terrain's baked occlusion light.
fn sunlight_brightness(
    normal: Vec3,
    sky_light: f32,
    ambient_occlusion: f32,
    sun_phase: f32,
) -> f32 {
    let sun = sunlight_direction(sun_phase);
    let daylight = sunlight_daylight(sun_phase);
    let direct = normal.dot(sun).max(0.0) * daylight;
    let ambient = 0.08 + daylight * 0.12 + sky_light.clamp(0.0, 1.0) * (0.10 + daylight * 0.22);
    let occlusion = 0.52 + ambient_occlusion.clamp(0.0, 1.0) * 0.48;
    ((ambient + direct * 0.68) * occlusion).clamp(0.0, 1.0)
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
    }
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
        assert!(top[1] < 1.0 / ATLAS_ROWS);
        assert!(water[1] > 2.0 / ATLAS_ROWS);
    }

    #[test]
    fn terrain_light_bucket_is_bounded_and_ordered() {
        assert_eq!(terrain_light_bucket(-1.0), 0);
        assert_eq!(terrain_light_bucket(0.0), 0);
        assert!(terrain_light_bucket(0.25) > terrain_light_bucket(0.0));
        assert_eq!(terrain_light_bucket(1.0), TERRAIN_LIGHT_BUCKETS - 1);
        assert_eq!(terrain_light_bucket(2.0), TERRAIN_LIGHT_BUCKETS - 1);
    }
}
