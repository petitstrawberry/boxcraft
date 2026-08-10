//! ScarletUI frontend for Boxcraft.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use boxcraft_core::{Block, Game, Mat4, PlayerInput, Vec3, mesh_world};
use scarlet_ui::prelude::*;
use scarlet_ui::{
    ApplicationRunExt, ComponentElement, HeaderBar, KeyCode, KeyEvent, MouseButton, PlatformWindow,
    SgfxCanvas, SgfxCanvasDraw, SgfxCanvasFrame, SgfxCanvasHandle, SgfxCanvasVertex, SgfxMesh,
    SgfxMeshHandle, hstack, vstack, zstack,
};

const APP_ID: &str = "org.scarlet-os.boxcraft";
const WINDOW_WIDTH: f32 = 1100.0;
const WINDOW_HEIGHT: f32 = 760.0;
const REFERENCE_ASPECT: f32 = 16.0 / 9.0;
const CAMERA_FOV: f32 = 70.0_f32.to_radians();
const REACH: f32 = 6.0;
const LOOK_SENSITIVITY: f32 = 0.003;
const WORLD_SEED: u64 = 0xB0CA_FE00_2026_0001;

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
}

impl Runtime {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            last_idle: now,
            fps_sample_start: now,
            frames_since_sample: 0,
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
    mesh: State<Option<Arc<SgfxMesh>>>,
    mesh_revision: State<u64>,
    frame_revision: State<u64>,
    fps: State<u32>,
    fps_text: State<String>,
    position: State<String>,
    selected_block: State<String>,
    status: State<String>,
    canvas_handle: SgfxCanvasHandle,
    mesh_handle: SgfxMeshHandle,
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
            mesh: State::new(StateId::new(11), None),
            mesh_revision: State::new(StateId::new(12), 0),
            frame_revision: State::new(StateId::new(13), 0),
            fps: State::new(StateId::new(14), 0),
            fps_text: State::new(StateId::new(15), String::from("FPS: 0")),
            position: State::new(StateId::new(16), String::from("Position: loading")),
            selected_block: State::new(StateId::new(17), String::from("1: Grass")),
            status: State::new(
                StateId::new(18),
                String::from("Click the terrain to capture the pointer"),
            ),
            canvas_handle: SgfxCanvasHandle::new(),
            mesh_handle: SgfxMeshHandle::new(),
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
        let core_mesh = self.with_game(|game| mesh_world(&game.world));
        let mut vertices = Vec::with_capacity(core_mesh.indices.len());
        for index in core_mesh.indices {
            let Some(vertex) = core_mesh.vertices.get(index as usize) else {
                continue;
            };
            vertices.push(SgfxCanvasVertex::new(
                [vertex.position.x, vertex.position.y, vertex.position.z, 1.0],
                shaded_block_color(vertex.color, vertex.normal),
            ));
        }
        self.mesh_revision
            .update(|revision| *revision = revision.wrapping_add(1));
        let revision = self.mesh_revision.get();
        let mesh = (!vertices.is_empty())
            .then(|| SgfxMesh::with_handle(self.mesh_handle, revision, vertices));
        self.mesh.set(mesh);
        self.refresh_frame();
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
        if let Some(mesh) = self.mesh.get() {
            frame = frame.draw(SgfxCanvasDraw::new(mesh, transform));
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
        let (delta_seconds, fps) = {
            let mut runtime = match self.runtime.lock() {
                Ok(runtime) => runtime,
                Err(poisoned) => poisoned.into_inner(),
            };
            let delta_seconds = now
                .saturating_duration_since(runtime.last_idle)
                .as_secs_f32()
                .min(0.1);
            runtime.last_idle = now;
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
            (delta_seconds, fps)
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

fn shaded_block_color(color: [f32; 4], normal: Vec3) -> [f32; 4] {
    let shade = if normal.y > 0.5 {
        1.0
    } else if normal.y < -0.5 {
        0.55
    } else {
        0.78
    };
    [
        color[0] * shade,
        color[1] * shade,
        color[2] * shade,
        color[3],
    ]
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
    }
}
