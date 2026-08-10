//! A dependency-free domain layer for a small Minecraft-inspired voxel game.
//!
//! The crate owns simulation data and deliberately does not know about windows,
//! graphics APIs, audio, or input backends.  A renderer can consume [`Mesh`] and
//! [`Camera`], while an application converts its input state into [`PlayerInput`].

use core::ops::{Add, AddAssign, Mul, Sub, SubAssign};

/// A three-dimensional floating-point vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// Horizontal x component.
    pub x: f32,
    /// Vertical y component.
    pub y: f32,
    /// Depth z component.
    pub z: f32,
}

impl Vec3 {
    /// Creates a vector from its components.
    ///
    /// # Arguments
    ///
    /// * `x` - Horizontal component.
    /// * `y` - Vertical component.
    /// * `z` - Depth component.
    ///
    /// # Returns
    ///
    /// A vector containing the supplied components.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Returns the vector with all components zero.
    ///
    /// # Returns
    ///
    /// The origin vector.
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Returns the dot product with `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - The vector to project onto this vector.
    ///
    /// # Returns
    ///
    /// The scalar dot product.
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Returns the cross product with `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - The vector crossed with this vector.
    ///
    /// # Returns
    ///
    /// A vector perpendicular to both input vectors.
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Returns the Euclidean length.
    ///
    /// # Returns
    ///
    /// The non-negative magnitude of this vector.
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Returns a unit vector, or zero when this vector has no direction.
    ///
    /// # Returns
    ///
    /// A unit-length vector, or [`Vec3::zero`] for a zero-length input.
    pub fn normalized(self) -> Self {
        let length = self.length();
        if length > f32::EPSILON {
            self * (1.0 / length)
        } else {
            Self::zero()
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

/// An integral voxel coordinate.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct IVec3 {
    /// Horizontal x coordinate.
    pub x: i32,
    /// Vertical y coordinate.
    pub y: i32,
    /// Depth z coordinate.
    pub z: i32,
}

impl IVec3 {
    /// Creates a voxel coordinate.
    ///
    /// # Arguments
    ///
    /// * `x` - Horizontal voxel coordinate.
    /// * `y` - Vertical voxel coordinate.
    /// * `z` - Depth voxel coordinate.
    ///
    /// # Returns
    ///
    /// The coordinate containing the supplied components.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

impl Add for IVec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

/// A column-major 4 by 4 matrix suitable for GPU uniforms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    /// Matrix values in column-major order.
    pub columns: [f32; 16],
}

impl Mat4 {
    /// Returns the identity matrix.
    ///
    /// # Returns
    ///
    /// A matrix that leaves positions unchanged.
    pub const fn identity() -> Self {
        Self {
            columns: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Builds a right-handed perspective matrix with an OpenGL-style depth range.
    ///
    /// # Arguments
    ///
    /// * `vertical_fov_radians` - Vertical field of view in radians.
    /// * `aspect` - Viewport width divided by height.
    /// * `near` - Positive near clipping distance.
    /// * `far` - Far clipping distance greater than `near`.
    ///
    /// # Returns
    ///
    /// A column-major projection matrix.
    pub fn perspective_rh_gl(vertical_fov_radians: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (vertical_fov_radians * 0.5).tan();
        let nf = 1.0 / (near - far);
        Self {
            columns: [
                f / aspect,
                0.0,
                0.0,
                0.0,
                0.0,
                f,
                0.0,
                0.0,
                0.0,
                0.0,
                (far + near) * nf,
                -1.0,
                0.0,
                0.0,
                2.0 * far * near * nf,
                0.0,
            ],
        }
    }

    /// Builds a right-handed view matrix looking from `eye` towards `target`.
    ///
    /// # Arguments
    ///
    /// * `eye` - Camera position.
    /// * `target` - Point the camera looks towards.
    /// * `up` - Approximate world-up direction.
    ///
    /// # Returns
    ///
    /// A column-major right-handed view matrix.
    pub fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let forward = (eye - target).normalized();
        let right = up.cross(forward).normalized();
        let camera_up = forward.cross(right);
        Self {
            columns: [
                right.x,
                camera_up.x,
                forward.x,
                0.0,
                right.y,
                camera_up.y,
                forward.y,
                0.0,
                right.z,
                camera_up.z,
                forward.z,
                0.0,
                -right.dot(eye),
                -camera_up.dot(eye),
                -forward.dot(eye),
                1.0,
            ],
        }
    }

    /// Multiplies this matrix by `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Matrix applied before this matrix.
    ///
    /// # Returns
    ///
    /// The product `self * other`.
    pub fn mul_mat4(self, other: Self) -> Self {
        let mut result = [0.0; 16];
        for column in 0..4 {
            for row in 0..4 {
                result[column * 4 + row] = (0..4)
                    .map(|index| self.columns[index * 4 + row] * other.columns[column * 4 + index])
                    .sum();
            }
        }
        Self { columns: result }
    }

    /// Returns this transform with its clip-space Y axis inverted.
    ///
    /// SGFX uses an upper-left viewport convention, so conventional OpenGL
    /// perspective transforms need this adjustment before submission.
    ///
    /// # Returns
    ///
    /// A matrix whose clip-space Y output is the negative of this matrix.
    pub fn with_inverted_clip_y(mut self) -> Self {
        for index in [1usize, 5, 9, 13] {
            self.columns[index] = -self.columns[index];
        }
        self
    }

    /// Transforms a position with homogeneous w=1.
    ///
    /// # Arguments
    ///
    /// * `point` - Object-space position to transform.
    ///
    /// # Returns
    ///
    /// The transformed position after homogeneous division when possible.
    pub fn transform_point(self, point: Vec3) -> Vec3 {
        let m = self.columns;
        let x = m[0] * point.x + m[4] * point.y + m[8] * point.z + m[12];
        let y = m[1] * point.x + m[5] * point.y + m[9] * point.z + m[13];
        let z = m[2] * point.x + m[6] * point.y + m[10] * point.z + m[14];
        let w = m[3] * point.x + m[7] * point.y + m[11] * point.z + m[15];
        if w.abs() > f32::EPSILON {
            Vec3::new(x / w, y / w, z / w)
        } else {
            Vec3::new(x, y, z)
        }
    }
}

/// The type of material stored by one voxel.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum Block {
    /// Empty space, which has no collision or mesh faces.
    #[default]
    Air,
    /// A grass-topped surface block.
    Grass,
    /// Soil below grass.
    Dirt,
    /// Deep, hard terrain.
    Stone,
    /// Tree trunk material.
    Wood,
    /// Tree canopy material.
    Leaves,
    /// Sandy shoreline material.
    Sand,
}

impl Block {
    /// Returns whether this block fills space and collides with the player.
    ///
    /// # Returns
    ///
    /// `true` for every block except [`Block::Air`].
    pub const fn is_solid(self) -> bool {
        !matches!(self, Self::Air)
    }

    /// Returns a stable linear RGBA color for simple renderers.
    ///
    /// # Returns
    ///
    /// An `[r, g, b, a]` color associated with this material.
    pub const fn color(self) -> [f32; 4] {
        match self {
            Self::Air => [0.0, 0.0, 0.0, 0.0],
            Self::Grass => [0.30, 0.67, 0.24, 1.0],
            Self::Dirt => [0.42, 0.25, 0.12, 1.0],
            Self::Stone => [0.45, 0.47, 0.50, 1.0],
            Self::Wood => [0.38, 0.22, 0.10, 1.0],
            Self::Leaves => [0.16, 0.48, 0.16, 1.0],
            Self::Sand => [0.76, 0.67, 0.39, 1.0],
        }
    }
}

/// A finite, densely stored voxel world.
#[derive(Clone, Debug, PartialEq)]
pub struct World {
    width: usize,
    height: usize,
    depth: usize,
    blocks: Vec<Block>,
}

impl World {
    /// Creates an empty world with the requested dimensions.
    ///
    /// # Arguments
    ///
    /// * `width` - Number of voxels on the x axis.
    /// * `height` - Number of voxels on the y axis.
    /// * `depth` - Number of voxels on the z axis.
    ///
    /// # Returns
    ///
    /// An all-air world with densely allocated storage.
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        Self {
            width,
            height,
            depth,
            blocks: vec![Block::Air; width.saturating_mul(height).saturating_mul(depth)],
        }
    }

    /// Creates the compact default world used by the game (32 by 16 by 32).
    ///
    /// # Returns
    ///
    /// A 32 by 16 by 32 all-air world.
    pub fn default_sized() -> Self {
        Self::new(32, 16, 32)
    }

    /// Generates deterministic terrain from `seed`, including a few trees.
    ///
    /// # Arguments
    ///
    /// * `seed` - Stable terrain seed.
    ///
    /// # Returns
    ///
    /// A default-sized world whose terrain is entirely determined by `seed`.
    pub fn generate(seed: u64) -> Self {
        let mut world = Self::default_sized();
        for z in 0..world.depth as i32 {
            for x in 0..world.width as i32 {
                let height = terrain_height(seed, x, z);
                for y in 0..=height {
                    let block = if y == height {
                        if height <= 4 {
                            Block::Sand
                        } else {
                            Block::Grass
                        }
                    } else if y + 3 >= height {
                        Block::Dirt
                    } else {
                        Block::Stone
                    };
                    world.set(IVec3::new(x, y, z), block);
                }
            }
        }
        for z in 2..world.depth as i32 - 2 {
            for x in 2..world.width as i32 - 2 {
                let ground = terrain_height(seed, x, z);
                if ground > 5
                    && ground + 5 < world.height as i32
                    && hash(seed ^ 0xA1B2_C3D4, x, z).is_multiple_of(29)
                {
                    world.add_tree(IVec3::new(x, ground + 1, z));
                }
            }
        }
        world
    }

    /// Returns `(width, height, depth)` in voxels.
    ///
    /// # Returns
    ///
    /// The world dimensions in x, y, z order.
    pub const fn dimensions(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.depth)
    }

    /// Returns whether a coordinate is inside this finite world.
    ///
    /// # Arguments
    ///
    /// * `position` - Voxel coordinate to examine.
    ///
    /// # Returns
    ///
    /// `true` when every coordinate component is within the world bounds.
    pub fn contains(&self, position: IVec3) -> bool {
        position.x >= 0
            && position.y >= 0
            && position.z >= 0
            && position.x < self.width as i32
            && position.y < self.height as i32
            && position.z < self.depth as i32
    }

    /// Returns the block at `position`, or `None` outside the world.
    ///
    /// # Arguments
    ///
    /// * `position` - Voxel coordinate to read.
    ///
    /// # Returns
    ///
    /// The stored block, or `None` when `position` is out of bounds.
    pub fn block(&self, position: IVec3) -> Option<Block> {
        self.index(position).map(|index| self.blocks[index])
    }

    /// Replaces a block when `position` is inside the world.
    ///
    /// # Arguments
    ///
    /// * `position` - Voxel coordinate to update.
    /// * `block` - New material for the cell.
    ///
    /// # Returns
    ///
    /// Returns `true` if a cell was updated.
    pub fn set(&mut self, position: IVec3, block: Block) -> bool {
        if let Some(index) = self.index(position) {
            self.blocks[index] = block;
            true
        } else {
            false
        }
    }

    /// Returns a natural player spawn above the terrain near the world's centre.
    ///
    /// # Returns
    ///
    /// A feet-centre position with two air cells above solid terrain when available.
    pub fn spawn_point(&self) -> Vec3 {
        let centre = IVec3::new((self.width / 2) as i32, 0, (self.depth / 2) as i32);
        for radius in 0..self.width.max(self.depth) as i32 {
            for offset in [
                IVec3::new(radius, 0, 0),
                IVec3::new(-radius, 0, 0),
                IVec3::new(0, 0, radius),
                IVec3::new(0, 0, -radius),
            ] {
                let position = centre + offset;
                if !self.contains(position) {
                    continue;
                }
                for y in (0..self.height as i32 - 1).rev() {
                    let below = IVec3::new(position.x, y, position.z);
                    if self.block(below).is_some_and(Block::is_solid)
                        && self.block(IVec3::new(position.x, y + 1, position.z)) == Some(Block::Air)
                        && self.block(IVec3::new(position.x, y + 2, position.z)) == Some(Block::Air)
                    {
                        return Vec3::new(
                            position.x as f32 + 0.5,
                            y as f32 + 1.001,
                            position.z as f32 + 0.5,
                        );
                    }
                }
            }
        }
        Vec3::new(0.5, self.height as f32, 0.5)
    }

    /// Casts a ray through voxels and returns the first solid block reached.
    ///
    /// # Arguments
    ///
    /// * `origin` - Ray starting position in world coordinates.
    /// * `direction` - Ray direction; it is normalized internally.
    /// * `max_distance` - Furthest distance at which a hit is accepted.
    ///
    /// # Returns
    ///
    /// The first solid voxel entered by the ray, or `None` for a miss.
    pub fn raycast(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<RaycastHit> {
        let direction = direction.normalized();
        if direction == Vec3::zero() || max_distance < 0.0 {
            return None;
        }
        let mut cell = IVec3::new(
            origin.x.floor() as i32,
            origin.y.floor() as i32,
            origin.z.floor() as i32,
        );
        if self.block(cell).is_some_and(Block::is_solid) {
            return Some(RaycastHit {
                position: cell,
                normal: IVec3::default(),
                distance: 0.0,
            });
        }
        let step = IVec3::new(sign(direction.x), sign(direction.y), sign(direction.z));
        let delta = Vec3::new(
            axis_delta(direction.x),
            axis_delta(direction.y),
            axis_delta(direction.z),
        );
        let mut next = Vec3::new(
            initial_t(origin.x, direction.x, step.x),
            initial_t(origin.y, direction.y, step.y),
            initial_t(origin.z, direction.z, step.z),
        );
        loop {
            let (distance, normal) = if next.x <= next.y && next.x <= next.z {
                let value = next.x;
                next.x += delta.x;
                cell.x += step.x;
                (value, IVec3::new(-step.x, 0, 0))
            } else if next.y <= next.z {
                let value = next.y;
                next.y += delta.y;
                cell.y += step.y;
                (value, IVec3::new(0, -step.y, 0))
            } else {
                let value = next.z;
                next.z += delta.z;
                cell.z += step.z;
                (value, IVec3::new(0, 0, -step.z))
            };
            if distance > max_distance || !self.contains(cell) {
                return None;
            }
            if self.block(cell).is_some_and(Block::is_solid) {
                return Some(RaycastHit {
                    position: cell,
                    normal,
                    distance,
                });
            }
        }
    }

    fn index(&self, position: IVec3) -> Option<usize> {
        if self.contains(position) {
            Some(
                (position.y as usize * self.depth + position.z as usize) * self.width
                    + position.x as usize,
            )
        } else {
            None
        }
    }

    fn add_tree(&mut self, base: IVec3) {
        for y in 0..3 {
            self.set(base + IVec3::new(0, y, 0), Block::Wood);
        }
        for y in 2..5 {
            for z in -1..=1 {
                for x in -1..=1 {
                    let position = base + IVec3::new(x, y, z);
                    if self.block(position) == Some(Block::Air) {
                        self.set(position, Block::Leaves);
                    }
                }
            }
        }
    }
}

/// The result of a camera ray reaching a solid voxel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaycastHit {
    /// The solid block entered by the ray.
    pub position: IVec3,
    /// The outward normal of the face through which the block was entered.
    pub normal: IVec3,
    /// Distance from the ray origin to the entered face.
    pub distance: f32,
}

/// A render-ready vertex for one exposed voxel face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    /// Object-space location.
    pub position: Vec3,
    /// Unit outward face direction.
    pub normal: Vec3,
    /// Stable block tint.
    pub color: [f32; 4],
    /// Source material, useful for atlas-based renderers.
    pub block: Block,
}

/// Triangulated exposed faces of a [`World`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// Four vertices per exposed quad.
    pub vertices: Vec<Vertex>,
    /// Counter-clockwise triangle indices, stored as `u32` for graphics APIs.
    pub indices: Vec<u32>,
}

/// Builds a simple, non-greedy mesh containing only faces bordering air.
///
/// # Arguments
///
/// * `world` - Voxel data to turn into exposed quads.
///
/// # Returns
///
/// A triangle mesh with one quad for each air-adjacent block face.
pub fn mesh_world(world: &World) -> Mesh {
    const FACES: [(IVec3, [Vec3; 4]); 6] = [
        (
            IVec3::new(-1, 0, 0),
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 1.0, 1.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        ),
        (
            IVec3::new(1, 0, 0),
            [
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
            ],
        ),
        (
            IVec3::new(0, -1, 0),
            [
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 1.0),
            ],
        ),
        (
            IVec3::new(0, 1, 0),
            [
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
        ),
        (
            IVec3::new(0, 0, -1),
            [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
        ),
        (
            IVec3::new(0, 0, 1),
            [
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(0.0, 1.0, 1.0),
            ],
        ),
    ];
    let mut mesh = Mesh::default();
    for y in 0..world.height as i32 {
        for z in 0..world.depth as i32 {
            for x in 0..world.width as i32 {
                let position = IVec3::new(x, y, z);
                let Some(block) = world.block(position).filter(|block| block.is_solid()) else {
                    continue;
                };
                for (normal, corners) in FACES {
                    if world
                        .block(position + normal)
                        .is_none_or(|neighbor| !neighbor.is_solid())
                    {
                        let first = mesh.vertices.len() as u32;
                        let offset = Vec3::new(x as f32, y as f32, z as f32);
                        let normal = Vec3::new(normal.x as f32, normal.y as f32, normal.z as f32);
                        mesh.vertices.extend(corners.map(|corner| Vertex {
                            position: offset + corner,
                            normal,
                            color: block.color(),
                            block,
                        }));
                        mesh.indices.extend_from_slice(&[
                            first,
                            first + 1,
                            first + 2,
                            first,
                            first + 2,
                            first + 3,
                        ]);
                    }
                }
            }
        }
    }
    mesh
}

/// A first-person camera derived from a player's head position and rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// Camera world location.
    pub position: Vec3,
    /// Horizontal rotation in radians; zero looks along negative z.
    pub yaw: f32,
    /// Vertical rotation in radians, clamped just short of vertical.
    pub pitch: f32,
}

impl Camera {
    /// Returns the unit direction the camera faces.
    ///
    /// # Returns
    ///
    /// A normalized world-space viewing direction.
    pub fn forward(self) -> Vec3 {
        let horizontal = self.pitch.cos();
        Vec3::new(
            self.yaw.sin() * horizontal,
            self.pitch.sin(),
            -self.yaw.cos() * horizontal,
        )
        .normalized()
    }

    /// Returns the camera's right-hand view matrix.
    ///
    /// # Returns
    ///
    /// A column-major matrix that transforms world positions into camera space.
    pub fn view_matrix(self) -> Mat4 {
        Mat4::look_at_rh(
            self.position,
            self.position + self.forward(),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }
}

/// Frame input interpreted by [`Player::step`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PlayerInput {
    /// Move in the facing direction on the horizontal plane.
    pub forward: bool,
    /// Move opposite the facing direction on the horizontal plane.
    pub backward: bool,
    /// Strafe left.
    pub left: bool,
    /// Strafe right.
    pub right: bool,
    /// Jump when standing on solid ground.
    pub jump: bool,
}

/// The simulation state of a single first-person player.
#[derive(Clone, Debug, PartialEq)]
pub struct Player {
    /// Centre of the player's feet.
    pub position: Vec3,
    /// Current motion in world units per second.
    pub velocity: Vec3,
    /// Horizontal camera rotation in radians.
    pub yaw: f32,
    /// Vertical camera rotation in radians.
    pub pitch: f32,
    /// Whether downward movement touched a solid voxel in the latest step.
    pub grounded: bool,
    /// Hotbar slot selected by the application.
    pub selected_slot: usize,
    /// Block type placed by the selected hotbar slot.
    pub selected_block: Block,
}

impl Player {
    /// Creates a standing player at `position`.
    ///
    /// # Arguments
    ///
    /// * `position` - Feet-centre world position for the player.
    ///
    /// # Returns
    ///
    /// A player facing negative z and holding dirt in slot zero.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::zero(),
            yaw: 0.0,
            pitch: 0.0,
            grounded: false,
            selected_slot: 0,
            selected_block: Block::Dirt,
        }
    }

    /// Returns this player's eye camera.
    ///
    /// # Returns
    ///
    /// A camera positioned 1.62 units above the player's feet.
    pub fn camera(&self) -> Camera {
        Camera {
            position: self.position + Vec3::new(0.0, 1.62, 0.0),
            yaw: self.yaw,
            pitch: self.pitch,
        }
    }

    /// Changes camera rotation and clamps pitch to a usable first-person range.
    ///
    /// # Arguments
    ///
    /// * `yaw_delta` - Horizontal rotation change in radians.
    /// * `pitch_delta` - Vertical rotation change in radians.
    pub fn look(&mut self, yaw_delta: f32, pitch_delta: f32) {
        self.yaw += yaw_delta;
        self.pitch = (self.pitch + pitch_delta).clamp(-1.54, 1.54);
    }

    /// Selects a hotbar slot and the block it will place.
    ///
    /// # Arguments
    ///
    /// * `slot` - Application-defined hotbar slot index.
    /// * `block` - Solid material placed by that slot.
    pub fn select_block(&mut self, slot: usize, block: Block) {
        self.selected_slot = slot;
        self.selected_block = block;
    }

    /// Advances movement, gravity, jumping, and axis-aligned voxel collisions.
    ///
    /// # Arguments
    ///
    /// * `world` - Collision world sampled during movement.
    /// * `input` - Movement and jump state for this frame.
    /// * `delta_seconds` - Elapsed frame duration, clamped to at most 0.1 seconds.
    pub fn step(&mut self, world: &World, input: PlayerInput, delta_seconds: f32) {
        let dt = delta_seconds.clamp(0.0, 0.1);
        let forward = Vec3::new(self.yaw.sin(), 0.0, -self.yaw.cos());
        let right = Vec3::new(-forward.z, 0.0, forward.x);
        let mut move_direction = Vec3::zero();
        if input.forward {
            move_direction += forward;
        }
        if input.backward {
            move_direction += forward * -1.0;
        }
        if input.left {
            move_direction += right * -1.0;
        }
        if input.right {
            move_direction += right;
        }
        move_direction = move_direction.normalized() * 4.5;
        self.velocity.x = move_direction.x;
        self.velocity.z = move_direction.z;
        if input.jump && self.grounded {
            self.velocity.y = 8.0;
            self.grounded = false;
        }
        self.velocity.y -= 24.0 * dt;
        self.grounded = false;
        self.move_axis(world, 0, self.velocity.x * dt);
        self.move_axis(world, 1, self.velocity.y * dt);
        self.move_axis(world, 2, self.velocity.z * dt);
    }

    /// Destroys the solid voxel at the centre of the current camera ray.
    ///
    /// # Arguments
    ///
    /// * `world` - World whose hit block will be cleared.
    /// * `reach` - Maximum camera-ray distance.
    ///
    /// # Returns
    ///
    /// The destroyed hit, or `None` if no solid block is in reach.
    pub fn break_block(&self, world: &mut World, reach: f32) -> Option<RaycastHit> {
        let hit = world.raycast(self.camera().position, self.camera().forward(), reach)?;
        world.set(hit.position, Block::Air).then_some(hit)
    }

    /// Places the selected block on the air cell adjoining the current raycast hit.
    ///
    /// # Arguments
    ///
    /// * `world` - World receiving the new voxel.
    /// * `reach` - Maximum camera-ray distance.
    ///
    /// # Returns
    ///
    /// The placed coordinate, or `None` if there is no valid, unoccupied air cell.
    pub fn place_block(&self, world: &mut World, reach: f32) -> Option<IVec3> {
        if !self.selected_block.is_solid() {
            return None;
        }
        let hit = world.raycast(self.camera().position, self.camera().forward(), reach)?;
        let target = hit.position + hit.normal;
        if world.block(target) != Some(Block::Air) || self.aabb_intersects_voxel(target) {
            return None;
        }
        world.set(target, self.selected_block).then_some(target)
    }

    fn move_axis(&mut self, world: &World, axis: usize, amount: f32) {
        let steps = (amount.abs() / 0.05).ceil().max(1.0) as usize;
        let part = amount / steps as f32;
        for _ in 0..steps {
            let old = self.position;
            match axis {
                0 => self.position.x += part,
                1 => self.position.y += part,
                _ => self.position.z += part,
            }
            if self.intersects_world(world) {
                self.position = old;
                match axis {
                    0 => self.velocity.x = 0.0,
                    1 => {
                        if part < 0.0 {
                            self.grounded = true;
                        }
                        self.velocity.y = 0.0
                    }
                    _ => self.velocity.z = 0.0,
                }
                return;
            }
        }
    }

    fn intersects_world(&self, world: &World) -> bool {
        let min = Vec3::new(
            self.position.x - 0.3,
            self.position.y,
            self.position.z - 0.3,
        );
        let max = Vec3::new(
            self.position.x + 0.3,
            self.position.y + 1.8,
            self.position.z + 0.3,
        );
        for y in min.y.floor() as i32..=max.y.floor() as i32 {
            for z in min.z.floor() as i32..=max.z.floor() as i32 {
                for x in min.x.floor() as i32..=max.x.floor() as i32 {
                    if world
                        .block(IVec3::new(x, y, z))
                        .is_some_and(Block::is_solid)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn aabb_intersects_voxel(&self, voxel: IVec3) -> bool {
        let min = Vec3::new(
            self.position.x - 0.3,
            self.position.y,
            self.position.z - 0.3,
        );
        let max = Vec3::new(
            self.position.x + 0.3,
            self.position.y + 1.8,
            self.position.z + 0.3,
        );
        max.x > voxel.x as f32
            && min.x < voxel.x as f32 + 1.0
            && max.y > voxel.y as f32
            && min.y < voxel.y as f32 + 1.0
            && max.z > voxel.z as f32
            && min.z < voxel.z as f32 + 1.0
    }
}

/// A convenience bundle for the world and its sole player.
#[derive(Clone, Debug, PartialEq)]
pub struct Game {
    /// Mutable generated voxel world.
    pub world: World,
    /// The controllable player.
    pub player: Player,
}

impl Game {
    /// Generates `world` and positions a player at its natural spawn.
    ///
    /// # Arguments
    ///
    /// * `seed` - Stable terrain seed passed to [`World::generate`].
    ///
    /// # Returns
    ///
    /// A generated world and its correctly positioned starting player.
    pub fn generated(seed: u64) -> Self {
        let world = World::generate(seed);
        let player = Player::new(world.spawn_point());
        Self { world, player }
    }
}

fn hash(seed: u64, x: i32, z: i32) -> u64 {
    let mut value = seed
        ^ (x as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (z as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn terrain_height(seed: u64, x: i32, z: i32) -> i32 {
    let coarse = (hash(seed, x.div_euclid(4), z.div_euclid(4)) % 4) as i32;
    let detail = (hash(seed ^ 0x55AA, x, z) % 3) as i32 - 1;
    (6 + coarse + detail).clamp(3, 11)
}

fn sign(value: f32) -> i32 {
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}
fn axis_delta(value: f32) -> f32 {
    if value == 0.0 {
        f32::INFINITY
    } else {
        1.0 / value.abs()
    }
}
fn initial_t(origin: f32, direction: f32, step: i32) -> f32 {
    if direction == 0.0 {
        f32::INFINITY
    } else if step > 0 {
        (origin.floor() + 1.0 - origin) / direction.abs()
    } else {
        (origin - origin.floor()) / direction.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::{Block, IVec3, Mat4, Player, PlayerInput, Vec3, World, mesh_world};

    #[test]
    fn generation_is_deterministic() {
        assert_eq!(World::generate(42), World::generate(42));
        assert_ne!(World::generate(42), World::generate(43));
    }

    #[test]
    fn mesh_omits_shared_internal_faces() {
        let mut world = World::new(3, 3, 3);
        world.set(IVec3::new(1, 1, 1), Block::Stone);
        assert_eq!(mesh_world(&world).indices.len(), 36);
        world.set(IVec3::new(2, 1, 1), Block::Stone);
        let mesh = mesh_world(&world);
        assert_eq!(mesh.vertices.len(), 40);
        assert_eq!(mesh.indices.len(), 60);
    }

    #[test]
    fn raycast_reports_hit_and_entered_face_normal() {
        let mut world = World::new(4, 4, 4);
        world.set(IVec3::new(2, 1, 1), Block::Stone);
        let hit = world
            .raycast(Vec3::new(0.5, 1.5, 1.5), Vec3::new(1.0, 0.0, 0.0), 5.0)
            .unwrap();
        assert_eq!(hit.position, IVec3::new(2, 1, 1));
        assert_eq!(hit.normal, IVec3::new(-1, 0, 0));
        assert!((hit.distance - 1.5).abs() < 0.0001);
    }

    #[test]
    fn placement_rejects_outside_world_and_player_space() {
        let mut world = World::new(3, 4, 3);
        world.set(IVec3::new(0, 1, 1), Block::Stone);
        let mut outside = Player::new(Vec3::new(-0.5, 0.0, 1.5));
        outside.yaw = core::f32::consts::FRAC_PI_2;
        assert_eq!(outside.place_block(&mut world, 5.0), None);

        world.set(IVec3::new(1, 2, 0), Block::Stone);
        let player = Player::new(Vec3::new(1.5, 1.0, 1.5));
        assert_eq!(player.place_block(&mut world, 5.0), None);
    }

    #[test]
    fn player_lands_jumps_and_cannot_enter_wall() {
        let mut world = World::new(5, 5, 5);
        for x in 0..5 {
            for z in 0..5 {
                world.set(IVec3::new(x, 0, z), Block::Stone);
            }
        }
        world.set(IVec3::new(3, 1, 2), Block::Stone);
        let mut player = Player::new(Vec3::new(2.5, 1.01, 2.5));
        player.step(&world, PlayerInput::default(), 0.1);
        assert!(player.grounded);
        let ground_y = player.position.y;
        player.step(
            &world,
            PlayerInput {
                jump: true,
                ..PlayerInput::default()
            },
            0.05,
        );
        assert!(player.position.y > ground_y);
        player.position.y = 1.01;
        player.grounded = true;
        player.yaw = core::f32::consts::FRAC_PI_2;
        player.step(
            &world,
            PlayerInput {
                forward: true,
                ..PlayerInput::default()
            },
            0.1,
        );
        assert!(player.position.x < 2.71);
    }

    #[test]
    fn matrix_transforms_and_camera_direction_are_consistent() {
        let view = Mat4::look_at_rh(
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::zero(),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let transformed = view.transform_point(Vec3::zero());
        assert!((transformed.z + 2.0).abs() < 0.0001);
        assert_eq!(
            Player::new(Vec3::zero()).camera().forward(),
            Vec3::new(0.0, 0.0, -1.0)
        );

        let projection = Mat4::perspective_rh_gl(core::f32::consts::FRAC_PI_2, 1.0, 0.05, 128.0);
        let point_above_camera = Vec3::new(0.0, 1.0, -2.0);
        let gl_clip = projection.transform_point(point_above_camera);
        let sgfx_clip = projection
            .with_inverted_clip_y()
            .transform_point(point_above_camera);
        assert!(gl_clip.y > 0.0);
        assert!(sgfx_clip.y < 0.0);
        assert!((gl_clip.z - sgfx_clip.z).abs() < 0.0001);
    }
}
