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
    /// Non-colliding water used to fill terrain below sea level.
    Water,
    /// High-altitude or cold-biome snow cover.
    Snow,
    /// Non-solid light source that propagates warm block light.
    Torch,
}

impl Block {
    /// Returns whether this block fills space and collides with the player.
    ///
    /// # Returns
    ///
    /// `true` for blocks that stop movement, excluding air and water.
    pub const fn is_solid(self) -> bool {
        !matches!(self, Self::Air | Self::Water | Self::Torch)
    }

    /// Returns whether the block contributes faces to a terrain mesh.
    ///
    /// # Returns
    ///
    /// `true` for every material except [`Block::Air`]. Water is renderable even
    /// though it is not solid, allowing the world to have a visible sea without
    /// changing player collision behaviour.
    pub const fn is_renderable(self) -> bool {
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
            Self::Water => [0.08, 0.34, 0.72, 0.82],
            Self::Snow => [0.93, 0.95, 0.98, 1.0],
            Self::Torch => [0.95, 0.78, 0.45, 1.0],
        }
    }

    /// Returns how much sky light this block absorbs per propagated cell.
    ///
    /// # Returns
    ///
    /// An opacity from 0 (transparent air) to 15 (fully opaque solids).
    pub const fn light_opacity(self) -> u8 {
        match self {
            Self::Air => 0,
            Self::Water => 2,
            Self::Leaves => 1,
            Self::Torch => 0,
            _ => 15,
        }
    }

    /// Returns how much block light this block emits on its own.
    ///
    /// # Returns
    ///
    /// An emission level from 0 (no emission) to 15 (brightest source).
    pub const fn light_emission(self) -> u8 {
        match self {
            Self::Torch => 14,
            _ => 0,
        }
    }

    /// Returns whether camera rays stop on this block for editing.
    ///
    /// # Returns
    ///
    /// `true` for solid blocks and torches, which players can break.
    pub const fn is_targetable(self) -> bool {
        self.is_solid() || matches!(self, Self::Torch)
    }

    /// Returns whether this block hides the neighbouring block faces.
    ///
    /// # Returns
    ///
    /// `true` for fully opaque blocks; water, torches and air show the
    /// terrain behind them.
    pub const fn occludes(self) -> bool {
        self.is_solid()
    }

    /// Returns whether players can place this block from the hotbar.
    ///
    /// # Returns
    ///
    /// `true` for structural blocks and torches.
    pub const fn is_placeable(self) -> bool {
        self.is_solid() || matches!(self, Self::Torch)
    }
}

/// Height of the still-water surface in generated worlds.
pub const SEA_LEVEL: i32 = 5;

/// Horizontal size of one renderable terrain chunk, in voxels.
pub const CHUNK_SIZE: i32 = 16;

/// A finite, densely stored voxel world.
#[derive(Clone, Debug, PartialEq)]
pub struct World {
    width: usize,
    height: usize,
    depth: usize,
    blocks: Vec<Block>,
    direct_sunlight: Vec<u8>,
    sunlight: Vec<u8>,
    block_light: Vec<u8>,
    light_queue_pending: Vec<u8>,
    lighting_initialized: bool,
}

/// Horizontal extent whose baked mesh lighting changed after a block edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightUpdate {
    min_x: i32,
    max_x: i32,
    min_z: i32,
    max_z: i32,
}

impl LightUpdate {
    const fn empty() -> Self {
        Self {
            min_x: i32::MAX,
            max_x: i32::MIN,
            min_z: i32::MAX,
            max_z: i32::MIN,
        }
    }

    /// Return the inclusive horizontal bounds of changed light cells.
    pub const fn horizontal_bounds(self) -> Option<(i32, i32, i32, i32)> {
        if self.min_x > self.max_x || self.min_z > self.max_z {
            None
        } else {
            Some((self.min_x, self.max_x, self.min_z, self.max_z))
        }
    }

    fn include(&mut self, position: IVec3) {
        self.min_x = self.min_x.min(position.x);
        self.max_x = self.max_x.max(position.x);
        self.min_z = self.min_z.min(position.z);
        self.max_z = self.max_z.max(position.z);
    }
}

/// Non-occluding voxel space reachable from the world exterior or a viewer.
///
/// Far-terrain meshing may omit faces bordering a completely sealed component,
/// but darkness alone is not enough to prove that a face is hidden. This map
/// preserves dark tunnels and cave mouths by flood-filling topology instead of
/// relying on the finite light-propagation distance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibleSpace {
    width: usize,
    height: usize,
    depth: usize,
    reachable: Vec<u8>,
}

impl VisibleSpace {
    /// Flood-fill transparent space from every world boundary and `viewer`.
    ///
    /// The returned snapshot remains valid until block topology changes.
    /// Lighting-only updates do not invalidate it.
    pub fn from_world(world: &World, viewer: IVec3) -> Self {
        let mut visible = Self {
            width: world.width,
            height: world.height,
            depth: world.depth,
            reachable: vec![0; world.blocks.len()],
        };
        let mut queue = std::collections::VecDeque::new();

        if world.width > 0 {
            for y in 0..world.height as i32 {
                for z in 0..world.depth as i32 {
                    visible.enqueue(world, &mut queue, IVec3::new(0, y, z));
                    visible.enqueue(world, &mut queue, IVec3::new(world.width as i32 - 1, y, z));
                }
            }
        }
        if world.height > 0 {
            for z in 0..world.depth as i32 {
                for x in 0..world.width as i32 {
                    visible.enqueue(world, &mut queue, IVec3::new(x, 0, z));
                    visible.enqueue(world, &mut queue, IVec3::new(x, world.height as i32 - 1, z));
                }
            }
        }
        if world.depth > 0 {
            for y in 0..world.height as i32 {
                for x in 0..world.width as i32 {
                    visible.enqueue(world, &mut queue, IVec3::new(x, y, 0));
                    visible.enqueue(world, &mut queue, IVec3::new(x, y, world.depth as i32 - 1));
                }
            }
        }
        visible.enqueue(world, &mut queue, viewer);

        while let Some(index) = queue.pop_front() {
            let position = world.position_of(index);
            for offset in [
                IVec3::new(-1, 0, 0),
                IVec3::new(1, 0, 0),
                IVec3::new(0, -1, 0),
                IVec3::new(0, 1, 0),
                IVec3::new(0, 0, -1),
                IVec3::new(0, 0, 1),
            ] {
                visible.enqueue(world, &mut queue, position + offset);
            }
        }
        visible
    }

    /// Return whether `position` belongs to reachable transparent space.
    ///
    /// Positions outside the finite world are exterior and therefore visible.
    pub fn reaches(&self, position: IVec3) -> bool {
        let Some(index) = self.index(position) else {
            return true;
        };
        self.reachable
            .get(index)
            .is_some_and(|reachable| *reachable != 0)
    }

    fn matches(&self, world: &World) -> bool {
        self.width == world.width
            && self.height == world.height
            && self.depth == world.depth
            && self.reachable.len() == world.blocks.len()
    }

    fn enqueue(
        &mut self,
        world: &World,
        queue: &mut std::collections::VecDeque<usize>,
        position: IVec3,
    ) {
        let Some(index) = world.index(position) else {
            return;
        };
        if self.reachable[index] != 0 || world.blocks[index].occludes() {
            return;
        }
        self.reachable[index] = 1;
        queue.push_back(index);
    }

    fn index(&self, position: IVec3) -> Option<usize> {
        if position.x < 0
            || position.y < 0
            || position.z < 0
            || position.x >= self.width as i32
            || position.y >= self.height as i32
            || position.z >= self.depth as i32
        {
            return None;
        }
        Some(
            (position.y as usize * self.depth + position.z as usize) * self.width
                + position.x as usize,
        )
    }
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
        let cells = width.saturating_mul(height).saturating_mul(depth);
        Self {
            width,
            height,
            depth,
            blocks: vec![Block::Air; cells],
            direct_sunlight: vec![0; cells],
            sunlight: vec![0; cells],
            block_light: vec![0; cells],
            light_queue_pending: vec![0; cells],
            lighting_initialized: false,
        }
    }

    /// Creates the default world used by the game (192 by 48 by 192).
    ///
    /// # Returns
    ///
    /// A 192 by 48 by 192 all-air world.
    pub fn default_sized() -> Self {
        Self::new(192, 48, 192)
    }

    /// Generates deterministic terrain from `seed`, including coasts and trees.
    ///
    /// # Arguments
    ///
    /// * `seed` - Stable terrain seed.
    ///
    /// # Returns
    ///
    /// A default-sized world whose terrain is entirely determined by `seed`.
    pub fn generate(seed: u64) -> Self {
        Self::generate_sized(seed, 192, 48, 192)
    }

    /// Generates deterministic terrain for an explicitly sized world.
    ///
    /// # Arguments
    ///
    /// * `seed` - Stable terrain seed.
    /// * `width` - World extent on the x axis.
    /// * `height` - World extent on the y axis.
    /// * `depth` - World extent on the z axis.
    ///
    /// # Returns
    ///
    /// A generated world whose terrain is entirely determined by the inputs.
    pub fn generate_sized(seed: u64, width: usize, height: usize, depth: usize) -> Self {
        let mut world = Self::new(width, height, depth);
        for z in 0..world.depth as i32 {
            for x in 0..world.width as i32 {
                let column = terrain_column(seed, x, z);
                world.fill_column(seed, x, z, &column);
            }
        }
        world.populate_trees(seed);
        world.recompute_light();
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

    /// Returns the propagated sky light at `position`, from 0 to 15.
    ///
    /// # Arguments
    ///
    /// * `position` - Voxel coordinate to examine.
    ///
    /// # Returns
    ///
    /// The stored sky light, or full daylight outside the world bounds.
    pub fn sunlight(&self, position: IVec3) -> u8 {
        self.index(position)
            .map(|index| self.sunlight[index])
            .unwrap_or(15)
    }

    /// Returns the propagated block light at `position`, from 0 to 15.
    ///
    /// # Arguments
    ///
    /// * `position` - Voxel coordinate to examine.
    ///
    /// # Returns
    ///
    /// The stored block light, or darkness outside the world bounds.
    pub fn block_light(&self, position: IVec3) -> u8 {
        self.index(position)
            .map(|index| self.block_light[index])
            .unwrap_or(0)
    }

    /// Recomputes sky light for the whole world with a flood fill.
    ///
    /// Direct sky columns receive light 15 that travels downward without
    /// attenuation, then light spreads sideways (and upward) through air and
    /// translucent materials, losing opacity per crossed block. This is what
    /// makes overhangs, cave mouths and the shaded side of hills fall dark
    /// gradually instead of switching per face.
    pub fn recompute_light(&mut self) {
        self.recompute_sky_light();
        self.recompute_block_light_global();
        self.lighting_initialized = true;
    }

    fn recompute_sky_light(&mut self) {
        self.direct_sunlight.iter_mut().for_each(|light| *light = 0);
        self.sunlight.iter_mut().for_each(|light| *light = 0);
        let mut queue = std::collections::VecDeque::new();
        for z in 0..self.depth as i32 {
            for x in 0..self.width as i32 {
                let mut level = 15_u8;
                for y in (0..self.height as i32).rev() {
                    let position = IVec3::new(x, y, z);
                    let opacity = self.block(position).unwrap_or(Block::Air).light_opacity();
                    if opacity > 0 || level < 15 {
                        level = level.saturating_sub(opacity.max(1));
                    }
                    if level == 0 {
                        break;
                    }
                    if let Some(index) = self.index(position) {
                        self.direct_sunlight[index] = level;
                        self.sunlight[index] = level;
                        queue.push_back(index);
                    }
                }
            }
        }
        while let Some(index) = queue.pop_front() {
            let level = self.sunlight[index];
            if level <= 1 {
                continue;
            }
            let position = self.position_of(index);
            for offset in [
                IVec3::new(-1, 0, 0),
                IVec3::new(1, 0, 0),
                IVec3::new(0, -1, 0),
                IVec3::new(0, 0, -1),
                IVec3::new(0, 0, 1),
                IVec3::new(0, 1, 0),
            ] {
                let neighbor = position + offset;
                let Some(neighbor_index) = self.index(neighbor) else {
                    continue;
                };
                let opacity = self.blocks[neighbor_index].light_opacity().max(1);
                let spread = level.saturating_sub(opacity);
                if spread > self.sunlight[neighbor_index] {
                    self.sunlight[neighbor_index] = spread;
                    queue.push_back(neighbor_index);
                }
            }
        }
    }

    fn recompute_block_light_global(&mut self) {
        self.block_light.iter_mut().for_each(|light| *light = 0);
        let mut queue = std::collections::VecDeque::new();
        for index in 0..self.blocks.len() {
            let emission = self.blocks[index].light_emission();
            if emission > 0 {
                self.block_light[index] = emission;
                queue.push_back(index);
            }
        }
        self.propagate_block_light(&mut queue, i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    }

    /// Flood block light from `queue` within an optional region bound.
    ///
    /// When region bounds are supplied, propagation stops at the region edge;
    /// `usize::MAX`/`usize::MIN` sentinels disable the bound.
    fn propagate_block_light(
        &mut self,
        queue: &mut std::collections::VecDeque<usize>,
        max_x: i32,
        max_z: i32,
        min_x: i32,
        min_z: i32,
    ) {
        while let Some(index) = queue.pop_front() {
            let level = self.block_light[index];
            if level <= 1 {
                continue;
            }
            let position = self.position_of(index);
            for offset in [
                IVec3::new(-1, 0, 0),
                IVec3::new(1, 0, 0),
                IVec3::new(0, -1, 0),
                IVec3::new(0, 1, 0),
                IVec3::new(0, 0, -1),
                IVec3::new(0, 0, 1),
            ] {
                let neighbor = position + offset;
                if neighbor.x > max_x
                    || neighbor.x < min_x
                    || neighbor.z > max_z
                    || neighbor.z < min_z
                {
                    continue;
                }
                let Some(neighbor_index) = self.index(neighbor) else {
                    continue;
                };
                let opacity = self.blocks[neighbor_index].light_opacity().max(1);
                let spread = level.saturating_sub(opacity);
                if spread > self.block_light[neighbor_index] {
                    self.block_light[neighbor_index] = spread;
                    queue.push_back(neighbor_index);
                }
            }
        }
    }

    fn fill_column(&mut self, seed: u64, x: i32, z: i32, column: &TerrainColumn) {
        for y in 0..=column.height {
            let mut block = if y == column.height {
                column.surface
            } else if y + 4 >= column.height {
                column.subsurface
            } else {
                Block::Stone
            };
            // Carve winding cave tunnels through the rock body, but keep a
            // sealed floor under the sea so oceans do not drain into caves.
            if y >= 2
                && y <= column.height
                && column.height > SEA_LEVEL + 2
                && is_cave(seed, x, y, z)
            {
                block = Block::Air;
            }
            self.set(IVec3::new(x, y, z), block);
        }
        for y in column.height + 1..=SEA_LEVEL.min(self.height as i32 - 1) {
            self.set(IVec3::new(x, y, z), Block::Water);
        }
    }

    /// Incrementally refresh both light channels after one block edit.
    ///
    /// # Arguments
    ///
    /// * `center` - Edited voxel around which light is refreshed.
    /// * `radius` - Retained for API compatibility; propagation now continues
    ///   until no light value changes and is not clipped to a fixed radius.
    ///
    /// # Returns
    ///
    /// Horizontal bounds of cells whose final sky or block light changed.
    pub fn recompute_light_around(&mut self, center: IVec3, _radius: i32) -> LightUpdate {
        self.recompute_light_after_edit(center)
    }

    /// Incrementally refresh both light channels after one block edit.
    pub fn recompute_light_after_edit(&mut self, center: IVec3) -> LightUpdate {
        let Some(center_index) = self.index(center) else {
            return LightUpdate::empty();
        };
        if !self.lighting_initialized {
            self.recompute_light();
            let mut changed = LightUpdate::empty();
            if self.width > 0 && self.depth > 0 {
                changed.include(IVec3::new(0, 0, 0));
                changed.include(IVec3::new(self.width as i32 - 1, 0, self.depth as i32 - 1));
            }
            return changed;
        }
        debug_assert!(self.light_queue_pending.iter().all(|pending| *pending == 0));

        let mut changed = LightUpdate::empty();
        let mut queue = std::collections::VecDeque::new();
        self.refresh_direct_sunlight_after_edit(center, &mut queue);
        Self::enqueue_light(&mut queue, &mut self.light_queue_pending, center_index);
        self.relax_sunlight(&mut queue, &mut changed);

        Self::enqueue_light(&mut queue, &mut self.light_queue_pending, center_index);
        self.relax_block_light(&mut queue, &mut changed);
        debug_assert!(self.light_queue_pending.iter().all(|pending| *pending == 0));
        changed
    }

    fn refresh_direct_sunlight_after_edit(
        &mut self,
        center: IVec3,
        queue: &mut std::collections::VecDeque<usize>,
    ) {
        // Direct sky only travels down a column. Everything above the edit is
        // unchanged, so resume from the cached value immediately above it and
        // stop as soon as the recurrence rejoins the old cached column.
        let mut level = self
            .index(center + IVec3::new(0, 1, 0))
            .map(|index| self.direct_sunlight[index])
            .unwrap_or(15);
        for y in (0..=center.y).rev() {
            let position = IVec3::new(center.x, y, center.z);
            let index = self
                .index(position)
                .expect("edited direct-sky column remains inside the world");
            if level > 0 {
                let opacity = self.blocks[index].light_opacity();
                if opacity > 0 || level < 15 {
                    level = level.saturating_sub(opacity.max(1));
                }
            }
            if self.direct_sunlight[index] == level {
                break;
            }
            self.direct_sunlight[index] = level;
            Self::enqueue_light(queue, &mut self.light_queue_pending, index);
        }
    }

    fn relax_sunlight(
        &mut self,
        queue: &mut std::collections::VecDeque<usize>,
        changed: &mut LightUpdate,
    ) {
        while let Some(index) = queue.pop_front() {
            self.light_queue_pending[index] = 0;
            let level = self.expected_sunlight(index);
            if self.sunlight[index] == level {
                continue;
            }
            self.sunlight[index] = level;
            let position = self.position_of(index);
            changed.include(position);
            for offset in [
                IVec3::new(-1, 0, 0),
                IVec3::new(1, 0, 0),
                IVec3::new(0, -1, 0),
                IVec3::new(0, 1, 0),
                IVec3::new(0, 0, -1),
                IVec3::new(0, 0, 1),
            ] {
                let Some(neighbor) = self.index(position + offset) else {
                    continue;
                };
                Self::enqueue_light(queue, &mut self.light_queue_pending, neighbor);
            }
        }
    }

    fn relax_block_light(
        &mut self,
        queue: &mut std::collections::VecDeque<usize>,
        changed: &mut LightUpdate,
    ) {
        while let Some(index) = queue.pop_front() {
            self.light_queue_pending[index] = 0;
            let level = self.expected_block_light(index);
            if self.block_light[index] == level {
                continue;
            }
            self.block_light[index] = level;
            let position = self.position_of(index);
            changed.include(position);
            for offset in [
                IVec3::new(-1, 0, 0),
                IVec3::new(1, 0, 0),
                IVec3::new(0, -1, 0),
                IVec3::new(0, 1, 0),
                IVec3::new(0, 0, -1),
                IVec3::new(0, 0, 1),
            ] {
                let Some(neighbor) = self.index(position + offset) else {
                    continue;
                };
                Self::enqueue_light(queue, &mut self.light_queue_pending, neighbor);
            }
        }
    }

    fn expected_sunlight(&self, index: usize) -> u8 {
        let attenuation = self.blocks[index].light_opacity().max(1);
        let position = self.position_of(index);
        let mut level = self.direct_sunlight[index];
        for offset in [
            IVec3::new(-1, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(0, 0, -1),
            IVec3::new(0, 0, 1),
        ] {
            let Some(neighbor) = self.index(position + offset) else {
                continue;
            };
            level = level.max(self.sunlight[neighbor].saturating_sub(attenuation));
        }
        level
    }

    fn expected_block_light(&self, index: usize) -> u8 {
        let attenuation = self.blocks[index].light_opacity().max(1);
        let position = self.position_of(index);
        let mut level = self.blocks[index].light_emission();
        for offset in [
            IVec3::new(-1, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(0, 0, -1),
            IVec3::new(0, 0, 1),
        ] {
            let Some(neighbor) = self.index(position + offset) else {
                continue;
            };
            level = level.max(self.block_light[neighbor].saturating_sub(attenuation));
        }
        level
    }

    fn enqueue_light(
        queue: &mut std::collections::VecDeque<usize>,
        pending: &mut [u8],
        index: usize,
    ) {
        if pending[index] == 0 {
            pending[index] = 1;
            queue.push_back(index);
        }
    }

    fn position_of(&self, index: usize) -> IVec3 {
        let width = self.width;
        let depth = self.depth;
        IVec3::new(
            (index % width) as i32,
            (index / (width * depth)) as i32,
            ((index / width) % depth) as i32,
        )
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
        if self.block(cell).is_some_and(Block::is_targetable) {
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
            if self.block(cell).is_some_and(Block::is_targetable) {
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

    fn populate_trees(&mut self, seed: u64) {
        const TREE_CELL_SIZE: i32 = 6;
        const CANOPY_RADIUS: i32 = 2;
        let cells_x = (self.width as i32 + TREE_CELL_SIZE - 1) / TREE_CELL_SIZE;
        let cells_z = (self.depth as i32 + TREE_CELL_SIZE - 1) / TREE_CELL_SIZE;

        for cell_z in 0..cells_z {
            for cell_x in 0..cells_x {
                let choice = hash(seed ^ 0xA1B2_C3D4_E5F6_0718, cell_x, cell_z);
                let x = cell_x * TREE_CELL_SIZE + ((choice >> 8) % TREE_CELL_SIZE as u64) as i32;
                let z = cell_z * TREE_CELL_SIZE + ((choice >> 16) % TREE_CELL_SIZE as u64) as i32;
                if x < CANOPY_RADIUS
                    || z < CANOPY_RADIUS
                    || x >= self.width as i32 - CANOPY_RADIUS
                    || z >= self.depth as i32 - CANOPY_RADIUS
                {
                    continue;
                }

                let column = terrain_column(seed, x, z);
                // Forest biomes roll one candidate per coarse cell against
                // their climate-derived density, which clumps woodland the way
                // real seed dispersal does instead of scattering it evenly.
                if (choice % 1000) as f32 / 1000.0 >= column.forest_density {
                    continue;
                }

                let ground = column.height;
                let trunk_height = 4 + ((choice >> 24) & 3) as i32;
                if ground <= SEA_LEVEL
                    || ground + trunk_height + 2 >= self.height as i32
                    || self.block(IVec3::new(x, ground, z)) != Some(Block::Grass)
                    || !self.tree_site_is_gentle(seed, x, z)
                {
                    continue;
                }
                self.add_tree(IVec3::new(x, ground + 1, z), trunk_height);
            }
        }
    }

    fn tree_site_is_gentle(&self, seed: u64, x: i32, z: i32) -> bool {
        let height = terrain_height(seed, x, z);
        [
            IVec3::new(-1, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(0, 0, -1),
            IVec3::new(0, 0, 1),
        ]
        .into_iter()
        .all(|offset| (terrain_height(seed, x + offset.x, z + offset.z) - height).abs() <= 1)
    }

    fn add_tree(&mut self, base: IVec3, trunk_height: i32) {
        for y in 0..trunk_height {
            self.set(base + IVec3::new(0, y, 0), Block::Wood);
        }

        let canopy_base = trunk_height - 2;
        for y in canopy_base..=trunk_height + 1 {
            let radius: i32 = match y - canopy_base {
                0 | 3 => 1,
                _ => 2,
            };
            for z in -radius..=radius {
                for x in -radius..=radius {
                    // Round the broad canopy corners and leave its trunk visible.
                    if x.abs() + z.abs() > radius + 1 || (x == 0 && z == 0 && y < trunk_height) {
                        continue;
                    }
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
    /// Normalized texture coordinate in the built-in four-by-three block atlas.
    pub atlas_uv: [f32; 2],
    /// Corner ambient-occlusion multiplier, from dark (`0.46`) to unoccluded (`1.0`).
    pub ambient_occlusion: f32,
    /// Directional sky-light multiplier for this corner, from `0.0` to `1.0`.
    pub light: f32,
    /// Propagated block-light multiplier for this corner, from `0.0` to `1.0`.
    pub torch_light: f32,
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
    mesh_region(
        world,
        0,
        world.width as i32 - 1,
        0,
        world.depth as i32 - 1,
        None,
    )
}

/// Builds the exposed-face mesh of one horizontal terrain chunk.
///
/// # Arguments
///
/// * `world` - Voxel data containing the chunk.
/// * `chunk_x` - Chunk coordinate on the x axis in `[CHUNK_SIZE]` units.
/// * `chunk_z` - Chunk coordinate on the z axis in `[CHUNK_SIZE]` units.
///
/// # Returns
///
/// The portion of the world mesh lying inside the requested chunk.
pub fn mesh_chunk(world: &World, chunk_x: i32, chunk_z: i32) -> Mesh {
    mesh_region(
        world,
        chunk_x * CHUNK_SIZE,
        (chunk_x + 1) * CHUNK_SIZE - 1,
        chunk_z * CHUNK_SIZE,
        (chunk_z + 1) * CHUNK_SIZE - 1,
        None,
    )
}

/// Builds an occlusion-safe reduced chunk mesh for the merged far ring.
///
/// Faces bordering a transparent component disconnected from both the world
/// exterior and the viewer are omitted. Dark but reachable tunnels remain in
/// the mesh; only topologically sealed space is removed.
///
/// # Arguments
///
/// * `world` - Voxel data containing the chunk.
/// * `chunk_x` - Chunk coordinate on the x axis in `[CHUNK_SIZE]` units.
/// * `chunk_z` - Chunk coordinate on the z axis in `[CHUNK_SIZE]` units.
/// * `visible_space` - Reachability snapshot for the current world topology.
///
/// # Returns
///
/// The chunk mesh without geometry facing topologically sealed space.
pub fn mesh_chunk_lod(
    world: &World,
    chunk_x: i32,
    chunk_z: i32,
    visible_space: &VisibleSpace,
) -> Mesh {
    // A stale or foreign map must fail open: extra geometry is safe, holes are
    // not. The frontend normally replaces the snapshot after every edit.
    let visible_space = visible_space.matches(world).then_some(visible_space);
    mesh_region(
        world,
        chunk_x * CHUNK_SIZE,
        (chunk_x + 1) * CHUNK_SIZE - 1,
        chunk_z * CHUNK_SIZE,
        (chunk_z + 1) * CHUNK_SIZE - 1,
        visible_space,
    )
}

fn mesh_region(
    world: &World,
    min_x: i32,
    max_x: i32,
    min_z: i32,
    max_z: i32,
    visible_space: Option<&VisibleSpace>,
) -> Mesh {
    const FACE_UVS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
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
        for z in min_z.max(0)..=max_z.min(world.depth as i32 - 1) {
            for x in min_x.max(0)..=max_x.min(world.width as i32 - 1) {
                let position = IVec3::new(x, y, z);
                let Some(block) = world.block(position).filter(|block| block.is_renderable())
                else {
                    continue;
                };
                // Torches render as a thin free-standing model that neither
                // occludes neighbours nor participates in face culling.
                if block == Block::Torch {
                    append_torch_mesh(world, &mut mesh, position);
                    continue;
                }
                for (normal, corners) in FACES {
                    let neighbor = position + normal;
                    let face_visible = world.block(neighbor).is_none_or(|neighbor| {
                        // Water and torches are transparent: terrain shows
                        // through them, but water hides its own internal faces.
                        !neighbor.occludes() && !(block == Block::Water && neighbor == Block::Water)
                    });
                    if face_visible {
                        if visible_space.is_some_and(|space| !space.reaches(neighbor)) {
                            continue;
                        }
                        // Still water sits slightly below a full block, the
                        // classic inset surface that makes shorelines readable.
                        let inset_surface = block == Block::Water
                            && world
                                .block(position + IVec3::new(0, 1, 0))
                                .is_none_or(|neighbor| !neighbor.is_renderable());
                        let surface_y = if inset_surface { 0.875 } else { 1.0 };
                        let first = mesh.vertices.len() as u32;
                        let offset = Vec3::new(x as f32, y as f32, z as f32);
                        let normal = Vec3::new(normal.x as f32, normal.y as f32, normal.z as f32);
                        for (corner, uv) in corners.into_iter().zip(FACE_UVS) {
                            let corner = if corner.y >= 1.0 {
                                Vec3::new(corner.x, surface_y, corner.z)
                            } else {
                                corner
                            };
                            let light = vertex_light(world, position, normal, corner);
                            mesh.vertices.push(Vertex {
                                position: offset + corner,
                                normal,
                                color: block.color(),
                                block,
                                atlas_uv: atlas_uv(block, normal, uv),
                                ambient_occlusion: vertex_ambient_occlusion(
                                    world, position, normal, corner,
                                ),
                                light: light[0],
                                torch_light: light[1],
                            });
                        }
                        mesh.indices.extend_from_slice(&[
                            first,
                            first + 1,
                            first + 2,
                            first,
                            first + 2,
                            first + 3,
                        ]);
                        // Emit a down-facing copy of an exposed water surface so
                        // it remains visible while swimming underneath it.
                        if inset_surface && normal.y > 0.0 {
                            let base = mesh.vertices.len() as u32;
                            let mut underside = Vec::with_capacity(4);
                            for vertex in &mesh.vertices[first as usize..first as usize + 4] {
                                underside.push(Vertex {
                                    normal: Vec3::new(0.0, -1.0, 0.0),
                                    ..*vertex
                                });
                            }
                            mesh.vertices.extend(underside);
                            mesh.indices.extend_from_slice(&[
                                base + 2,
                                base + 1,
                                base,
                                base + 3,
                                base + 2,
                                base,
                            ]);
                        }
                    }
                }
            }
        }
    }
    mesh
}

fn atlas_uv(block: Block, normal: Vec3, local_uv: [f32; 2]) -> [f32; 2] {
    const ATLAS_COLUMNS: f32 = 4.0;
    const ATLAS_ROWS: f32 = 4.0;
    let tile = match block {
        Block::Air => [0.0, 0.0],
        Block::Grass if normal.y > 0.0 => [0.0, 0.0],
        Block::Grass if normal.y < 0.0 => [2.0, 0.0],
        Block::Grass => [1.0, 0.0],
        Block::Dirt => [2.0, 0.0],
        Block::Stone => [3.0, 0.0],
        Block::Wood if normal.y == 0.0 => [0.0, 1.0],
        Block::Wood => [1.0, 1.0],
        Block::Leaves => [2.0, 1.0],
        Block::Sand => [3.0, 1.0],
        Block::Water => [0.0, 2.0],
        Block::Snow => [1.0, 2.0],
        // The torch samples a narrow sub-region of its tile so the thin
        // model shows the stick texture rather than the tile average.
        Block::Torch => [2.0, 2.0],
    };
    let local_uv = if block == Block::Torch {
        [0.42 + local_uv[0] * 0.16, local_uv[1] * 0.7]
    } else {
        local_uv
    };
    [
        (tile[0] + local_uv[0]) / ATLAS_COLUMNS,
        (tile[1] + local_uv[1]) / ATLAS_ROWS,
    ]
}

/// Append the thin torch model: a small box with an unoccluded appearance.
///
/// The model never culls against neighbours and skips its bottom face so it
/// reads as a stick standing on the ground.
fn append_torch_mesh(world: &World, mesh: &mut Mesh, position: IVec3) {
    const FACE_UVS_TORCH: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    const MIN: Vec3 = Vec3::new(0.4375, 0.0, 0.4375);
    const MAX: Vec3 = Vec3::new(0.5625, 0.625, 0.5625);
    let offset = Vec3::new(position.x as f32, position.y as f32, position.z as f32);
    let faces: [(IVec3, [Vec3; 4]); 5] = [
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
        (
            IVec3::new(0, 1, 0),
            [
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
        ),
    ];
    for (normal, corners) in faces {
        let first = mesh.vertices.len() as u32;
        let normal_vec = Vec3::new(normal.x as f32, normal.y as f32, normal.z as f32);
        for (corner, uv) in corners.into_iter().zip(FACE_UVS_TORCH) {
            let scaled = Vec3::new(
                MIN.x + corner.x * (MAX.x - MIN.x),
                MIN.y + corner.y * (MAX.y - MIN.y),
                MIN.z + corner.z * (MAX.z - MIN.z),
            );
            let light = vertex_light(world, position, normal_vec, corner);
            mesh.vertices.push(Vertex {
                position: offset + scaled,
                normal: normal_vec,
                color: Block::Torch.color(),
                block: Block::Torch,
                atlas_uv: atlas_uv(Block::Torch, normal_vec, uv),
                ambient_occlusion: 1.0,
                light: light[0],
                torch_light: light[1],
            });
        }
        mesh.indices
            .extend_from_slice(&[first, first + 1, first + 2, first, first + 2, first + 3]);
    }
}

fn vertex_ambient_occlusion(world: &World, position: IVec3, normal: Vec3, corner: Vec3) -> f32 {
    let normal = IVec3::new(normal.x as i32, normal.y as i32, normal.z as i32);
    let mut sides = [false; 2];
    let mut side_count = 0;
    let mut diagonal = IVec3::default();
    for (axis, corner_component) in [(0, corner.x), (1, corner.y), (2, corner.z)] {
        if [normal.x, normal.y, normal.z][axis] != 0 {
            continue;
        }
        let direction = if corner_component < 0.5 { -1 } else { 1 };
        let offset = match axis {
            0 => IVec3::new(direction, 0, 0),
            1 => IVec3::new(0, direction, 0),
            _ => IVec3::new(0, 0, direction),
        };
        sides[side_count] = world
            .block(position + normal + offset)
            .is_some_and(Block::is_solid);
        diagonal = diagonal + offset;
        side_count += 1;
    }
    debug_assert_eq!(side_count, 2);
    let corner_is_occluded = world
        .block(position + normal + diagonal)
        .is_some_and(Block::is_solid);
    let occluders = if sides[0] && sides[1] {
        3
    } else {
        sides.into_iter().filter(|side| *side).count() as i32 + corner_is_occluded as i32
    };
    1.0 - occluders as f32 * 0.18
}

fn vertex_light(world: &World, position: IVec3, normal: Vec3, corner: Vec3) -> [f32; 2] {
    // Minecraft-style smooth lighting: each corner averages the propagated
    // sky and block-light values in the four cells sharing that corner. The
    // light arrays are already flood-filled at integer levels (0..15), so the
    // renderer only interpolates the baked field; it never invents a local
    // per-block shadow or a minimum ambient value here.
    let normal_axis = if normal.x != 0.0 {
        0
    } else if normal.y != 0.0 {
        1
    } else {
        2
    };
    let normal_offset = IVec3::new(normal.x as i32, normal.y as i32, normal.z as i32);
    let mut tangent_ranges = [(0_i32, 0_i32); 2];
    let mut tangent = 0;
    for (axis, corner_component) in [(0, corner.x), (1, corner.y), (2, corner.z)] {
        if axis == normal_axis {
            continue;
        }
        tangent_ranges[tangent] = if corner_component >= 1.0 {
            (0, 1)
        } else {
            (-1, 0)
        };
        tangent += 1;
    }
    let mut sum = 0.0;
    let mut torch_sum = 0.0_f32;
    let mut samples = 0.0_f32;
    for first in [tangent_ranges[0].0, tangent_ranges[0].1] {
        for second in [tangent_ranges[1].0, tangent_ranges[1].1] {
            let mut offset = normal_offset;
            let components = [first, second];
            let mut index = 0;
            for (axis, _) in [(0, corner.x), (1, corner.y), (2, corner.z)] {
                if axis == normal_axis {
                    continue;
                }
                let value = components[index];
                offset = match axis {
                    0 => IVec3::new(value, offset.y, offset.z),
                    1 => IVec3::new(offset.x, value, offset.z),
                    _ => IVec3::new(offset.x, offset.y, value),
                };
                index += 1;
            }
            let cell = position + offset;
            let sky = if world.block(cell).is_some_and(Block::is_solid) {
                0.0
            } else {
                world.sunlight(cell) as f32 / 15.0
            };
            let torch = if world.block(cell).is_some_and(Block::is_solid) {
                0.0
            } else {
                world.block_light(cell) as f32 / 15.0
            };
            sum += sky;
            torch_sum += torch;
            samples += 1.0;
        }
    }
    let sky_light = sum / samples.max(1.0);
    let torch_light = torch_sum / samples.max(1.0);
    [sky_light, torch_light]
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
        self.break_block_with_light_update(world, reach)
            .map(|(hit, _)| hit)
    }

    /// Destroy a targeted block and return the exact changed-light extent.
    pub fn break_block_with_light_update(
        &self,
        world: &mut World,
        reach: f32,
    ) -> Option<(RaycastHit, LightUpdate)> {
        let hit = world.raycast(self.camera().position, self.camera().forward(), reach)?;
        if world.set(hit.position, Block::Air) {
            let light_update = world.recompute_light_after_edit(hit.position);
            Some((hit, light_update))
        } else {
            None
        }
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
        self.place_block_with_light_update(world, reach)
            .map(|(position, _)| position)
    }

    /// Place the selected block and return the exact changed-light extent.
    pub fn place_block_with_light_update(
        &self,
        world: &mut World,
        reach: f32,
    ) -> Option<(IVec3, LightUpdate)> {
        if !self.selected_block.is_placeable() {
            return None;
        }
        let hit = world.raycast(self.camera().position, self.camera().forward(), reach)?;
        let target = hit.position + hit.normal;
        if world.block(target) != Some(Block::Air)
            || (self.selected_block.is_solid() && self.aabb_intersects_voxel(target))
        {
            return None;
        }
        if world.set(target, self.selected_block) {
            let light_update = world.recompute_light_after_edit(target);
            Some((target, light_update))
        } else {
            None
        }
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

/// The height of the snow line; peaks above this are covered in snow.
const SNOW_LINE: i32 = 31;

/// A generated vertical terrain profile plus its surface materials.
#[derive(Clone, Copy, Debug)]
struct TerrainColumn {
    height: i32,
    surface: Block,
    subsurface: Block,
    forest_density: f32,
}

/// A climate classification that selects surface materials and tree density.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Biome {
    Ocean,
    Beach,
    Plains,
    Forest,
    Desert,
    Mountains,
    SnowPeaks,
}

fn terrain_column(seed: u64, x: i32, z: i32) -> TerrainColumn {
    let height = terrain_height(seed, x, z);
    let temperature = fbm(seed ^ 0x1D3E_5A6B_7C8D_9E0F, x, z, 40, 2);
    let moisture = fbm(seed ^ 0x2C4F_6E80_91A2_B3C4, x, z, 32, 2);
    let (biome, surface, subsurface) = if height <= SEA_LEVEL - 2 {
        (Biome::Ocean, Block::Sand, Block::Sand)
    } else if height <= SEA_LEVEL + 1 {
        (Biome::Beach, Block::Sand, Block::Sand)
    } else if height >= SNOW_LINE {
        (Biome::SnowPeaks, Block::Snow, Block::Stone)
    } else if height >= SNOW_LINE - 6 {
        (Biome::Mountains, Block::Stone, Block::Stone)
    } else if moisture < -0.32 && temperature > 0.05 {
        (Biome::Desert, Block::Sand, Block::Sand)
    } else {
        let forest = moisture > 0.12;
        (
            if forest { Biome::Forest } else { Biome::Plains },
            Block::Grass,
            Block::Dirt,
        )
    };
    let forest_density = match biome {
        Biome::Forest => 0.62,
        Biome::Plains => 0.16,
        _ => 0.0,
    };
    TerrainColumn {
        height,
        surface,
        subsurface,
        forest_density,
    }
}

/// Returns the terrain surface height at a column, from 1 to 43.
pub fn terrain_height(seed: u64, x: i32, z: i32) -> i32 {
    // Domain warping bends the whole noise field so ranges and coastlines
    // curve naturally instead of forming axis-aligned blobs.
    let warp_x = fbm(seed ^ 0x6A09_E667_F3BC_C908, x, z, 36, 2) * 22.0;
    let warp_z = fbm(seed ^ 0xBB67_AE85_84CA_A73B, x, z, 36, 2) * 22.0;
    let wx = x as f32 + warp_x;
    let wz = z as f32 + warp_z;

    // Broad continents establish oceans, shelves and land masses.
    let continent = fbm_at(seed ^ 0xF135_7AEA_2E62_A9C5, wx, wz, 56, 3);
    // Rolling hills composed of several octaves of warped value noise.
    let hills = fbm_at(seed ^ 0x8D58_AC26_AA16_3A41, wx, wz, 14, 4) * 5.5;
    // Ridged noise creates sharp alpine crests where a regional mask allows.
    let mountain_mask = smoothstep(
        (((fbm_at(seed ^ 0x7B29_4D0F_91D7_05B3, wx, wz, 64, 2) + 1.0) * 0.5 - 0.30) / 0.30)
            .clamp(0.0, 1.0),
    );
    let ridge = (1.0 - fbm_at(seed ^ 0x4CF5_AD43_2745_937F, wx, wz, 28, 3).abs()).powi(3) * 32.0;

    let elevation = SEA_LEVEL as f32 + 2.2 + continent * 7.0 + hills + ridge * mountain_mask;
    elevation.round().clamp(1.0, 43.0) as i32
}

/// Returns whether a 3D noise sample opens a cave tunnel at this cell.
fn is_cave(seed: u64, x: i32, y: i32, z: i32) -> bool {
    // Two independent ridged fields intersected produce winding tunnels
    // rather than noisy blobs: a cave exists only where both are near zero.
    let tunnel_a = value_noise_3d(seed ^ 0x3A7B_2D4E_6F80_91A2, x, y * 2, z, 18);
    let tunnel_b = value_noise_3d(seed ^ 0x5C9E_1F30_7153_B4D6, x, y * 2, z, 18);
    let room = value_noise_3d(seed ^ 0x6E80_91A2_B3C4_D5E6, x, y, z, 12);
    tunnel_a.abs() < 0.085 && tunnel_b.abs() < 0.085 || room > 0.72
}

fn fbm(seed: u64, x: i32, z: i32, wavelength: i32, octaves: u32) -> f32 {
    fbm_at(seed, x as f32, z as f32, wavelength, octaves)
}

fn fbm_at(seed: u64, x: f32, z: f32, wavelength: i32, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut total = 0.0;
    let mut scale = wavelength;
    for octave in 0..octaves {
        sum += value_noise_f(
            seed ^ (octave as u64).wrapping_mul(0x9E37_79B9_7F4A_C15),
            x,
            z,
            scale,
        ) * amplitude;
        total += amplitude;
        amplitude *= 0.5;
        scale = (scale + 1) / 2;
    }
    sum / total
}

fn value_noise_f(seed: u64, x: f32, z: f32, wavelength: i32) -> f32 {
    debug_assert!(wavelength > 0);
    let grid_x = x.div_euclid(wavelength as f32);
    let grid_z = z.div_euclid(wavelength as f32);
    let local_x = x.rem_euclid(wavelength as f32) / wavelength as f32;
    let local_z = z.rem_euclid(wavelength as f32) / wavelength as f32;
    let blend_x = smoothstep(local_x);
    let blend_z = smoothstep(local_z);
    let north = lerp(
        hash_to_unit(seed, grid_x as i32, grid_z as i32),
        hash_to_unit(seed, grid_x as i32 + 1, grid_z as i32),
        blend_x,
    );
    let south = lerp(
        hash_to_unit(seed, grid_x as i32, grid_z as i32 + 1),
        hash_to_unit(seed, grid_x as i32 + 1, grid_z as i32 + 1),
        blend_x,
    );
    lerp(north, south, blend_z)
}

fn value_noise_3d(seed: u64, x: i32, y: i32, z: i32, wavelength: i32) -> f32 {
    debug_assert!(wavelength > 0);
    let grid_x = x.div_euclid(wavelength);
    let grid_y = y.div_euclid(wavelength);
    let grid_z = z.div_euclid(wavelength);
    let local_x = smoothstep(x.rem_euclid(wavelength) as f32 / wavelength as f32);
    let local_y = smoothstep(y.rem_euclid(wavelength) as f32 / wavelength as f32);
    let local_z = smoothstep(z.rem_euclid(wavelength) as f32 / wavelength as f32);
    let mut corners = [0.0_f32; 8];
    let mut corner = 0;
    for dy in 0..2 {
        for dz in 0..2 {
            for dx in 0..2 {
                corners[corner] =
                    hash_to_unit(seed, grid_x + dx, grid_y * 31 + dy * 7 + grid_z + dz);
                corner += 1;
            }
        }
    }
    let mut along_x = [0.0_f32; 4];
    for layer in 0..4 {
        along_x[layer] = lerp(corners[layer * 2], corners[layer * 2 + 1], local_x);
    }
    let mut along_z = [0.0_f32; 2];
    for layer in 0..2 {
        along_z[layer] = lerp(along_x[layer * 2], along_x[layer * 2 + 1], local_z);
    }
    lerp(along_z[0], along_z[1], local_y)
}

fn hash_to_unit(seed: u64, x: i32, z: i32) -> f32 {
    let bits = (hash(seed, x, z) >> 40) as u32;
    bits as f32 / ((1_u32 << 24) - 1) as f32 * 2.0 - 1.0
}

fn smoothstep(value: f32) -> f32 {
    value * value * (3.0 - 2.0 * value)
}

fn lerp(from: f32, to: f32, amount: f32) -> f32 {
    from + (to - from) * amount
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
    use super::{
        Block, CHUNK_SIZE, IVec3, Mat4, Mesh, Player, PlayerInput, SEA_LEVEL, Vec3, VisibleSpace,
        World, mesh_chunk, mesh_chunk_lod, mesh_world, terrain_height,
    };

    #[test]
    fn generation_is_deterministic() {
        assert_eq!(World::generate(42), World::generate(42));
        assert_ne!(World::generate(42), World::generate(43));
    }

    #[test]
    fn generated_terrain_is_bounded_and_includes_a_coastline() {
        let world = World::generate(42);
        let (width, height, depth) = world.dimensions();
        let mut water = 0;
        let mut sand = 0;
        for z in 0..depth as i32 {
            for x in 0..width as i32 {
                let mut highest = None;
                for y in 0..height as i32 {
                    let block = world.block(IVec3::new(x, y, z)).unwrap();
                    water += (block == Block::Water) as usize;
                    sand += (block == Block::Sand) as usize;
                    if block.is_renderable() {
                        highest = Some(y);
                    }
                }
                assert!(highest.is_some_and(|y| y < height as i32));
            }
        }
        assert!(
            water > 0,
            "seed should create a visible sea at level {SEA_LEVEL}"
        );
        assert!(sand > 0, "sea-adjacent terrain should create sandy beaches");
    }

    #[test]
    fn terrain_uses_multiple_smooth_height_octaves_and_natural_trees() {
        let mut levels = [false; 44];
        let mut heights = Vec::new();
        for z in 0..96 {
            for x in 0..96 {
                let height = terrain_height(42, x, z);
                heights.push(height);
                levels[height as usize] = true;
            }
        }
        assert!(levels.into_iter().filter(|present| *present).count() >= 12);
        // Alpine relief must exist alongside the gentle coastal plains.
        assert!(heights.iter().max().copied().unwrap() >= 30);

        let world = World::generate(42);
        let wood = world
            .blocks
            .iter()
            .filter(|block| **block == Block::Wood)
            .count();
        let leaves = world
            .blocks
            .iter()
            .filter(|block| **block == Block::Leaves)
            .count();
        assert!(wood >= 3);
        assert!(leaves > wood);
    }

    #[test]
    fn generated_world_has_caves_and_snow_but_no_flooded_ones() {
        let world = World::generate(7);
        let mut air_below_surface = 0;
        let mut snow = 0;
        for z in 0..world.dimensions().2 as i32 {
            for x in 0..world.dimensions().0 as i32 {
                let surface = terrain_height(7, x, z);
                for y in 1..surface {
                    if world.block(IVec3::new(x, y, z)) == Some(Block::Air) {
                        air_below_surface += 1;
                    }
                }
                // Carving is disabled near the water table, so sea columns
                // must never contain air pockets that should have flooded.
                if surface <= SEA_LEVEL + 2 {
                    for y in 0..=SEA_LEVEL {
                        let block = world.block(IVec3::new(x, y, z)).unwrap();
                        assert!(
                            block.is_solid() || block == Block::Water,
                            "unflooded cavity at {x},{y},{z}"
                        );
                    }
                }
                snow += (world.block(IVec3::new(x, surface, z)) == Some(Block::Snow)) as usize;
            }
        }
        assert!(
            air_below_surface > 200,
            "3D cave noise should carve tunnels"
        );
        assert!(snow > 0, "peaks above the snow line should be snowy");
    }

    #[test]
    fn sunlight_propagates_and_responds_to_edits() {
        let mut world = World::new(8, 8, 8);
        for x in 0..8 {
            for z in 0..8 {
                if x >= 1 {
                    world.set(IVec3::new(x, 3, z), Block::Stone);
                }
            }
        }
        world.recompute_light();
        // Open sky above the slab is fully lit.
        assert_eq!(world.sunlight(IVec3::new(4, 4, 4)), 15);
        // Deep under the slab the column stays dark; near the open edge the
        // flood fill fades in sideways instead of a hard per-face switch.
        assert!(world.sunlight(IVec3::new(4, 2, 4)) < 12);
        assert!(world.sunlight(IVec3::new(0, 2, 4)) > world.sunlight(IVec3::new(2, 2, 4)));

        // Breaking a hole lets light pour down through the opening.
        let mut player = Player::new(Vec3::new(4.5, 5.01, 4.5));
        player.pitch = -1.54;
        assert!(player.break_block(&mut world, 8.0).is_some());
        assert!(world.sunlight(IVec3::new(4, 2, 4)) >= 12);

        // Water absorbs more light per cell than air.
        let mut world = World::new(4, 8, 4);
        for y in 0..6 {
            for x in 0..4 {
                for z in 0..4 {
                    world.set(IVec3::new(x, y, z), Block::Water);
                }
            }
        }
        world.recompute_light();
        assert!(world.sunlight(IVec3::new(2, 5, 2)) < world.sunlight(IVec3::new(2, 7, 2)));
    }

    #[test]
    fn local_sunlight_refresh_matches_a_full_recompute() {
        let mut world = World::generate_sized(42, 64, 32, 64);
        let edits = [
            IVec3::new(32, 18, 32),
            IVec3::new(8, 10, 40),
            IVec3::new(50, 24, 12),
        ];
        for edit in edits {
            world.set(
                edit,
                if world.block(edit) == Some(Block::Air) {
                    Block::Stone
                } else {
                    Block::Air
                },
            );
            world.recompute_light_around(edit, 17);
            let mut reference = world.clone();
            reference.recompute_light();
            for y in 0..32 {
                for z in 0..64 {
                    for x in 0..64 {
                        assert_eq!(
                            world.sunlight(IVec3::new(x, y, z)),
                            reference.sunlight(IVec3::new(x, y, z)),
                            "light mismatch at {x},{y},{z} after editing {edit:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn incremental_light_matches_full_recompute_through_mixed_edits() {
        let mut world = World::generate_sized(0x51A7, 32, 20, 32);
        let mut random = 0xD1B5_4A32_D192_ED03_u64;
        let materials = [
            Block::Air,
            Block::Stone,
            Block::Water,
            Block::Leaves,
            Block::Torch,
        ];

        for step in 0..96 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let x = (random % 32) as i32;
            let y = ((random >> 8) % 20) as i32;
            let z = ((random >> 16) % 32) as i32;
            let block = materials[((random >> 24) as usize) % materials.len()];
            let edit = IVec3::new(x, y, z);
            world.set(edit, block);
            world.recompute_light_after_edit(edit);

            let mut reference = world.clone();
            reference.recompute_light();
            assert_eq!(
                world.direct_sunlight, reference.direct_sunlight,
                "direct sky mismatch after mixed edit {step} at {edit:?}"
            );
            assert_eq!(
                world.sunlight, reference.sunlight,
                "sky mismatch after mixed edit {step} at {edit:?}"
            );
            assert_eq!(
                world.block_light, reference.block_light,
                "block light mismatch after mixed edit {step} at {edit:?}"
            );
        }
    }

    #[test]
    fn torch_light_update_bounds_cross_chunk_edges() {
        let mut world = World::new(48, 8, 16);
        world.recompute_light();
        let torch = IVec3::new(CHUNK_SIZE - 1, 3, 8);
        world.set(torch, Block::Torch);
        let update = world.recompute_light_after_edit(torch);
        let (min_x, max_x, min_z, max_z) = update.horizontal_bounds().unwrap();

        assert!(min_x < CHUNK_SIZE - 1);
        assert!(max_x >= CHUNK_SIZE);
        assert!(min_z < torch.z && max_z > torch.z);
        assert_eq!(world.block_light(torch), 14);
        assert_eq!(world.block_light(torch + IVec3::new(13, 0, 0)), 1);
    }

    #[test]
    fn incrementally_built_enclosed_room_goes_dark_and_leaks_light_gradually() {
        let mut world = World::new(13, 10, 13);
        // Build a floating shell room exactly the way gameplay does: one
        // block at a time, refreshing sunlight locally after each placement.
        // Shell spans x/z 3..=9 and y 1..=5 with a hollow 5x3x5 interior.
        let mut placements = Vec::new();
        for y in 1..=5 {
            for z in 3..=9 {
                for x in 3..=9 {
                    let on_shell = y == 1 || y == 5 || x == 3 || x == 9 || z == 3 || z == 9;
                    if on_shell {
                        placements.push(IVec3::new(x, y, z));
                    }
                }
            }
        }
        for position in placements {
            world.set(position, Block::Stone);
            world.recompute_light_around(position, 17);
        }

        // The incremental local refreshes must agree with a full recompute.
        let mut reference = world.clone();
        reference.recompute_light();
        for y in 0..10 {
            for z in 0..13 {
                for x in 0..13 {
                    assert_eq!(
                        world.sunlight(IVec3::new(x, y, z)),
                        reference.sunlight(IVec3::new(x, y, z)),
                        "incremental room light mismatch at {x},{y},{z}"
                    );
                }
            }
        }

        // Every interior cell of the sealed room is fully dark.
        for y in 2..=4 {
            for z in 4..=8 {
                for x in 4..=8 {
                    assert_eq!(world.sunlight(IVec3::new(x, y, z)), 0);
                }
            }
        }
        // Baked floor light inside is far darker than the roof outside.
        let mesh = mesh_world(&world);
        let interior_floor = mesh
            .vertices
            .iter()
            .filter(|vertex| {
                vertex.normal.y > 0.0
                    && vertex.position.y == 2.0
                    && vertex.position.x > 4.0
                    && vertex.position.x < 8.0
                    && vertex.position.z > 4.0
                    && vertex.position.z < 8.0
            })
            .map(|vertex| vertex.light)
            .fold(1.0_f32, f32::min);
        let roof = mesh
            .vertices
            .iter()
            .filter(|vertex| vertex.normal.y > 0.0 && vertex.position.y == 6.0)
            .map(|vertex| vertex.light)
            .fold(0.0_f32, f32::max);
        assert!(roof > 0.99);
        assert!(
            interior_floor < roof - 0.7,
            "room floor {interior_floor} must be much darker than roof {roof}"
        );

        // Opening one window lets light leak in and fade with distance.
        world.set(IVec3::new(6, 3, 3), Block::Air);
        world.recompute_light_around(IVec3::new(6, 3, 3), 17);
        // The window cell itself is one step from open air (14), and each
        // interior cell fades one more level with distance from the opening.
        assert_eq!(world.sunlight(IVec3::new(6, 3, 3)), 14);
        assert_eq!(world.sunlight(IVec3::new(6, 3, 4)), 13);
        assert_eq!(world.sunlight(IVec3::new(6, 3, 5)), 12);
        assert!(world.sunlight(IVec3::new(6, 3, 8)) < world.sunlight(IVec3::new(6, 3, 4)));
    }

    #[test]
    fn torches_emit_block_light_that_spreads_and_fades() {
        let mut world = World::new(16, 8, 16);
        for z in 0..16 {
            for x in 0..16 {
                world.set(IVec3::new(x, 0, z), Block::Stone);
            }
        }
        let mut player = Player::new(Vec3::new(8.5, 1.01, 8.5));
        player.selected_block = Block::Torch;
        player.pitch = -1.54;
        let placed = player.place_block(&mut world, 4.0).unwrap();
        assert_eq!(world.block(placed), Some(Block::Torch));

        // The torch emits level 14 and light fades one level per block.
        assert_eq!(world.block_light(placed), 14);
        assert_eq!(world.block_light(placed + IVec3::new(1, 0, 0)), 13);
        assert_eq!(world.block_light(placed + IVec3::new(3, 0, 0)), 11);
        assert_eq!(world.sunlight(placed), 15, "sky light is unaffected");

        // Mesh vertices near the torch carry baked torch light.
        let mesh = mesh_world(&world);
        assert!(mesh.vertices.iter().any(|vertex| vertex.torch_light > 0.8));

        // The torch is not solid: players pass through it, and it can be
        // broken by aiming at it.
        assert!(!Block::Torch.is_solid());
        let hit = world
            .raycast(player.camera().position, player.camera().forward(), 4.0)
            .unwrap();
        assert_eq!(hit.position, placed);
        assert!(player.break_block(&mut world, 4.0).is_some());
        assert_eq!(world.block(placed), Some(Block::Air));
        assert_eq!(world.block_light(placed + IVec3::new(1, 0, 0)), 0);
    }

    #[test]
    fn enclosed_room_with_torch_is_lit_by_block_light_only() {
        let mut world = World::new(13, 10, 13);
        for y in 1..=5 {
            for z in 3..=9 {
                for x in 3..=9 {
                    let on_shell = y == 1 || y == 5 || x == 3 || x == 9 || z == 3 || z == 9;
                    if on_shell {
                        world.set(IVec3::new(x, y, z), Block::Stone);
                    }
                }
            }
        }
        world.set(IVec3::new(6, 2, 6), Block::Torch);
        world.recompute_light();

        // Sky light cannot enter the sealed room...
        assert_eq!(world.sunlight(IVec3::new(6, 2, 6)), 0);
        // ...but the torch floods the interior through the shell gap at 15?
        // No: the torch sits inside, so nearby walls and floor are lit.
        assert_eq!(world.block_light(IVec3::new(6, 2, 6)), 14);
        assert!(world.block_light(IVec3::new(4, 2, 4)) >= 8);

        let mesh = mesh_world(&world);
        let interior_floor_light = mesh
            .vertices
            .iter()
            .filter(|vertex| {
                vertex.normal.y > 0.0
                    && vertex.position.y == 2.0
                    && vertex.position.x >= 4.0
                    && vertex.position.x <= 8.0
                    && vertex.position.z >= 4.0
                    && vertex.position.z <= 8.0
            })
            .map(|vertex| vertex.torch_light)
            .fold(0.0_f32, f32::max);
        assert!(
            interior_floor_light > 0.5,
            "torch-lit floor should bake meaningful block light"
        );
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
    fn mesh_supplies_atlas_uvs_and_corner_shading_metadata() {
        let mut world = World::new(4, 4, 4);
        world.set(IVec3::new(1, 1, 1), Block::Stone);
        // These two blocks meet one top-face corner and must darken it.
        world.set(IVec3::new(0, 2, 1), Block::Stone);
        world.set(IVec3::new(1, 2, 0), Block::Stone);
        let mesh = mesh_world(&world);
        assert!(mesh.vertices.iter().all(|vertex| {
            vertex.atlas_uv[0] >= 0.0
                && vertex.atlas_uv[0] <= 1.0
                && vertex.atlas_uv[1] >= 0.0
                && vertex.atlas_uv[1] <= 1.0
                && vertex.ambient_occlusion > 0.0
                && vertex.ambient_occlusion <= 1.0
                && vertex.light >= 0.0
                && vertex.light <= 1.0
        }));
        let darkest_top_corner = mesh
            .vertices
            .iter()
            .filter(|vertex| vertex.position.y == 2.0 && vertex.normal.y == 1.0)
            .map(|vertex| vertex.ambient_occlusion)
            .fold(1.0_f32, f32::min);
        assert!(darkest_top_corner < 1.0);
    }

    #[test]
    fn water_is_meshable_without_becoming_collision_solid() {
        let mut world = World::new(2, 2, 2);
        world.set(IVec3::new(0, 0, 0), Block::Water);
        assert!(!Block::Water.is_solid());
        assert!(
            mesh_world(&world)
                .vertices
                .iter()
                .any(|vertex| vertex.block == Block::Water)
        );
    }

    #[test]
    fn chunk_meshes_combine_into_the_full_world_mesh() {
        let world = World::generate_sized(42, 48, 24, 48);
        let full = mesh_world(&world);
        let mut combined = Mesh::default();
        for chunk_z in 0..3 {
            for chunk_x in 0..3 {
                let chunk = mesh_chunk(&world, chunk_x, chunk_z);
                let base = combined.vertices.len() as u32;
                combined.vertices.extend(chunk.vertices);
                combined
                    .indices
                    .extend(chunk.indices.into_iter().map(|index| index + base));
            }
        }
        assert_eq!(combined.vertices.len(), full.vertices.len());
        assert_eq!(combined.indices.len(), full.indices.len());
    }

    #[test]
    fn lod_meshing_drops_sealed_interiors_but_keeps_exposed_faces() {
        let mut world = World::new(8, 8, 8);
        for y in 0..8 {
            for z in 0..8 {
                for x in 0..8 {
                    world.set(IVec3::new(x, y, z), Block::Stone);
                }
            }
        }
        // Hollow out a fully enclosed pocket: its walls are only visible from
        // an unlit interior, so the LOD mesh must skip them.
        for y in 2..6 {
            for z in 2..6 {
                for x in 2..6 {
                    world.set(IVec3::new(x, y, z), Block::Air);
                }
            }
        }
        world.recompute_light();
        assert_eq!(world.sunlight(IVec3::new(4, 4, 4)), 0);

        let visible_space = VisibleSpace::from_world(&world, IVec3::new(0, 0, 0));
        let full = mesh_chunk(&world, 0, 0);
        let lod = mesh_chunk_lod(&world, 0, 0, &visible_space);
        assert!(lod.vertices.len() < full.vertices.len());
        assert!(!lod.vertices.is_empty());
        // The exposed exterior faces survive the reduction.
        assert!(lod.vertices.iter().any(|vertex| vertex.position.x == 0.0));
    }

    #[test]
    fn lod_meshing_keeps_a_dark_tunnel_connected_to_the_exterior() {
        let mut world = World::new(32, 8, 8);
        for y in 0..8 {
            for z in 0..8 {
                for x in 0..32 {
                    world.set(IVec3::new(x, y, z), Block::Stone);
                }
            }
        }
        // This tunnel is open at x=0 but receives no skylight because its
        // ceiling is solid. A light-level LOD incorrectly deleted its walls.
        for x in 0..31 {
            world.set(IVec3::new(x, 3, 3), Block::Air);
        }
        world.recompute_light();
        assert_eq!(world.sunlight(IVec3::new(24, 3, 3)), 0);

        let visible_space = VisibleSpace::from_world(&world, IVec3::new(0, 3, 3));
        let full = mesh_chunk(&world, 1, 0);
        let lod = mesh_chunk_lod(&world, 1, 0, &visible_space);
        assert_eq!(lod.vertices.len(), full.vertices.len());
        assert_eq!(lod.indices.len(), full.indices.len());
    }

    #[test]
    fn lod_meshing_keeps_the_viewers_enclosed_component() {
        let mut world = World::new(8, 8, 8);
        for y in 0..8 {
            for z in 0..8 {
                for x in 0..8 {
                    world.set(IVec3::new(x, y, z), Block::Stone);
                }
            }
        }
        for y in 2..6 {
            for z in 2..6 {
                for x in 2..6 {
                    world.set(IVec3::new(x, y, z), Block::Air);
                }
            }
        }

        let visible_space = VisibleSpace::from_world(&world, IVec3::new(4, 4, 4));
        let full = mesh_chunk(&world, 0, 0);
        let lod = mesh_chunk_lod(&world, 0, 0, &visible_space);
        assert_eq!(lod.vertices.len(), full.vertices.len());
        assert_eq!(lod.indices.len(), full.indices.len());
    }

    #[test]
    fn enclosed_corners_are_darker_than_open_surface() {
        // Open flat ground: every corner sample is a sky-lit air cell.
        let mut open = World::new(8, 8, 8);
        for z in 0..8 {
            for x in 0..8 {
                open.set(IVec3::new(x, 3, z), Block::Stone);
            }
        }
        open.recompute_light();
        let open_light = mesh_world(&open)
            .vertices
            .iter()
            .filter(|vertex| vertex.normal.y > 0.0 && vertex.position.y == 4.0)
            .map(|vertex| vertex.light)
            .fold(0.0_f32, f32::max);

        // A one-block shaft through solid rock: three of each floor corner's
        // four sample cells are rock and must count as darkness, so the shaft
        // floor stays darker than open ground instead of reading full sky.
        let mut shaft = World::new(8, 8, 8);
        for y in 0..8 {
            for z in 0..8 {
                for x in 0..8 {
                    shaft.set(IVec3::new(x, y, z), Block::Stone);
                }
            }
        }
        for y in 1..8 {
            shaft.set(IVec3::new(4, y, 4), Block::Air);
        }
        shaft.recompute_light();
        assert_eq!(shaft.sunlight(IVec3::new(4, 1, 4)), 15);
        let shaft_light = mesh_world(&shaft)
            .vertices
            .iter()
            .filter(|vertex| vertex.normal.y > 0.0 && vertex.position.y == 1.0)
            .map(|vertex| vertex.light)
            .fold(1.0_f32, f32::min);

        assert!(open_light > 0.99);
        assert!(
            shaft_light < open_light - 0.25,
            "shaft floor light {shaft_light} should be clearly darker than open {open_light}"
        );
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
        let clip = projection.transform_point(point_above_camera);
        let near = projection.transform_point(Vec3::new(0.0, 0.0, -0.1));
        let far = projection.transform_point(Vec3::new(0.0, 0.0, -100.0));
        assert!(clip.y > 0.0);
        assert!(near.z < far.z);
    }
}
