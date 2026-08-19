//! Background CPU meshing for streamed Boxcraft terrain.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use boxcraft_core::{IVec3, LightUpdate, Mesh, VisibleSpace, World, mesh_chunk, mesh_chunk_lod};

/// Two workers leave the UI and SGFX submission path on their own thread while
/// still providing useful parallelism to a Scarlet guest launched with SMP.
pub const WORKER_COUNT: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightEdit {
    pub position: IVec3,
    pub topology_changed: bool,
}

pub enum MeshJob {
    Near {
        id: u64,
        terrain_revision: u64,
        chunk: (i32, i32),
        world: Arc<World>,
    },
    Far {
        id: u64,
        terrain_revision: u64,
        far_generation: u64,
        chunks: Vec<(i32, i32)>,
        world: Arc<World>,
        viewer: IVec3,
        visible_space: Option<Arc<VisibleSpace>>,
        cached_meshes: Arc<HashMap<(i32, i32), Arc<Mesh>>>,
    },
    Edit {
        id: u64,
        terrain_revision: u64,
        world: Arc<World>,
        edits: Vec<LightEdit>,
    },
}

pub struct NearMeshResult {
    pub id: u64,
    pub terrain_revision: u64,
    pub chunk: (i32, i32),
    pub mesh: Arc<Mesh>,
}

pub struct FarMeshResult {
    pub id: u64,
    pub terrain_revision: u64,
    pub far_generation: u64,
    pub chunks: Vec<(i32, i32)>,
    pub visible_space: Arc<VisibleSpace>,
    pub meshes: HashMap<(i32, i32), Arc<Mesh>>,
}

pub struct EditMeshResult {
    pub id: u64,
    pub terrain_revision: u64,
    pub world: Arc<World>,
    pub edits: Vec<LightEdit>,
    pub light_update: LightUpdate,
}

pub enum MeshResult {
    Near(NearMeshResult),
    Far(FarMeshResult),
    Edit(EditMeshResult),
}

struct Shared {
    jobs: Mutex<VecDeque<MeshJob>>,
    wake_workers: Condvar,
    results: Mutex<VecDeque<MeshResult>>,
}

/// A small fixed worker pool. Workers only produce CPU-side core meshes;
/// ScarletUI state and SGFX objects remain confined to the application thread.
#[derive(Clone)]
pub struct MeshWorkers {
    shared: Arc<Shared>,
}

impl MeshWorkers {
    pub fn new() -> Self {
        let shared = Arc::new(Shared {
            jobs: Mutex::new(VecDeque::new()),
            wake_workers: Condvar::new(),
            results: Mutex::new(VecDeque::new()),
        });
        for _ in 0..WORKER_COUNT {
            let worker_shared = Arc::clone(&shared);
            thread::spawn(move || worker_loop(worker_shared));
        }
        Self { shared }
    }

    pub fn submit(&self, job: MeshJob) {
        self.shared
            .jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push_back(job);
        self.shared.wake_workers.notify_one();
    }

    /// Put the newest light snapshot ahead of queued streaming work and
    /// discard every queued job for older world revisions. Jobs already
    /// executing are harmless: the UI rejects their stale revision.
    pub fn submit_edit(&self, job: MeshJob) {
        let mut jobs = self
            .shared
            .jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        jobs.clear();
        jobs.push_front(job);
        drop(jobs);
        self.shared
            .results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.shared.wake_workers.notify_all();
    }

    pub fn try_recv(&self) -> Option<MeshResult> {
        self.shared
            .results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop_front()
    }

    #[cfg_attr(not(target_os = "scarlet"), allow(dead_code))]
    pub fn defer(&self, result: MeshResult) {
        self.shared
            .results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push_front(result);
    }
}

fn worker_loop(shared: Arc<Shared>) {
    loop {
        let job = {
            let mut jobs = shared
                .jobs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            while jobs.is_empty() {
                jobs = shared
                    .wake_workers
                    .wait(jobs)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            jobs.pop_front().expect("mesh worker woke without a job")
        };
        let result = run_job(job);
        let mut results = shared
            .results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if matches!(&result, MeshResult::Edit(_)) {
            results.push_front(result);
        } else {
            results.push_back(result);
        }
    }
}

fn run_job(job: MeshJob) -> MeshResult {
    match job {
        MeshJob::Near {
            id,
            terrain_revision,
            chunk,
            world,
        } => MeshResult::Near(NearMeshResult {
            id,
            terrain_revision,
            chunk,
            mesh: Arc::new(mesh_chunk(&world, chunk.0, chunk.1)),
        }),
        MeshJob::Far {
            id,
            terrain_revision,
            far_generation,
            chunks,
            world,
            viewer,
            visible_space,
            cached_meshes,
        } => {
            let visible_space =
                visible_space.unwrap_or_else(|| Arc::new(VisibleSpace::from_world(&world, viewer)));
            let mut meshes = HashMap::with_capacity(chunks.len());
            for chunk in chunks.iter().copied() {
                let mesh = cached_meshes.get(&chunk).cloned().unwrap_or_else(|| {
                    Arc::new(mesh_chunk_lod(&world, chunk.0, chunk.1, &visible_space))
                });
                meshes.insert(chunk, mesh);
            }
            MeshResult::Far(FarMeshResult {
                id,
                terrain_revision,
                far_generation,
                chunks,
                visible_space,
                meshes,
            })
        }
        MeshJob::Edit {
            id,
            terrain_revision,
            world,
            edits,
        } => {
            let mut next_world = (*world).clone();
            let positions = edits.iter().map(|edit| edit.position).collect::<Vec<_>>();
            let light_update = next_world.recompute_light_after_edits(&positions);
            MeshResult::Edit(EditMeshResult {
                id,
                terrain_revision,
                world: Arc::new(next_world),
                edits,
                light_update,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use boxcraft_core::Block;

    use super::*;

    #[test]
    fn worker_returns_identified_near_mesh() {
        let workers = MeshWorkers::new();
        let world = Arc::new(World::generate_sized(7, 32, 24, 32));
        workers.submit(MeshJob::Near {
            id: 41,
            terrain_revision: 9,
            chunk: (0, 0),
            world,
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(MeshResult::Near(result)) = workers.try_recv() {
                assert_eq!(result.id, 41);
                assert_eq!(result.terrain_revision, 9);
                assert_eq!(result.chunk, (0, 0));
                assert!(!result.mesh.indices.is_empty());
                break;
            }
            assert!(Instant::now() < deadline, "mesh worker timed out");
            thread::yield_now();
        }
    }

    #[test]
    fn worker_builds_far_meshes_and_visible_space() {
        let workers = MeshWorkers::new();
        let world = Arc::new(World::generate_sized(11, 32, 24, 32));
        workers.submit(MeshJob::Far {
            id: 52,
            terrain_revision: 12,
            far_generation: 3,
            chunks: vec![(0, 0)],
            world,
            viewer: IVec3::new(8, 20, 8),
            visible_space: None,
            cached_meshes: Arc::new(HashMap::new()),
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(MeshResult::Far(result)) = workers.try_recv() {
                assert_eq!(result.id, 52);
                assert_eq!(result.terrain_revision, 12);
                assert_eq!(result.far_generation, 3);
                assert_eq!(result.chunks, vec![(0, 0)]);
                assert!(result.visible_space.reaches(IVec3::new(-1, 0, 0)));
                assert!(!result.meshes[&(0, 0)].indices.is_empty());
                break;
            }
            assert!(Instant::now() < deadline, "far mesh worker timed out");
            thread::yield_now();
        }
    }

    #[test]
    fn worker_recomputes_batched_edit_light_off_thread() {
        let workers = MeshWorkers::new();
        let mut world = World::new(8, 8, 8);
        let position = IVec3::new(3, 3, 3);
        world.set(position, Block::Stone);
        world.recompute_light();
        world.set(position, Block::Air);
        let edit = LightEdit {
            position,
            topology_changed: true,
        };
        workers.submit_edit(MeshJob::Edit {
            id: 61,
            terrain_revision: 14,
            world: Arc::new(world),
            edits: vec![edit],
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(MeshResult::Edit(result)) = workers.try_recv() {
                assert_eq!(result.id, 61);
                assert_eq!(result.terrain_revision, 14);
                assert_eq!(result.edits, vec![edit]);
                assert_eq!(result.world.block(position), Some(Block::Air));
                assert!(result.light_update.horizontal_bounds().is_some());
                break;
            }
            assert!(Instant::now() < deadline, "edit worker timed out");
            thread::yield_now();
        }
    }
}
