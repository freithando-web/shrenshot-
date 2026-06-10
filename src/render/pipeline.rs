use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

const MAX_VERTICES: usize = 1_048_576;
const MAX_INDICES: usize = 3_145_728;
const FRAME_BUFFER_COUNT: usize = 3;
const TILE_SIZE: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 { pub x: f32, pub y: f32 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

impl Vec3 {
    pub fn dot(self, rhs: Vec3) -> f32 { self.x*rhs.x + self.y*rhs.y + self.z*rhs.z }
    pub fn cross(self, rhs: Vec3) -> Vec3 {
        Vec3 { x: self.y*rhs.z - self.z*rhs.y,
               y: self.z*rhs.x - self.x*rhs.z,
               z: self.x*rhs.y - self.y*rhs.x }
    }
    pub fn length(self) -> f32 { (self.x*self.x + self.y*self.y + self.z*self.z).sqrt() }
    pub fn normalize(self) -> Vec3 {
        let len = self.length();
        if len < 1e-8 { return Vec3 { x:0.0, y:0.0, z:0.0 }; }
        Vec3 { x: self.x/len, y: self.y/len, z: self.z/len }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Mat4([[f32; 4]; 4]);

impl Mat4 {
    pub fn identity() -> Self {
        Mat4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let nf = 1.0 / (near - far);
        Mat4([
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) * nf, -1.0],
            [0.0, 0.0, 2.0 * far * near * nf, 0.0],
        ])
    }

    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = Vec3 { x: target.x-eye.x, y: target.y-eye.y, z: target.z-eye.z }.normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Mat4([
            [s.x, u.x, -f.x, 0.0],
            [s.y, u.y, -f.y, 0.0],
            [s.z, u.z, -f.z, 0.0],
            [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
        ])
    }

    pub fn mul(self, rhs: Mat4) -> Mat4 {
        let mut out = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    out[i][j] += self.0[i][k] * rhs.0[k][j];
                }
            }
        }
        Mat4(out)
    }

    pub fn transform_vec4(self, v: Vec4) -> Vec4 {
        let m = self.0;
        Vec4 {
            x: m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3]*v.w,
            y: m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3]*v.w,
            z: m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]*v.w,
            w: m[3][0]*v.x + m[3][1]*v.y + m[3][2]*v.z + m[3][3]*v.w,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub color: Vec4,
    pub tangent: Vec4,
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_id: u32,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
}

impl Mesh {
    pub fn compute_aabb(&mut self) {
        let mut min = Vec3 { x: f32::MAX, y: f32::MAX, z: f32::MAX };
        let mut max = Vec3 { x: f32::MIN, y: f32::MIN, z: f32::MIN };
        for v in &self.vertices {
            min.x = min.x.min(v.position.x);
            min.y = min.y.min(v.position.y);
            min.z = min.z.min(v.position.z);
            max.x = max.x.max(v.position.x);
            max.y = max.y.max(v.position.y);
            max.z = max.z.max(v.position.z);
        }
        self.aabb_min = min;
        self.aabb_max = max;
    }

    pub fn compute_normals(&mut self) {
        let n = self.vertices.len();
        let mut normals = vec![Vec3 { x:0.0, y:0.0, z:0.0 }; n];
        let indices = self.indices.clone();
        for tri in indices.chunks_exact(3) {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let pa = self.vertices[a].position;
            let pb = self.vertices[b].position;
            let pc = self.vertices[c].position;
            let ab = Vec3 { x: pb.x-pa.x, y: pb.y-pa.y, z: pb.z-pa.z };
            let ac = Vec3 { x: pc.x-pa.x, y: pc.y-pa.y, z: pc.z-pa.z };
            let n_face = ab.cross(ac);
            for idx in [a, b, c] {
                normals[idx].x += n_face.x;
                normals[idx].y += n_face.y;
                normals[idx].z += n_face.z;
            }
        }
        for (i, v) in self.vertices.iter_mut().enumerate() {
            v.normal = normals[i].normalize();
        }
    }
}

#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: Vec3,
    pub albedo_map: Option<u32>,
    pub normal_map: Option<u32>,
    pub metallic_roughness_map: Option<u32>,
    pub emissive_map: Option<u32>,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            albedo: Vec4 { x:1.0, y:1.0, z:1.0, w:1.0 },
            metallic: 0.0,
            roughness: 0.5,
            emissive: Vec3 { x:0.0, y:0.0, z:0.0 },
            albedo_map: None,
            normal_map: None,
            metallic_roughness_map: None,
            emissive_map: None,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }
}

#[derive(Debug)]
pub struct RenderTarget {
    pub width: u32,
    pub height: u32,
    pub color_buffer: Vec<u32>,
    pub depth_buffer: Vec<f32>,
    pub samples: u8,
}

impl RenderTarget {
    pub fn new(width: u32, height: u32, msaa_samples: u8) -> Self {
        let size = (width * height) as usize;
        RenderTarget {
            width, height,
            color_buffer: vec![0u32; size],
            depth_buffer: vec![1.0f32; size],
            samples: msaa_samples,
        }
    }

    pub fn clear(&mut self, color: u32) {
        self.color_buffer.fill(color);
        self.depth_buffer.fill(1.0);
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, color: u32, depth: f32) -> bool {
        if x >= self.width || y >= self.height { return false; }
        let idx = (y * self.width + x) as usize;
        if depth < self.depth_buffer[idx] {
            self.depth_buffer[idx] = depth;
            self.color_buffer[idx] = color;
            return true;
        }
        false
    }
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at(self.position, self.target, self.up)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective(self.fov_y, self.aspect, self.near, self.far)
    }

    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix().mul(self.view_matrix())
    }
}

pub struct TileRenderer {
    tiles_x: u32,
    tiles_y: u32,
    tile_size: u32,
    jobs: Vec<Arc<Mutex<TileJob>>>,
}

struct TileJob {
    x: u32, y: u32,
    width: u32, height: u32,
    done: bool,
}

impl TileRenderer {
    pub fn new(target: &RenderTarget) -> Self {
        let tiles_x = (target.width + TILE_SIZE - 1) / TILE_SIZE;
        let tiles_y = (target.height + TILE_SIZE - 1) / TILE_SIZE;
        let mut jobs = Vec::new();
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let x = tx * TILE_SIZE;
                let y = ty * TILE_SIZE;
                let w = TILE_SIZE.min(target.width - x);
                let h = TILE_SIZE.min(target.height - y);
                jobs.push(Arc::new(Mutex::new(TileJob { x, y, width: w, height: h, done: false })));
            }
        }
        TileRenderer { tiles_x, tiles_y, tile_size: TILE_SIZE, jobs }
    }

    pub fn tile_count(&self) -> usize { self.jobs.len() }
}

#[derive(Debug, Default)]
pub struct FrameStats {
    pub draw_calls: u32,
    pub triangles_submitted: u64,
    pub triangles_rendered: u64,
    pub vertices_processed: u64,
    pub frame_time_ms: f32,
    pub cpu_time_ms: f32,
    pub gpu_time_ms: f32,
}

pub struct Renderer {
    target: RenderTarget,
    camera: Camera,
    materials: HashMap<u32, Material>,
    meshes: Vec<Mesh>,
    frame_index: u64,
    stats: FrameStats,
    start_time: Instant,
}

impl Renderer {
    pub fn new(width: u32, height: u32) -> Self {
        Renderer {
            target: RenderTarget::new(width, height, 1),
            camera: Camera {
                position: Vec3 { x:0.0, y:0.0, z:5.0 },
                target: Vec3 { x:0.0, y:0.0, z:0.0 },
                up: Vec3 { x:0.0, y:1.0, z:0.0 },
                fov_y: std::f32::consts::PI / 4.0,
                aspect: width as f32 / height as f32,
                near: 0.1, far: 1000.0,
            },
            materials: HashMap::new(),
            meshes: Vec::new(),
            frame_index: 0,
            stats: FrameStats::default(),
            start_time: Instant::now(),
        }
    }

    pub fn add_mesh(&mut self, mesh: Mesh) -> usize {
        let id = self.meshes.len();
        self.meshes.push(mesh);
        id
    }

    pub fn set_material(&mut self, id: u32, material: Material) {
        self.materials.insert(id, material);
    }

    pub fn begin_frame(&mut self) {
        self.target.clear(0xFF1A1A2E);
        self.stats = FrameStats::default();
    }

    pub fn submit_mesh(&mut self, mesh_id: usize, transform: Mat4) {
        if mesh_id >= self.meshes.len() { return; }
        let mesh = &self.meshes[mesh_id];
        let mvp = self.camera.view_projection().mul(transform);
        self.stats.draw_calls += 1;
        self.stats.triangles_submitted += (mesh.indices.len() / 3) as u64;
        self.stats.vertices_processed += mesh.vertices.len() as u64;

        for tri in mesh.indices.chunks_exact(3) {
            let v0 = &mesh.vertices[tri[0] as usize];
            let v1 = &mesh.vertices[tri[1] as usize];
            let v2 = &mesh.vertices[tri[2] as usize];
            let p0 = mvp.transform_vec4(Vec4 { x: v0.position.x, y: v0.position.y, z: v0.position.z, w: 1.0 });
            let p1 = mvp.transform_vec4(Vec4 { x: v1.position.x, y: v1.position.y, z: v1.position.z, w: 1.0 });
            let p2 = mvp.transform_vec4(Vec4 { x: v2.position.x, y: v2.position.y, z: v2.position.z, w: 1.0 });
            if p0.w <= 0.0 || p1.w <= 0.0 || p2.w <= 0.0 { continue; }
            self.stats.triangles_rendered += 1;
        }
    }

    pub fn end_frame(&mut self) -> &FrameStats {
        self.frame_index += 1;
        self.stats.frame_time_ms = self.start_time.elapsed().as_secs_f32() * 1000.0;
        &self.stats
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.target = RenderTarget::new(width, height, self.target.samples);
        self.camera.aspect = width as f32 / height as f32;
    }

    pub fn frame_index(&self) -> u64 { self.frame_index }
}
