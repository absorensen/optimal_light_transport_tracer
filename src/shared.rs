use std::f64::consts::PI;

use rand::{Rng, rngs::ThreadRng};
use ultraviolet::DVec3;

#[inline(always)]
pub fn to_int(x: f64) -> i32 {
    (clamp(x).powf(1.0 / 2.2) * 255.0 + 0.5) as i32
}

#[inline(always)]
pub fn clamp(x: f64) -> f64 {
    let result: f64 = 
        if x < 0.0 { 0.0 } else {
            if 1.0 < x { 1.0 } else { x }
        };
    result
}

#[inline(always)]
pub fn random_cosine_direction() -> DVec3 {
    let mut rng: ThreadRng = rand::thread_rng();
    
    let r1: f64 = rng.gen::<f64>();
    let r2: f64 = rng.gen::<f64>();
    let z: f64 = (1.0 - r2).sqrt();
    
    let phi: f64 = 2.0 * PI * r1;
    let x: f64 = phi.cos() * r2.sqrt();
    let y: f64 = phi.sin() * r2.sqrt();

    DVec3::new(x, y, z)
}

#[inline(always)]
pub fn random_to_sphere(radius: f64, distance_squared: f64) -> DVec3 {
    let mut rng: ThreadRng = rand::thread_rng();
    
    let r1: f64 = rng.gen::<f64>();
    let r2: f64 = rng.gen::<f64>();
    let z: f64 = 1.0 + r2 * ((1.0 - radius * radius / distance_squared) - 1.0);
    
    let phi: f64 = 2.0 * PI * r1;
    let x: f64 = phi.cos() * (1.0 - z * z).sqrt();
    let y: f64 = phi.sin() * (1.0 - z * z).sqrt();

    DVec3::new(x, y, z)
}

#[inline(always)]
pub fn random_on_sphere(radius: f64, position: &DVec3) -> DVec3 {
    let mut random_direction: DVec3 = random_in_unit_sphere();
    random_direction.normalize();
    
    random_direction * radius + *position
}

#[inline(always)]
pub fn random_in_unit_sphere() -> DVec3 {
    let mut rng: ThreadRng = rand::thread_rng();

    loop {
        let p: DVec3 = DVec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
        if p.mag_sq() <= 1.0  { return p; };
    }
}

#[inline(always)]
pub fn random_in_hemisphere(normal: &DVec3) -> DVec3 {
    let in_unit_sphere = random_in_unit_sphere();

    if 0.0 < in_unit_sphere.dot(*normal) { in_unit_sphere } else { -in_unit_sphere }
}


pub struct OrthoNormalBase {
    pub axis: [DVec3; 3],
}

impl OrthoNormalBase {
    pub fn new() -> Self {
        OrthoNormalBase { axis: [DVec3::zero(), DVec3::zero(), DVec3::zero()] }
    }

    pub fn local(&self, a: &DVec3) -> DVec3 {
        a.x * self.axis[0] + a.y * self.axis[1] + a.z * self.axis[2]
    }

    pub fn build_from_w(&mut self, n: &DVec3) -> () {
        self.axis[2] = n.normalized();
        let a: DVec3 = if 0.9 < self.axis[2].x.abs() { DVec3::unit_y() } else { DVec3::unit_x() };
        self.axis[1] = self.axis[2].cross(a).normalized();
        self.axis[0] = self.axis[2].cross(self.axis[1]);
    }
}

#[derive(Debug, Clone)]
pub enum LightTransportAlgorithm {
    NAIVE,
    MIXTURE_PDF,
    DYNAMIC_SAMPLING,
    BIDIRECTIONAL,
    METROPOLIS,
    RESTIR,
    RESTIRGI,
}

#[derive(Debug, Clone)]
pub struct RenderContext {
    pub width: usize,
    pub height: usize,
    pub max_depth: usize,
    pub monte_carlo_depth: usize,
    pub subpixels_count: usize,
    pub subpixels_offset: f64,
    pub subsamples_count: usize,
    pub sample_scale: f64,
    pub camera: Ray,
    pub cx: DVec3,
    pub cy: DVec3,
    pub total_pixels: usize,
    pub algorithm: LightTransportAlgorithm,
}

impl RenderContext {
    pub fn default() -> Self {
        let width: usize = 512;
        let height: usize = 384;
        let max_depth: usize = 10;
        let monte_carlo_depth: usize = 6;
        let subpixels_count: usize = 8;
        let subpixels_offset: f64 = 1.0 / subpixels_count as f64;
        let subsamples_count: usize = 4;
        let sample_scale: f64 = 1.0 / (subpixels_count * subpixels_count * subsamples_count) as f64;
    
        let camera: Ray = Ray::new(DVec3::new(50.0, 52.0, 295.6), DVec3::new(0.0, -0.042612, -1.0).normalized());
        let cx: DVec3 = DVec3::new((width as f64) * 0.5135 / (height as f64), 0.0, 0.0);
        let cy: DVec3 = cx.cross(camera.direction).normalized() * 0.5135;

        let total_pixels: usize = height * width;

        let algorithm: LightTransportAlgorithm = LightTransportAlgorithm::BIDIRECTIONAL;

        RenderContext { width, height, max_depth, monte_carlo_depth, subpixels_count, subpixels_offset, subsamples_count, sample_scale, camera, cx, cy, total_pixels, algorithm }
    }
}

pub static SPHERES: &'static [Sphere] = 
    &[
        Sphere{radius: 1e5,   position: DVec3::new( 1e5+1.0,40.8,81.6   ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.75,0.25,0.25   ), reflection: ReflectionType::DIFFUSE   },//Left
        Sphere{radius: 1e5,   position: DVec3::new( -1e5+99.0,40.8,81.6 ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.25,0.25,0.75   ), reflection: ReflectionType::DIFFUSE   },//Right
        Sphere{radius: 1e5,   position: DVec3::new( 50.0,40.8, 1e5      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.75, 0.75, 0.75 ), reflection: ReflectionType::DIFFUSE   },//Back
        Sphere{radius: 1e5,   position: DVec3::new( 50.0,40.8,-1e5+170.0), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.0,0.0,0.0      ), reflection: ReflectionType::DIFFUSE   },//Front
        Sphere{radius: 1e5,   position: DVec3::new( 50.0, 1e5, 81.6     ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.75,0.75,0.75   ), reflection: ReflectionType::DIFFUSE   },//Bottom
        Sphere{radius: 1e5,   position: DVec3::new( 50.0,-1e5+81.6,81.6 ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.75,0.75,0.75   ), reflection: ReflectionType::DIFFUSE   },//Top
        Sphere{radius: 16.5,  position: DVec3::new( 27.0,16.5,47.0      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.999,0.999,0.999), reflection: ReflectionType::SPECULAR  },//Mirror
        Sphere{radius: 16.5,  position: DVec3::new( 73.0,16.5,78.0      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.999,0.999,0.999), reflection: ReflectionType::REFRACTIVE},//Glass
        Sphere{radius: 1.0,   position: DVec3::new( 17.0,60.0,20.0      ), emission: DVec3::new(36.0, 36.0, 36.0), color: DVec3::new(0.0,0.0,0.0      ), reflection: ReflectionType::DIFFUSE   },//Left Light
        Sphere{radius: 8.0,   position: DVec3::new( 17.0,60.0,20.0      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.999,0.0,0.999  ), reflection: ReflectionType::REFRACTIVE},//Left Light Inner Glass
        Sphere{radius: 16.0,  position: DVec3::new( 17.0,60.0,20.0      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.999,0.999,0.0  ), reflection: ReflectionType::REFRACTIVE},//Left Light Outer Glass
        Sphere{radius: 2.0,   position: DVec3::new( 77.0,62.0,20.0      ), emission: DVec3::new(24.0, 24.0, 24.0), color: DVec3::new(0.0,0.0,0.0      ), reflection: ReflectionType::DIFFUSE   },//Right Light
        Sphere{radius: 10.0,  position: DVec3::new( 67.0,62.0,40.0      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.999,0.999,0.999), reflection: ReflectionType::DIFFUSE   },//Right Light Diffuse
        Sphere{radius: 6.0,   position: DVec3::new( 90.0,62.0,15.0      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.999,0.999,0.999), reflection: ReflectionType::SPECULAR  },//Right Light Diffuse
    ];

pub struct Light {
    pub radius: f64,
    pub position: DVec3,
    pub spheres_index: usize,
}

impl Light {
    pub fn get_random_point(&self) -> DVec3 {
        let mut rng = rand::thread_rng();
        let mut result: DVec3 = DVec3::new(rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>());
        result.normalize();

        result
    }
}
pub static LIGHTS: &'static [Light] = 
&[
    Light{radius: 1.0,   position: DVec3::new( 17.0,60.0,20.0      ), spheres_index: 8 },//Left Light
    Light{radius: 2.0,   position: DVec3::new( 77.0,62.0,20.0      ), spheres_index: 11 },//Right Light
];



#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: DVec3,
    pub direction: DVec3,
}

impl Ray {
    pub fn new(origin: DVec3, direction: DVec3) -> Self {
        Ray{origin, direction}
    }

    pub fn default() -> Self {
        Ray{origin: DVec3::zero(), direction: DVec3::zero() }
    }
}

pub enum ReflectionType {
    DIFFUSE,
    SPECULAR,
    REFRACTIVE,
}

pub struct Sphere {
    pub radius: f64,
    pub position: DVec3,
    pub emission: DVec3,
    pub color: DVec3,
    pub reflection: ReflectionType,
}

impl Sphere {
    pub fn new(radius: f64, position: DVec3, emission: DVec3, color: DVec3, reflection: ReflectionType ) -> Self {
        Sphere{radius, position, emission, color, reflection}
    }

    #[inline(always)]
    pub fn intersect(&self, ray: &Ray) -> f64 {
        let epsilon: f64 = 1e-4;
        let op : DVec3 = self.position - ray.origin;
        let b: f64 = op.dot(ray.direction);
        let det: f64 = b * b - op.dot(op) + self.radius * self.radius;

        if det < 0.0 { return 0.0 }
        let det: f64 = det.sqrt();

        let t: f64 = b - det;
        let result: f64 = 
            if epsilon < t { t } else {
                let t: f64 = b + det;
                if epsilon < t {
                    t
                } else {
                    0.0
                }
            };
        result
    }
}