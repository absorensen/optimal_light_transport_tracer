use std::{mem::size_of, f64::consts::PI};

use minifb::{ScaleMode, Window, WindowOptions, Key};
use rand::Rng;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use ultraviolet::DVec3;

#[derive(Debug, Clone)]
pub struct Ray {
    origin: DVec3,
    direction: DVec3,
}

impl Ray {
    pub fn new(origin: DVec3, direction: DVec3) -> Self {
        Ray{origin, direction}
    }
}

pub enum ReflectionType {
    DIFFUSE,
    SPECULAR,
    REFRACTIVE,
}

pub struct Sphere {
    radius: f64,
    position: DVec3,
    emission: DVec3,
    color: DVec3,
    reflection: ReflectionType,
}

impl Sphere {
    pub fn new(radius: f64, position: DVec3, emission: DVec3, color: DVec3, reflection: ReflectionType ) -> Self {
        Sphere{radius, position, emission, color, reflection}
    }

    pub fn intersect(&self, ray: &Ray) -> f64 {
        let op : DVec3 = self.position - ray.origin;
        let epsilon: f64 = 1e-4;
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

static SPHERES: &'static [Sphere] = 
    &[
        Sphere{radius: 1e5,   position: DVec3::new( 1e5+1.0,40.8,81.6   ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.75,0.25,0.25   ), reflection: ReflectionType::DIFFUSE},//Left
        Sphere{radius: 1e5,   position: DVec3::new( -1e5+99.0,40.8,81.6 ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.25,0.25,0.75   ), reflection: ReflectionType::DIFFUSE},//Right
        Sphere{radius: 1e5,   position: DVec3::new( 50.0,40.8, 1e5      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.75, 0.75, 0.75 ), reflection: ReflectionType::DIFFUSE},//Back
        Sphere{radius: 1e5,   position: DVec3::new( 50.0,40.8,-1e5+170.0), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.0,0.0,0.0      ), reflection: ReflectionType::DIFFUSE},//Front
        Sphere{radius: 1e5,   position: DVec3::new( 50.0, 1e5, 81.6     ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.75,0.75,0.75   ), reflection: ReflectionType::DIFFUSE},//Bottom
        Sphere{radius: 1e5,   position: DVec3::new( 50.0,-1e5+81.6,81.6 ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.75,0.75,0.75   ), reflection: ReflectionType::DIFFUSE},//Top
        Sphere{radius: 16.5,  position: DVec3::new( 27.0,16.5,47.0      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.999,0.999,0.999), reflection: ReflectionType::SPECULAR},//Mirror
        Sphere{radius: 16.5,  position: DVec3::new( 73.0,16.5,78.0      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.999,0.999,0.999), reflection: ReflectionType::REFRACTIVE},//Glass
        Sphere{radius: 1.0,   position: DVec3::new( 17.0,60.0,20.0      ), emission: DVec3::new(36.0, 36.0, 36.0), color: DVec3::new(0.0,0.0,0.0      ), reflection: ReflectionType::DIFFUSE},//Left Light
        Sphere{radius: 8.0,   position: DVec3::new( 17.0,60.0,20.0      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.999,0.0,0.999  ), reflection: ReflectionType::REFRACTIVE},//Left Light Inner Glass
        Sphere{radius: 16.0,  position: DVec3::new( 17.0,60.0,20.0      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.999,0.999,0.0  ), reflection: ReflectionType::REFRACTIVE},//Left Light Outer Glass
        Sphere{radius: 2.0,   position: DVec3::new( 77.0,62.0,20.0      ), emission: DVec3::new(24.0, 24.0, 24.0), color: DVec3::new(0.0,0.0,0.0      ), reflection: ReflectionType::DIFFUSE},//Right Light
        Sphere{radius: 10.0,  position: DVec3::new( 67.0,62.0,40.0      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.999,0.999,0.999), reflection: ReflectionType::DIFFUSE},//Right Light Diffuse
        Sphere{radius: 6.0,  position: DVec3::new( 90.0,62.0,15.0      ), emission: DVec3::new(0.0, 0.0, 0.0   ), color: DVec3::new(0.999,0.999,0.999), reflection: ReflectionType::SPECULAR},//Right Light Diffuse
    ];

pub fn clamp(x: f64) -> f64 {
    let result: f64 = 
        if x < 0.0 { 0.0 } else {
            if 1.0 < x { 1.0 } else { x }
        };
    result
}

pub fn to_int(x: f64) -> i32 {
    (clamp(x).powf(1.0 / 2.2) * 255.0 + 0.5) as i32
}

pub fn intersect(ray: &Ray, t: &mut f64, id: &mut usize) -> bool {
    let n: f64 = (SPHERES.len() / size_of::<Sphere>()) as f64;
    let spheres_count = SPHERES.len();
    // Maybe just do this in the loop
    let mut d: f64 = 0.0;
    let inf: f64 = 1e20;
    *t = 1e20;

    for index in (0..(spheres_count as usize)).rev() {
        d = SPHERES[index].intersect(ray);
        if d != 0.0 && d < *t {
            *t = d;
            *id = index;
        } else {
        }

    }

    *t < inf
}

pub fn radiance(ray: &Ray, depth: i32) -> DVec3 {
    let mut t: f64 = 0.0;
    let mut id: usize = 0;
    let mut depth: i32 = depth;
    let mut ray: Ray = ray.clone();
    
    let mut cl: DVec3 = DVec3::broadcast(0.0);
    let mut cf: DVec3 = DVec3::broadcast(1.0);

    loop {
        if !intersect(&ray, &mut t, &mut id) { return cl; };

        let hit_object: &Sphere = &SPHERES[id];
        let x: DVec3 = ray.origin + ray.direction * t;
        let n: DVec3 = (x - hit_object.position).normalized();
        let nl: DVec3 = if n.dot(ray.direction) < 0.0 { n } else { -1.0 * n };
        let mut f: DVec3 = hit_object.color.clone();
        let p: f64 = f.component_max();
        cl += cf * hit_object.emission;

        depth += 1;
        if 5 < depth {
            if rand::thread_rng().gen::<f64>() < p { f = f * (1.0 / p) } else { 
                return cl; 
            }
        }

        cf = cf * f;

        match hit_object.reflection {
            ReflectionType::DIFFUSE => {
                let r1: f64 = 2.0 * PI * rand::thread_rng().gen::<f64>();
                let r2: f64 = rand::thread_rng().gen::<f64>();
                let r2s: f64 = r2.sqrt();

                let w: DVec3 = nl;
                let u: DVec3 = ((if 0.1 < w.x.abs() { DVec3::new(0.0, 1.0, 0.0 ) } else { DVec3::new(1.0, 0.0, 0.0) }).cross(w)).normalized();
                let v: DVec3 = w.cross(u);
                let d: DVec3 = (u * r1.cos() * r2s + v * r1.sin() * r2s + w * (1.0 - r2).sqrt()).normalized();

                ray.origin = x;
                ray.direction = d;

                continue;
            },
            ReflectionType::SPECULAR => {
                ray.origin = x;
                ray.direction = ray.direction - n * 2.0 * n.dot(ray.direction);
                continue;
            },
            ReflectionType::REFRACTIVE => {
                let reflected_ray: Ray = Ray::new(x, ray.direction - n * 2.0 * n.dot(ray.direction));
                let into: bool = 0.0 < n.dot(nl);
                let nc: f64 = 1.0;
                let nt: f64 = 1.5;
                let nnt: f64 = if into { nc / nt } else { nt / nc };
                let ddn: f64 = ray.direction.dot(nl);
                let cos2t: f64 = 1.0 - nnt * nnt * ( 1.0 - ddn * ddn);

                if cos2t < 0.0 {
                    ray = reflected_ray;
                    continue;
                }

                let tdir: DVec3 = (ray.direction * nnt - n * ((if into {1.0} else {-1.0}) * (ddn * nnt + cos2t.sqrt()))).normalized();
                let a: f64 = nt - nc;
                let b: f64 = nt + nc;
                let R0: f64 = a * a / (b * b);
                let c: f64 = 1.0 - (if into { -ddn } else { tdir.dot(n)});
                let Re: f64 = R0 + (1.0 - R0) * c * c * c * c * c;
                let Tr: f64 = 1.0 - Re;
                let P: f64 = 0.25 + 0.5 * Re;
                let RP: f64 = Re / P;
                let TP: f64 = Tr / ( 1.0 - P);

                if rand::thread_rng().gen::<f64>() < P {
                    cf = cf * RP;
                    ray = reflected_ray;
                } else {
                    cf = cf * TP;
                    ray.origin = x;
                    ray.direction = tdir;
                }

                continue;
            },
        }

    }

    cl
}

pub fn run() -> bool {
    let width: i64 = 512;
    let height: i64 = 384;
    let subpixels_count: i32 = 8;
    let subpixels_offset: f64 = 1.0 / subpixels_count as f64;
    let subpixels_scale: f64 = 1.0 / (subpixels_count * subpixels_count) as f64;
    let subsamples_count: i32 = 2;
    let subsampling_scale: f64 = 1.0 / subsamples_count as f64;


    let camera: Ray = Ray::new(DVec3::new(50.0, 52.0, 295.6), DVec3::new(0.0, -0.042612, -1.0).normalized());
    let cx: DVec3 = DVec3::new((width as f64) * 0.5135 / (height as f64), 0.0, 0.0);
    let cy: DVec3 = cx.cross(camera.direction).normalized() * 0.5135;

    // let mut image_buffer: Vec<f64> = vec![0.0; (width * height * colors) as usize];

    let total_pixels = height * width;
    let image_buffer: Vec<DVec3> = 
        (0..total_pixels).into_par_iter().map(|pixel_index:i64| {
            let mut output_color: DVec3 = DVec3::zero();
            let column_index = pixel_index % width;
            let row_index = height - pixel_index / width - 1;
            for sy in 0..subpixels_count {
                for sx in 0..subpixels_count {
                    let mut color: DVec3 = DVec3::zero();
                    for _sample_index in 0..subsamples_count {
                        let r1 = 2.0 * rand::thread_rng().gen::<f64>();
                        let dx = if r1 < 1.0 { r1.sqrt() - 1.0 } else { 1.0 - (2.0 - r1).sqrt() };

                        let r2 = 2.0 * rand::thread_rng().gen::<f64>();
                        let dy = if r2 < 1.0 { r2.sqrt() - 1.0 } else { 1.0 - (2.0 - r2).sqrt() };

                        let d: DVec3 = cx * ( ( ( (sx as f64) + 0.5 + (dx as f64)) * subpixels_offset + (column_index as f64)) / (width as f64) - 0.5 ) +
                                       cy * ( ( ( (sy as f64) + 0.5 + (dy as f64)) * subpixels_offset + (row_index as f64)) / (height as f64) - 0.5) + camera.direction; 
                    
                        let contribution: DVec3 = radiance(&Ray::new(camera.origin+d*140.0, d.normalized()), 0); 
                        color += contribution * subsampling_scale;
                    }

                    output_color.x += clamp(color.x) * subpixels_scale;
                    output_color.y += clamp(color.y) * subpixels_scale;
                    output_color.z += clamp(color.z) * subpixels_scale;
                }
            }
            output_color
        
        }).collect();

    let window_buffer: Vec<u32> = image_buffer.into_par_iter()
        .map(|v| ((to_int(v.x) as u32) << 16) | ((to_int(v.y) as u32) << 8) | to_int(v.z) as u32)
        .collect();

    let window_name : String = format!("Optimal Light Transporter - {} samples per pixel - Press ESC to exit", subsamples_count * subpixels_count * subpixels_count);
    let mut window = Window::new(
        window_name.as_str(),
        width as usize,
        height as usize,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::Center,
            ..WindowOptions::default()
        },
    )
    .expect("Unable to open Window");


    while window.is_open() && !window.is_key_down(Key::Escape) {
        window.update_with_buffer(
                &window_buffer,
                width as usize,
                height as usize,
            ).unwrap();
    }

    true
}