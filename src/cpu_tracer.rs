use std::f64::consts::PI;

use rand::Rng;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use ultraviolet::DVec3;

use crate::{shared::{RenderContext, Ray, SPHERES, clamp, LightTransportAlgorithm}, naive_cpu::radiance_naive, mixture_pdf_cpu::radiance_mixture_pdf, dynamic_sampling_cpu::radiance_dynamic_sampling};

#[inline(always)]
pub fn intersect(ray: &Ray, t: &mut f64, id: &mut usize) -> bool {
    let spheres_count: usize = SPHERES.len();
    let inf: f64 = 1e20;
    *t = 1e20;

    for index in 0..spheres_count {
        let d = SPHERES[index].intersect(ray);
        if d != 0.0 && d < *t {
            *t = d;
            *id = index;
        }
    }

    *t < inf
}


#[inline(always)]
pub fn hit_diffuse(ray: &mut Ray, hit_normal: &DVec3, hit_position: &DVec3) {
    let r1: f64 = 2.0 * PI * rand::thread_rng().gen::<f64>();
    let r2: f64 = rand::thread_rng().gen::<f64>();
    let r2s: f64 = r2.sqrt();

    let w: DVec3 = *hit_normal;
    let u: DVec3 = ((if 0.1 < w.x.abs() { DVec3::new(0.0, 1.0, 0.0 ) } else { DVec3::new(1.0, 0.0, 0.0) }).cross(w)).normalized();
    let v: DVec3 = w.cross(u);
    let d: DVec3 = (u * r1.cos() * r2s + v * r1.sin() * r2s + w * (1.0 - r2).sqrt()).normalized();

    ray.origin = *hit_position;
    ray.direction = d;
}


#[inline(always)]
pub fn hit_specular(ray: &mut Ray, hit_normal: &DVec3, hit_position: &DVec3) {
    ray.origin = *hit_position;
    ray.direction = ray.direction - *hit_normal * 2.0 * hit_normal.dot(ray.direction);
}

#[inline(always)]
pub fn hit_refractive(ray: &mut Ray, hit_normal: &DVec3, hit_normal_corrected: &DVec3, hit_position: &DVec3, attenuation: &mut DVec3) {
    let reflected_ray: Ray = Ray::new(*hit_position, ray.direction - *hit_normal * 2.0 * hit_normal.dot(ray.direction));
    let into: bool = 0.0 < hit_normal.dot(*hit_normal_corrected);
    let nc: f64 = 1.0;
    let nt: f64 = 1.5;
    let nnt: f64 = if into { nc / nt } else { nt / nc };
    let ddn: f64 = ray.direction.dot(*hit_normal_corrected);
    let cos2t: f64 = 1.0 - nnt * nnt * ( 1.0 - ddn * ddn);

    if cos2t < 0.0 {
        *ray = reflected_ray;
        return;
    }

    let tdir: DVec3 = (ray.direction * nnt - *hit_normal * ((if into {1.0} else {-1.0}) * (ddn * nnt + cos2t.sqrt()))).normalized();
    let a: f64 = nt - nc;
    let b: f64 = nt + nc;
    let r0: f64 = a * a / (b * b);
    let c: f64 = 1.0 - (if into { -ddn } else { tdir.dot(*hit_normal)});
    let re: f64 = r0 + (1.0 - r0) * c * c * c * c * c;
    let tr: f64 = 1.0 - re;
    let p: f64 = 0.25 + 0.5 * re;
    let rp: f64 = re / p;
    let tp: f64 = tr / ( 1.0 - p);

    if rand::thread_rng().gen::<f64>() < p {
        *attenuation = *attenuation * rp;
        *ray = reflected_ray;
    } else {
        *attenuation = *attenuation * tp;
        ray.origin = *hit_position;
        ray.direction = tdir;
    }
}


#[inline(always)]
pub fn generate_subpixel_origin(
    index: usize, 
    subpixels_count: usize, 
    subsamples_count: usize, 
    seed0: f64, 
    seed1: f64, 
    cx: DVec3, 
    cy: DVec3, 
    subpixels_offset: f64, 
    camera: &Ray, 
    column_index: usize, 
    row_index: usize, 
    width: usize, 
    height: usize
) -> DVec3 {
    let sx: usize = (index % (subpixels_count * subsamples_count)) / subsamples_count;
    let sy: usize = index / (subsamples_count * subpixels_count);

    let r1: f64 = 2.0 * seed0;
    let dx: f64 = if r1 < 1.0 { r1.sqrt() - 1.0 } else { 1.0 - (2.0 - r1).sqrt() };

    let r2: f64 = 2.0 * seed1;
    let dy: f64 = if r2 < 1.0 { r2.sqrt() - 1.0 } else { 1.0 - (2.0 - r2).sqrt() };

    let d: DVec3 = cx * ( ( ( (sx as f64) + 0.5 + (dx as f64)) * subpixels_offset + (column_index as f64)) / (width as f64) - 0.5 ) +
                    cy * ( ( ( (sy as f64) + 0.5 + (dy as f64)) * subpixels_offset + (row_index as f64)) / (height as f64) - 0.5) + camera.direction;
                    
    d
}

pub fn naive_path_trace_cpu(context: RenderContext) -> Vec<DVec3> {
    (0..context.total_pixels).into_par_iter().map(|pixel_index:usize| {
        let mut rng = rand::thread_rng();
        let samples: Vec<(usize, f64, f64)> = (0..(context.subsamples_count * context.subpixels_count * context.subpixels_count)).into_iter().map(|index| (index, rng.gen::<f64>(), rng.gen::<f64>()) ).collect();
        let column_index: usize = pixel_index % context.width;
        let row_index: usize = context.height - pixel_index / context.width - 1;
        let mut output_color: DVec3 = samples.into_iter().map(|(index, seed0, seed1)| {
            let d = generate_subpixel_origin(index, context.subpixels_count, context.subsamples_count, seed0, seed1, context.cx, context.cy, context.subpixels_offset, &context.camera, column_index, row_index, context.width, context.height);
            let camera_ray: Ray = Ray::new(context.camera.origin+d*140.0, d.normalized());
            let contribution: DVec3 = match context.algorithm {
                LightTransportAlgorithm::NAIVE => radiance_naive(camera_ray, context.max_depth, context.monte_carlo_depth),
                LightTransportAlgorithm::MIXTURE_PDF => radiance_mixture_pdf(camera_ray, context.max_depth, context.monte_carlo_depth),
                LightTransportAlgorithm::DYNAMIC_SAMPLING => radiance_dynamic_sampling(camera_ray, context.max_depth, context.monte_carlo_depth),
                _ => radiance_naive(camera_ray, context.max_depth, context.monte_carlo_depth),
            }; 
            contribution * context.sample_scale
        }).sum();

        output_color.x = clamp(output_color.x);
        output_color.y = clamp(output_color.y); 
        output_color.z = clamp(output_color.z);

        output_color
    }).collect()
} 