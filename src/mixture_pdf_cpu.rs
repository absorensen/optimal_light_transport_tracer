use std::f64::consts::PI;

use rand::Rng;
use ultraviolet::DVec3;

use crate::{shared::{Ray, SPHERES, Sphere, ReflectionType, LIGHTS, random_on_sphere, Light, OrthoNormalBase, random_cosine_direction}, cpu_tracer::{intersect, hit_specular, hit_refractive}};

#[inline(always)]
pub fn hit_lambertian(ray: &mut Ray, hit_normal: &DVec3, hit_position: &DVec3, pdf: &mut f64) {
    let mut uvw: OrthoNormalBase = OrthoNormalBase::new();
    uvw.build_from_w(hit_normal);
    ray.direction = uvw.local(&random_cosine_direction()).normalized(); 
    ray.origin = *hit_position;

    *pdf = uvw.axis[2].dot(ray.direction) / PI;
}


// Inspired by Peter Shirley's https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html
pub fn radiance_mixture_pdf(ray: Ray, max_depth: usize, monte_carlo_depth: usize) -> DVec3 {
    let mut t: f64 = 0.0;
    let mut id: usize = 0;
    let mut depth: usize = 0;
    let mut ray: Ray = ray.clone();
    
    let mut radiance: DVec3 = DVec3::broadcast(0.0);
    let mut attenuation: DVec3 = DVec3::broadcast(1.0);

    loop {
        if !intersect(&ray, &mut t, &mut id) { break; };

        let hit_object: &Sphere = &SPHERES[id];
        radiance += attenuation * hit_object.emission;


        let hit_position: DVec3 = ray.origin + ray.direction * t;
        let hit_normal: DVec3 = (hit_position - hit_object.position).normalized();
        let hit_normal_corrected: DVec3 = if hit_normal.dot(ray.direction) < 0.0 { hit_normal } else { -1.0 * hit_normal };

        let mut cos_pdf: f64 = 1.0;
        let mut light_pdf: f64 = 1.0;
        let mut scattering_pdf: f64 = 1.0;
        // Get new direction
        match hit_object.reflection {
            ReflectionType::DIFFUSE => {
                // After this the ray variable assumes the same function as the one usually called scattered in Shirley
                hit_lambertian(&mut ray, &hit_normal_corrected, &hit_position, &mut cos_pdf);

                // Hittable (Light) sampling
                let light_index: usize = rand::thread_rng().gen_range(0..LIGHTS.len()) as usize;
                let light: &Light = &LIGHTS[light_index];
                let on_light: DVec3 = random_on_sphere(light.radius, &light.position);
                let mut to_light: DVec3 = on_light - hit_position;
                let light_distance_squared: f64 = to_light.mag_sq();
                to_light.normalize();


                // Cosine sampling
                let mut uvw: OrthoNormalBase = OrthoNormalBase::new();
                uvw.build_from_w(&hit_normal_corrected);
                let cosine_direction: DVec3 = uvw.local(&random_cosine_direction());
                ray.direction = cosine_direction;

                // Generate scattered direction
                if rand::thread_rng().gen_range(0.0..1.0) < 0.9 {
                    ray.direction = to_light;
                } else {
                    ray.direction = cosine_direction;
                }

                // Hittable (Light) PDF
                let cos_theta_max: f64 = (1.0 - light.radius * light.radius / light_distance_squared).sqrt();
                let solid_angle: f64 = 2.0 * PI * (1.0 - cos_theta_max);
                light_pdf = 1.0 / solid_angle;

                // Scattering PDF
                let cosine: f64 = hit_normal_corrected.dot(ray.direction) / PI;
                scattering_pdf = if cosine < 0.0 { 0.0 } else { cosine };
                

            },
            ReflectionType::SPECULAR => {
                hit_specular(&mut ray, &hit_normal_corrected, &hit_position);
            },
            ReflectionType::REFRACTIVE => {
                hit_refractive(&mut ray, &hit_normal, &hit_normal_corrected, &hit_position, &mut attenuation);
            },
        }

        let pdf: f64 = 0.5 * (cos_pdf + light_pdf);
        if pdf == 0.0 || pdf.is_nan() { break; }

        let contribution: DVec3 = scattering_pdf * hit_object.color / pdf;
        if contribution.x.is_nan() || contribution.y.is_nan() || contribution.z.is_nan() { break; }
        attenuation *= contribution;

        depth += 1;
        if monte_carlo_depth < depth {
            let p: f64 = attenuation.component_max();
            if p < rand::thread_rng().gen::<f64>() {
                break;
            }

            attenuation *= 1.0 / p;
        } 

        if max_depth <= depth {
            break;
        }
    }

    radiance
}