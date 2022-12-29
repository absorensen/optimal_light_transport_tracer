use rand::Rng;
use ultraviolet::DVec3;

use crate::{shared::{Ray, SPHERES, Sphere, ReflectionType}, cpu_tracer::{intersect, hit_diffuse, hit_specular, hit_refractive}};

pub fn radiance_naive(ray: Ray, max_depth: usize, monte_carlo_depth: usize) -> DVec3 {
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

        // Get new direction
        match hit_object.reflection {
            ReflectionType::DIFFUSE => {
                hit_diffuse(&mut ray, &hit_normal_corrected, &hit_position);
            },
            ReflectionType::SPECULAR => {
                hit_specular(&mut ray, &hit_normal_corrected, &hit_position);
            },
            ReflectionType::REFRACTIVE => {
                hit_refractive(&mut ray, &hit_normal, &hit_normal_corrected, &hit_position, &mut attenuation);
            },
        }

        attenuation *= hit_object.color;

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