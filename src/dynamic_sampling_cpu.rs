use std::f64::consts::PI;

use rand::Rng;
use ultraviolet::DVec3;

use crate::{shared::{Ray, SPHERES, Sphere, ReflectionType, LIGHTS, Light, random_on_sphere}, cpu_tracer::{intersect, hit_diffuse, hit_specular, hit_refractive}};

pub fn radiance_dynamic_sampling(ray: Ray, max_depth: usize, monte_carlo_depth: usize) -> DVec3 {
    let mut t: f64 = 0.0;
    let mut id: usize = 0;
    
    let mut radiance: DVec3 = DVec3::broadcast(0.0);
    let mut ray_queue: Vec<(usize, DVec3, Ray)> = Vec::<(usize, DVec3, Ray)>::new();
    ray_queue.push((0, DVec3::broadcast(1.0), ray.clone()));

    let extra_diffuse_samples_count: i32 = 2;
    let extra_diffuse_bounce_max_depth: usize = 2;
    let light_sample_count: i32 = 1;

    let mut total_samples: i32 = 0;
    while 0 < ray_queue.len() {
        let (mut depth, mut attenuation, mut ray) = ray_queue.pop().unwrap();
        total_samples += 1;
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
                    // If N first bounces create multiple new rays for diffuse surfaces and add them to the queue
                    if depth < extra_diffuse_bounce_max_depth {
                        let initial_ray: Ray = ray.clone();

                        for _ in 0..extra_diffuse_samples_count {
                            let mut new_ray: Ray = initial_ray.clone();
                            hit_diffuse(&mut new_ray, &hit_normal_corrected, &hit_position);
                            let new_attenuation: DVec3 = attenuation * hit_object.color;
                            ray_queue.push((depth + 1, new_attenuation, new_ray));
                        }
                    }
                    
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

            match hit_object.reflection {
                ReflectionType::SPECULAR => {},
                ReflectionType::REFRACTIVE => {},
                _ => {
                    // Generate light sampling rays
                    for _ in 0..light_sample_count {
                        let light_index: usize = rand::thread_rng().gen_range(0..LIGHTS.len()) as usize;
                        let light: &Light = &LIGHTS[light_index];
                        let on_light: DVec3 = random_on_sphere(light.radius, &light.position);
                        let mut to_light: DVec3 = on_light - hit_position;
                        let light_distance_squared: f64 = to_light.mag_sq();
                        to_light.normalize();
                        let new_ray: Ray = Ray::new(hit_position, to_light);
                        if intersect(&new_ray, &mut t, &mut id) { 
                            if id == light.spheres_index {
                                let cos_theta_max: f64 = (1.0 - light.radius * light.radius / light_distance_squared).sqrt();
                                let solid_angle: f64 = 2.0 * PI * (1.0 - cos_theta_max);
                                let light_pdf: f64 = 1.0 / solid_angle;
                                if light_pdf < 0.0000001 { continue; }
                                let contribution: DVec3 = 0.5 * SPHERES[light.spheres_index].emission * attenuation / light_pdf;
                                if contribution.x.is_nan() || contribution.y.is_nan() || contribution.z.is_nan() { continue; }
                                radiance += contribution;
                                total_samples += 1;
                            }

                        };
                    }
                }
            }

            if max_depth <= depth {
                break;
            }
        }
    }

    radiance / total_samples as f64
}