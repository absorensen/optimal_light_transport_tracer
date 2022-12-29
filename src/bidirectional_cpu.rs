use rand::Rng;
use ultraviolet::DVec3;

use crate::{shared::{Ray, SPHERES, Sphere, ReflectionType, LIGHTS, Light, random_on_sphere, OrthoNormalBase, random_cosine_direction}, cpu_tracer::{intersect, hit_diffuse, hit_specular, hit_refractive}};

pub fn generate_ray_vertices(initial_ray: Ray, spheres_index: usize, max_depth: usize, vertex_list: &mut Vec<(Ray, usize)>) {
    vertex_list.push((initial_ray.clone(), spheres_index));

    let mut ray: Ray = initial_ray.clone();
    let mut id: usize = 0;
    let mut t: f64 = 0.0;
    let mut attenuation: DVec3 = DVec3::zero();
    for _ in 0..max_depth {
        if !intersect(&ray, &mut t, &mut id) { break; };

        let hit_object: &Sphere = &SPHERES[id];
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

        vertex_list.push((ray.clone(), id));
    }
}

pub fn visibility_test(vertex_a: &DVec3, vertex_b: &DVec3) -> bool {
    let a_to_b: DVec3 = *vertex_b - *vertex_a;
    let direction: DVec3 = a_to_b.normalized(); 
    let distance: f64 = a_to_b.mag();

    let ray: Ray = Ray::new(vertex_a.clone(), direction);
    let mut t: f64 = 0.0;
    let mut id: usize = 0;

    if !intersect(&ray, &mut t, &mut id) { return false };

    distance <= t
}

pub fn radiance_bidirectional(ray: Ray, max_depth: usize, _: usize) -> DVec3 {
    // Generate vertices from camera
    let max_depth_from_camera: usize = rand::thread_rng().gen_range(0..max_depth) as usize;
    let mut camera_vertices: Vec<(Ray, usize)> = Vec::<(Ray, usize)>::new();
    generate_ray_vertices(ray, 0, max_depth_from_camera, &mut camera_vertices);

    // Generate vertices from light
    let light_index: usize = rand::thread_rng().gen_range(0..LIGHTS.len()) as usize;
    let light: &Light = &LIGHTS[light_index];
    let on_light: DVec3 = random_on_sphere(light.radius, &light.position);
    let from_light: DVec3 = (on_light - light.position).normalized(); // Maybe try random cosine direction using this vector as the normal
    let mut uvw: OrthoNormalBase = OrthoNormalBase::new();
    uvw.build_from_w(&from_light);
    let from_light: DVec3 = uvw.local(&random_cosine_direction());


    let max_depth_from_light: usize = rand::thread_rng().gen_range(0..max_depth) as usize;
    let mut light_vertices: Vec<(Ray, usize)> = Vec::<(Ray, usize)>::new();
    generate_ray_vertices(Ray::new(on_light, from_light), light.spheres_index, max_depth_from_light, &mut light_vertices);

    // Visibility testing and connection of vertices
    let mut light_paths: Vec<(usize, usize)> = Vec::<(usize, usize)>::new();
    for camera_vertex_index in 0..camera_vertices.len() {
        for light_vertex_index in 0..light_vertices.len() {
            if visibility_test(&camera_vertices[camera_vertex_index].0.origin, &light_vertices[light_vertex_index].0.origin) {
                light_paths.push((camera_vertex_index, light_vertex_index));
            }
        }
    }

    if light_paths.len() == 0 {
        return DVec3::zero()
    }

    // Follow paths
    let mut final_radiance: DVec3 = DVec3::broadcast(0.0);
    for (last_vertex_index_camera, first_vertex_index_light) in &light_paths {
        let mut radiance: DVec3 = DVec3::broadcast(0.0);
        let mut attenuation: DVec3 = DVec3::broadcast(1.0);

        for camera_vertex_index in 1..*last_vertex_index_camera {
            let id: usize = camera_vertices[camera_vertex_index].1;
            let hit_object: &Sphere = &SPHERES[id];
            radiance += attenuation * hit_object.emission;
            attenuation *= hit_object.color;
        }

        for light_vertex_index in *first_vertex_index_light..0 {
            let id: usize = light_vertices[light_vertex_index].1;
            let hit_object: &Sphere = &SPHERES[id];
            radiance += attenuation * hit_object.emission;
            attenuation *= hit_object.color;
        }

        final_radiance += radiance;
    }

    final_radiance / light_paths.len() as f64
}