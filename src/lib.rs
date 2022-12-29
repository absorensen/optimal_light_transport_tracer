use std::{time::Instant};

use minifb::{ScaleMode, Window, WindowOptions, Key};
use ultraviolet::DVec3;

use crate::{cpu_tracer::naive_path_trace_cpu, shared::{RenderContext, to_int}};

mod cpu_tracer;
mod naive_cpu;
mod mixture_pdf_cpu;
mod dynamic_sampling_cpu;
mod bidirectional_cpu;
mod gpu_tracer;
mod shared;




pub fn run() -> bool {
    let context: RenderContext = RenderContext::default();
    let now: Instant = Instant::now();

    let image_buffer: Vec<DVec3> = naive_path_trace_cpu(context.clone());

    let total_samples: usize = context.subpixels_count * context.subpixels_count * context.subsamples_count;
    println!("{} seconds elapsed tracing {} samples", now.elapsed().as_millis() as f32 * 0.001, total_samples);

    let window_buffer: Vec<u32> = image_buffer.iter()
        .map(|v| ((to_int(v.x) as u32) << 16) | ((to_int(v.y) as u32) << 8) | to_int(v.z) as u32)
        .collect();

    let window_name : String = format!("Optimal Light Transporter - {} samples per pixel - Press ESC to exit", total_samples);
    let mut window: Window = Window::new(
        window_name.as_str(),
        context.width as usize,
        context.height as usize,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::Center,
            ..WindowOptions::default()
        },
    )
    .expect("Unable to open Window");

    let width: usize = context.width as usize;
    let height: usize = context.height as usize;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        window.update_with_buffer(
                &window_buffer,
                width,
                height,
            ).unwrap();
    }

    let ouput_buffer: Vec<u8> = 
        image_buffer.iter()
            .flat_map(|vector| [to_int(vector.x) as u8, to_int(vector.y) as u8, to_int(vector.z) as u8])
            .collect();



    let output_path: &str = "output.png";
    let save_result = image::save_buffer_with_format(
        output_path, 
        &ouput_buffer, 
        context.width.try_into().unwrap(), 
        context.height.try_into().unwrap(), 
        image::ColorType::Rgb8, 
        image::ImageFormat::Png
    );

    if save_result.is_ok() {
        println!("Saved output image to {}", output_path);
    } else {
        let error = save_result.unwrap_err();
        panic!("{}", error.to_string());
    }

    true
}