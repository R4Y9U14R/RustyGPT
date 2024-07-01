use std::fs::File;
use std::io::ErrorKind;
use std::env;
use std::io::{self, Write};
use std::path::Path;
use image::{GenericImageView, Luma};

mod common;
use common::dense::*;
mod model;
use model::*;

fn load_image(path: &str) -> Vec<f32> {
    println!("{}", path);
    let img = image::open(path).expect("Failed to open image");

    let grayscale = img.to_luma8();

    let (width, height) = img.dimensions();

    println!("Image dimensions: {}x{}", width, height);

    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let pixel = grayscale.get_pixel(x, y);
            let Luma([gray]) = *pixel;
            pixels.push(gray);
        }
    }
    let pixels: Vec<f32> = pixels.iter().map(|&x| x as f32).collect();
    pixels
}

fn main() {
    println!("{:?}", env::current_dir());

    let mut path = String::new();
    print!("Enter file path to MNIST image >>> ");
    io::stdout().flush().unwrap();

    io::stdin()
        .read_line(&mut path)
        .expect("Failed to read file path");

    let path = path.trim();

    if Path::new(&path).exists() {
        println!("File exists at path: {}", path);
        let pixels = load_image(&path);
        
        let layers = construct_model_from_dir("src/weights/processed");

        // manual forward pass idfk wut im doing
        let mut x = layers[0].forward(&pixels);
        x = relu(&x, x.len(), 1);
        x = layers[1].forward(&x);
        x = relu(&x, x.len(), 1);
        x = layers[2].forward(&x);
        x = relu(&x, x.len(), 1);
        x = layers[3].forward(&x);
        x = softmax(&x, 10);

        println!("Prediction: {:?}", argmax(&x).unwrap());
    } else {
        println!("File \"{}\" does not exist", path);
    }
}
