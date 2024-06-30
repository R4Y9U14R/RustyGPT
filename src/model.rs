use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;
use regex::Regex;

use rusty_gpt::common::layers::*;
use rusty_gpt::utils::*;

pub fn read_from_bin<P: AsRef<Path>>(path: P) -> io::Result<Vec<f32>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let float_count = buffer.len() / std::mem::size_of::<f32>();
    let mut floats = Vec::with_capacity(float_count);

    for chunk in buffer.chunks_exact(std::mem::size_of::<f32>()) {
        let float = f32::from_le_bytes(chunk.try_into().expect("Slice with incorrect length"));
        floats.push(float);
    }

    Ok(floats)
}

pub enum ParamType {
    Weights,
    Bias,
}

impl ParamType {
    fn from_str(s: &str) -> Option<ParamType> {
        match s.to_lowercase().as_str() {
            "weight" => Some(ParamType::Weights),
            "bias" => Some(ParamType::Bias),
            _ => None,
        }
    }
}

pub struct ParamInfo {
    filename: String,
    number: u32,
    param_type: ParamType,
    shape: (usize, usize),
}

impl ParamInfo {
    fn new(filename: &str, number: Option<u32>, param_type: &str, num_rows: u32, num_cols: u32) -> Self {
        Self {
            filename: filename.to_string(),
            number: number.unwrap(),
            param_type: ParamType::from_str(param_type).unwrap(),
            shape: (num_rows as usize, num_cols as usize),
        }
    }
}

fn extract_param_info<'a>(filename_str: &'a str, re: &Regex) -> Option<ParamInfo> {
    if let Some(captures) = re.captures(filename_str) {
        let layer_number = captures.name("layer_number")?.as_str().parse::<u32>().ok();
        let param_type = captures.name("param_type")?.as_str().to_string();
        let num_rows = captures.name("num_rows")?.as_str().parse::<u32>().ok();
        let num_cols = captures.name("num_cols")?.as_str().parse::<u32>().ok();

        Some(ParamInfo::new(filename_str, layer_number, param_type.as_str(), num_rows.unwrap(), num_cols.unwrap()))
    } else {
        None
    }
}

fn sort_files_by_layer_number<'a, P: AsRef<Path>>(
    path: P,re: Regex
) -> Result<(Vec<ParamInfo>, Vec<ParamInfo>), Box<dyn std::error::Error>> {
    let mut weights_files = vec![];
    let mut biases_files = vec![];
    for entry in fs::read_dir(path)? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if let Some(filename) = path.file_name() {
            if let Some(filename_str) = filename.to_str() {
                if let Some(file) = extract_param_info(filename_str, &re) {
                    match file.param_type {
                        ParamType::Weights => weights_files.push(file),
                        ParamType::Bias => biases_files.push(file),
                    }
                }
            }
        }
    }

    weights_files.sort_by_key(|file: &ParamInfo| file.number);
    biases_files.sort_by_key(|file: &ParamInfo| file.number);

    Ok((weights_files, biases_files))
}

pub fn construct_model_from_dir<P: AsRef<Path>>(path: P) {
    // First iter through the file system
    // Then extract layer number, layer type, shape from bin file name
    // Finally use this info to construct respective models

    let re = Regex::new(r"(?P<layer_type>\w+)(?p<layer_number>\d+)\.(?P<param_type>\w+)-\(?P<num_rows>\d+, ?<num_cols>\d+\)\.bin")
        .expect("failed to create regex for parameter file sorting");
    let (weights_files, biases_files) = sort_files_by_layer_number(&path, re).expect("Failed to sort paramter files");
    let mut sequential_model = vec![];

    for layer in 0..(count_files_in_dir(&path).unwrap()) {
        let current_weight_file = &weights_files[layer];
        let current_bias_file = &biases_files[layer];
        
        let weights = read_from_bin(path.as_ref().join(current_weight_file.filename.clone())).ok();
        let bias = read_from_bin(path.as_ref().join(current_bias_file.filename.clone())).ok();

        let layer = FullyConnected::new(weights_files[layer].shape.0, weights_files[layer].shape.1)
            .weights(weights)
            .bias(bias);
        sequential_model.push(layer);
    }
}
