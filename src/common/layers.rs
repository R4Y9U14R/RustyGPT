use crate::common::dense::*;

#[derive(Debug)]
pub struct FullyConnected {
    input_size: usize,
    output_size: usize,
    weights: Option<Vec<f32>>,
    bias: Option<Vec<f32>>,
}

impl FullyConnected {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            weights: None,
            bias: None,
        }
    }

    pub fn weights(mut self, weights: Vec<f32>) -> Self {
        self.weights = Some(weights);
        self
    }

    pub fn bias(mut self, bias: Vec<f32>) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn forward(self, x: &Vec<f32>) -> Vec<f32> {
        add(
            &matmul(&self.weights.unwrap(), &transpose(x, x.len(), 1), self.output_size, self.input_size, x.len(), 1),
            &self.bias.unwrap(),
            self.output_size,
            1
        )
    }
}
