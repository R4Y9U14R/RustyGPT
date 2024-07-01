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

    pub fn weights(mut self, weights: Option<Vec<f32>>) -> Self {
        self.weights = weights;
        self
    }

    pub fn bias(mut self, bias: Option<Vec<f32>>) -> Self {
        self.bias = bias;
        self
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let weights = self.weights.as_ref().expect("Weights not set");
        let bias = self.bias.as_ref().expect("Bias not set");

        assert_eq!(x.len(), self.input_size, "Input size mismatched");

        let matmul_result = matmul(&weights, &x, self.output_size, self.input_size, x.len(), 1);
        assert_eq!(matmul_result.len(), self.output_size, "Matmul output size mismatch");

        let result = add(&matmul_result, &bias, self.output_size, 1);

        result
    }
}
