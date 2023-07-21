// src/lib.rs

use ndarray::{Array2, arr2};

pub struct NeuralNetwork {
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        // Initialize the neural network with random weights
        let weights_input_hidden = arr2(&[[0.1, 0.4], [0.3, 0.6]]);
        let weights_hidden_output = arr2(&[[0.2], [0.5]]);

        NeuralNetwork {
            weights_input_hidden,
            weights_hidden_output,
        }
    }

    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn forward(&self, input: Array2<f64>) -> Array2<f64> {
        // Calculate the hidden layer output
        let hidden_layer_input = input.dot(&self.weights_input_hidden);
        let hidden_layer_output = hidden_layer_input.mapv(NeuralNetwork::sigmoid);

        // Calculate the output layer output
        let output_layer_input = hidden_layer_output.dot(&self.weights_hidden_output);
        let output_layer_output = output_layer_input.mapv(NeuralNetwork::sigmoid);

        output_layer_output
    }
}
