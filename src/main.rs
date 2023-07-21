// src/main.rs

mod lib;

use ndarray::{Array2, array};

fn main() {
    let nn = lib::NeuralNetwork::new();

    // Test input
    let input_data = array![[0.1, 0.2]];
    let input: Array2<f64> = input_data.into_dyn();
    let output = nn.forward(input);

    println!("Output: {:?}", output);
}
