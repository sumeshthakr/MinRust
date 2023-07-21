// src/activation/activation.rs


// Example function for demonstration
pub fn add(a: f64, b: f64) -> f64 {
    a + b
}

// Activation functions

/// Binary Step
pub fn binary_step(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Linear activation function
pub fn linear(x: f64) -> f64 {
    x
}

/// Sigmoid activation function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Tanh activation function
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Rectified Linear Unit (ReLU)
pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Leaky ReLU
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        alpha * x
    }
}

/// Parametric ReLU
pub fn parametric_relu(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        alpha * x
    }
}

/// Exponential Linear Unit (ELU)
pub fn elu(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        alpha * (x.exp() - 1.0)
    }
}

/// Swish activation function
pub fn swish(x: f64) -> f64 {
    x * sigmoid(x)
}

/// Softmax activation function
pub fn softmax(input: &[f64]) -> Vec<f64> {
    let max_val = input.iter().cloned().fold(std::f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = input.iter().map(|x| (x - max_val).exp()).collect();
    let sum_exp_values: f64 = exp_values.iter().sum();
    exp_values.iter().map(|x| x / sum_exp_values).collect()
}
