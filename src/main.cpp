#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

// A tiny "AI": one artificial neuron that learns the AND rule.
//
// Each training example contains two inputs and the correct answer:
//   0 AND 0 = 0
//   0 AND 1 = 0
//   1 AND 0 = 0
//   1 AND 1 = 1

struct Example {
  double x1;
  double x2;
  double answer;
};

double sigmoid(double value) {
  // Converts any number into a value between 0 and 1.
  return 1.0 / (1.0 + std::exp(-value));
}

int main() {
  const std::array<Example, 4> training_data{{
      {0.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {1.0, 0.0, 0.0},
      {1.0, 1.0, 1.0},
  }};

  // These are the values the neuron will learn.
  double weight1 = 0.0;
  double weight2 = 0.0;
  double bias = 0.0;

  const double learning_rate = 1.0;
  const int epochs = 5000;

  for (int epoch = 0; epoch <= epochs; ++epoch) {
    double weight1_gradient = 0.0;
    double weight2_gradient = 0.0;
    double bias_gradient = 0.0;
    double total_loss = 0.0;

    for (const Example &example : training_data) {
      // 1. FORWARD PASS: make a prediction with the current parameters.
      const double score =
          example.x1 * weight1 + example.x2 * weight2 + bias;
      const double prediction = sigmoid(score);

      // 2. LOSS: measure how far the prediction is from the answer.
      // This is binary cross-entropy loss.
      constexpr double epsilon = 1e-12;
      total_loss +=
          -(example.answer * std::log(prediction + epsilon) +
            (1.0 - example.answer) * std::log(1.0 - prediction + epsilon));

      // 3. BACKPROPAGATION: calculate how each parameter affected the error.
      const double error = prediction - example.answer;
      weight1_gradient += error * example.x1;
      weight2_gradient += error * example.x2;
      bias_gradient += error;
    }

    const double example_count = static_cast<double>(training_data.size());

    // 4. GRADIENT DESCENT: move the parameters in the direction that
    // reduces the error.
    weight1 -= learning_rate * weight1_gradient / example_count;
    weight2 -= learning_rate * weight2_gradient / example_count;
    bias -= learning_rate * bias_gradient / example_count;

    if (epoch % 1000 == 0) {
      std::cout << "Epoch " << std::setw(4) << epoch
                << " | average loss = " << total_loss / example_count << '\n';
    }
  }

  std::cout << "\nLearned parameters:\n"
            << "weight1 = " << weight1 << '\n'
            << "weight2 = " << weight2 << '\n'
            << "bias    = " << bias << "\n\n";

  std::cout << "Predictions after training:\n";
  std::cout << std::fixed << std::setprecision(4);

  for (const Example &example : training_data) {
    const double prediction =
        sigmoid(example.x1 * weight1 + example.x2 * weight2 + bias);
    const int decision = prediction >= 0.5 ? 1 : 0;

    std::cout << example.x1 << " AND " << example.x2
              << " -> probability: " << prediction
              << ", decision: " << decision
              << ", correct answer: " << example.answer << '\n';
  }
}
