package test;

import activation.Sigmoid;
import core.Matrix;
import layers.DenseLayer;
import loss.MeanSquaredError;
import optimiser.Adam;

public class XorTest {
    public static void main(String[] args) {
        System.out.println("=== XOR Neural Network Test ===\n");

        // Create the network
        // Architecture: 2 input → 4 hidden (sigmoid) → 1 output (sigmoid)
        DenseLayer hiddenLayer = new DenseLayer(2, 4);
        Sigmoid hiddenActivation = new Sigmoid();
        DenseLayer outputLayer = new DenseLayer(4, 1);
        Sigmoid outputActivation = new Sigmoid();

        // XOR training data
        // 4 examples, each with 2 inputs
        double[][] inputs = {
                {0.0, 0.0},  // XOR(0,0) = 0
                {0.0, 1.0},  // XOR(0,1) = 1
                {1.0, 0.0},  // XOR(1,0) = 1
                {1.0, 1.0}   // XOR(1,1) = 0
        };

        double[][] targets = {
                {0.0},  // Expected output for [0,0]
                {1.0},  // Expected output for [0,1]
                {1.0},  // Expected output for [1,0]
                {0.0}   // Expected output for [1,1]
        };

        // Training parameters
        double learningRate = 0.5;  // Higher for XOR (simple problem)
        int epochs = 10000;
        int printEvery = 1000;
        Adam adam = new Adam(learningRate);

        System.out.println("Network Architecture:");
        System.out.println("  Input: 2 neurons");
        System.out.println("  Hidden: 4 neurons (sigmoid activation)");
        System.out.println("  Output: 1 neuron (sigmoid activation)");
        System.out.println("\nTraining Parameters:");
        System.out.println("  Learning Rate: " + learningRate);
        System.out.println("  Epochs: " + epochs);
        System.out.println("\n" + "=".repeat(50) + "\n");


        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;

            // Train on each example
            for (int i = 0; i < inputs.length; i++) {
                // Convert to Matrix (2×1 column vector)
                Matrix input = new Matrix(2, 1);
                input.set(0, 0, inputs[i][0]);
                input.set(1, 0, inputs[i][1]);

                // Convert target to Matrix (1×1)
                Matrix target = new Matrix(1, 1);
                target.set(0, 0, targets[i][0]);

                // === FORWARD PASS ===

                // Hidden layer
                Matrix hiddenWeightedSum = hiddenLayer.forward(input);
                Matrix hiddenOutput = hiddenActivation.forward(hiddenWeightedSum);

                // Output layer
                Matrix outputWeightedSum = outputLayer.forward(hiddenOutput);
                Matrix prediction = outputActivation.forward(outputWeightedSum);

                // Compute loss (Mean Squared Error)
                MeanSquaredError mse = new MeanSquaredError();
                double loss = mse.compute(prediction, target);
                totalLoss += loss;

                // === BACKWARD PASS ===

                // Gradient of loss w.r.t. prediction (MSE derivative)
                // grad = (prediction - target)^2 -> d(grad)/d(prediction) = 2(predication - target)
                Matrix lossGrad = mse.gradient(prediction, target);

                // Backprop through output activation
                Matrix outputActivationGrad = outputActivation.backward(outputWeightedSum, lossGrad);

                // Backprop through output layer
                Matrix outputLayerGrad = outputLayer.backward(outputActivationGrad);

                // Backprop through hidden activation
                Matrix hiddenActivationGrad = hiddenActivation.backward(hiddenWeightedSum, outputLayerGrad);

                // Backprop through hidden layer
                hiddenLayer.backward(hiddenActivationGrad);

                // === UPDATE WEIGHTS ===
                adam.update(hiddenLayer);
                adam.update(outputLayer);
            }

            // Print progress
            if ((epoch + 1) % printEvery == 0 || epoch == 0) {
                double avgLoss = totalLoss / inputs.length;
                System.out.println("Epoch " + (epoch + 1) + "/" + epochs +
                        " - Average Loss: " + String.format("%.6f", avgLoss));
            }
        }

        System.out.println("\n" + "=".repeat(50));
        System.out.println("Training Complete!\n");

        // === TEST THE TRAINED NETWORK ===
        System.out.println("Testing XOR predictions:\n");
        System.out.println("Input      | Target | Prediction | Correct?");
        System.out.println("-".repeat(50));

        for (int i = 0; i < inputs.length; i++) {
            // Forward pass
            Matrix input = new Matrix(2, 1);
            input.set(0, 0, inputs[i][0]);
            input.set(1, 0, inputs[i][1]);

            Matrix hiddenWeightedSum = hiddenLayer.forward(input);
            Matrix hiddenOutput = hiddenActivation.forward(hiddenWeightedSum);
            Matrix outputWeightedSum = outputLayer.forward(hiddenOutput);
            Matrix prediction = outputActivation.forward(outputWeightedSum);

            double predicted = prediction.getData()[0][0];
            double target = targets[i][0];

            // Round to 0 or 1 for classification
            int predictedClass = predicted > 0.5 ? 1 : 0;
            int targetClass = (int) target;
            boolean correct = predictedClass == targetClass;

            System.out.printf("[%.0f, %.0f]   |   %d    |   %.4f    |   %s%n",
                    inputs[i][0], inputs[i][1],
                    targetClass,
                    predicted,
                    correct ? "✓" : "✗");
        }

        System.out.println("\n" + "=".repeat(50));
        System.out.println("\nInterpretation:");
        System.out.println("  - Predictions close to 0.0 or 1.0 are confident");
        System.out.println("  - If all predictions match targets, network learned XOR!");
        System.out.println("  - Typical final predictions: ~0.05 for 0, ~0.95 for 1");
    }
}
