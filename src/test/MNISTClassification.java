package test;

import activation.ReLU;
import activation.Softmax;
import core.MathUtils;
import core.Matrix;
import layers.DenseLayer;
import layers.Layer;
import loss.CategoricalCrossEntropyLoss;
import optimiser.Adam;

import java.io.IOException;

public class MNISTClassification {
    private static String imageTrainFilepath = "train-images.idx3-ubyte";
    private static String labelTrainFilepath = "train-labels.idx1-ubyte";

    private static String imageTestFilepath = "t10k-images.idx3-ubyte";
    private static String labelTestFilepath = "t10k-labels.idx1-ubyte";

    // Create the network
    // Architecture: 784 input -> 128 hidden -> 64 hidden -> 10 output
    private static final DenseLayer hiddenLayer1 = new DenseLayer(784, 128);
    private static final ReLU hiddenActivation1 = new ReLU();

    private static final DenseLayer hiddenLayer2 = new DenseLayer(128, 64);
    private static final ReLU hiddenActivation2 = new ReLU();

    private static final DenseLayer outputLayer = new DenseLayer(64, 10);
    private static final Softmax outputActivation = new Softmax();

    private static final double learningRate = 0.001;
    private static final Adam adam = new Adam(learningRate);

    public static void main(String[] args) throws IOException {
        System.out.println("=== MNIST Neural Network Test ===\n");

        // MNIST training data
        int batchSize = 64;

        double[][] trainImages = MNISTLoader.loadImages(imageTrainFilepath);
        int[] trainLabels = MNISTLoader.loadLabels(labelTrainFilepath);
        double[][] oneHotEncodeLabels = oneHotEncode(trainLabels);

        // Training parameters
        int epochs = 5;

        System.out.println("Network Architecture:");
        System.out.println("  Input: 784 neurons");
        System.out.println("  Hidden: 128 neurons (ReLU activation)");
        System.out.println("  Hidden: 64 neurons (ReLU activation)");
        System.out.println("  Output: 10 neurons (Softmax activation)");
        System.out.println("\nTraining Parameters:");
        System.out.println("  Learning Rate: " + learningRate);
        System.out.println("  Epochs: " + epochs);
        System.out.println("\n" + "=".repeat(50) + "\n");

        train(trainImages, oneHotEncodeLabels, batchSize, epochs);

        System.out.println("\n" + "=".repeat(50));
        System.out.println("Training Complete!\n");

        // === TEST THE TRAINED NETWORK ===
        System.out.println("Testing MNIST predictions:\n");

        double[][] testImages = MNISTLoader.loadImages(imageTestFilepath);
        int[] testLabels = MNISTLoader.loadLabels(labelTestFilepath);

        test(testImages, testLabels, batchSize);

    }

    private static double[][] oneHotEncode(int[] labels) {
        double[][] encoded = new double[labels.length][10];

        for (int i = 0; i < labels.length; i++) {
            int digit = labels[i];

            // Set the index corresponding to the digit to 1.0, others remain 0.0
            encoded[i][digit] = 1.0;
        }

        return encoded;
    }

    private static double[][] getBatches(double[][] input, int startingPosition, int batchSize) {
        int realBatchSize = Math.min(batchSize, input.length - startingPosition);
        double[][] batch = new double[input[0].length][realBatchSize];

        for (int j = 0;  j < batch[0].length; j++) {
            // Assigns batch[i][j] to input[startingPosition + i][j]
            for (int i = 0; i < batch.length; i++) {
                batch[i][j] = input[startingPosition + j][i];
            }
        }
        return batch;
    }

    private static void train(double[][] inputs, double[][] targets, int batchSize, int epochs) {

        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;

            // Train on a batch of images at a time
            for (int i = 0; i < inputs.length; i+= batchSize) {
                resetLayer(hiddenLayer1);
                resetLayer(hiddenLayer2);
                resetLayer(outputLayer);
                // Convert input to a matrix (784 * batchSize)
                double[][] imageBatch = getBatches(inputs, i, batchSize);
                Matrix input = new Matrix(imageBatch);

                // Convert target to Matrix (10 * batchSize)
                double[][] labelBatch = getBatches(targets, i, batchSize);
                Matrix target =  new Matrix(labelBatch);

                // === FORWARD PASS ===

                // 1st hidden layer
                Matrix hiddenWeightedSum1 = hiddenLayer1.forward(input);
                Matrix hiddenOutput1 = hiddenActivation1.forward(hiddenWeightedSum1);

                // 2nd hidden layer
                Matrix hiddenWeightedSum2 = hiddenLayer2.forward(hiddenOutput1);
                Matrix hiddenOutput2 = hiddenActivation2.forward(hiddenWeightedSum2);

                // Output layer
                Matrix outputWeightedSum = outputLayer.forward(hiddenOutput2);
                Matrix prediction = outputActivation.forward(outputWeightedSum);

                // Compute loss (CCE)
                CategoricalCrossEntropyLoss cce = new CategoricalCrossEntropyLoss();
                double loss = cce.compute(prediction, target);
                totalLoss += loss;

                // === BACKWARD PASS ===

                // Gradient of loss w.r.t. prediction (CCE derivative)
                Matrix lossGrad = cce.gradient(prediction, target);
                lossGrad = lossGrad.multiply(1.0 / imageBatch[0].length);

                // Backprop through output activation
                Matrix outputActivationGrad = outputActivation.backward(outputWeightedSum, lossGrad);

                // Backprop through output layer
                Matrix outputLayerGrad = outputLayer.backward(outputActivationGrad);

                // Backprop through 2nd hidden activation
                Matrix hiddenActivationGrad2 = hiddenActivation2.backward(hiddenWeightedSum2, outputLayerGrad);

                // Backprop through 2nd hidden layer
                Matrix hiddenLayer2Grad = hiddenLayer2.backward(hiddenActivationGrad2);

                // Backprop through 1st hidden activation
                Matrix hiddenActivationGrad1 = hiddenActivation1.backward(hiddenWeightedSum1, hiddenLayer2Grad);

                // Backprop through 1st hidden layer
                hiddenLayer1.backward(hiddenActivationGrad1);

                // === UPDATE WEIGHTS ===
                adam.update(hiddenLayer1);
                adam.update(hiddenLayer2);
                adam.update(outputLayer);
            }

            // Print progress
            double avgLoss = totalLoss / inputs.length;
            System.out.println("Epoch " + (epoch + 1) + "/" + epochs +
                    " - Average Loss: " + String.format("%.6f", avgLoss));

        }
    }

    private static void test(double[][] inputs, int[] targets, int batchSize) {
       int correctness = 0;

        for (int i = 0; i < inputs.length; i+= batchSize) {
            double[][] imageBatch = getBatches(inputs, i, batchSize);
            Matrix input = new Matrix(imageBatch);

            // Forward pass
            Matrix hiddenWeightedSum1 = hiddenLayer1.forward(input);
            Matrix hiddenOutput1 = hiddenActivation1.forward(hiddenWeightedSum1);

            // 2nd hidden layer
            Matrix hiddenWeightedSum2 = hiddenLayer2.forward(hiddenOutput1);
            Matrix hiddenOutput2 = hiddenActivation2.forward(hiddenWeightedSum2);

            // Output layer
            Matrix outputWeightedSum = outputLayer.forward(hiddenOutput2);
            Matrix prediction = outputActivation.forward(outputWeightedSum); // (10 * batchSize)


            for (int j = 0; j < prediction.getCols(); j++) {
                double[] predCol = prediction.getColumn(j);
                int maxIndex = MathUtils.argMax(predCol);

                System.out.println("Number: " + targets[i + j] + "    Prediction: " + maxIndex);

                if (maxIndex == targets[i + j]) {
                    correctness++;
                }
            }


        }
        double accuracy = ((double) correctness / inputs.length) * 100;
        System.out.println(accuracy);
    }

    public static void resetLayer(DenseLayer layer) {
        layer.resetGradients();
        layer.resetState();
    }

}
