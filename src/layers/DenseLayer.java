package layers;

import core.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a Fully Connected (Dense) Layer in a neural network.
 * <p>
 * This layer performs a linear transformation on the input data using a weight
 * matrix and a bias vector. It is typically used as the output layer of the
 * network to project the LSTM's hidden state into the final output dimension
 * (e.g., character class probabilities).
 * </p>
 * <p>
 * The mathematical operation is: {@code y = W^T x + b}, where {@code W} is the weight matrix,
 * {@code x} is the input, and {@code b} is the bias.
 * </p>
 */
public class DenseLayer implements Layer {
    private Matrix weights;
    private Matrix biases;
    // Stored to be used during backpropagation. A list is used to the previous input getting overwritten through
    // multiple forward calls in LSTM
    private final List<Matrix> input = new ArrayList<>();
    private int timeStep = 0;

    private Matrix weightGradients;
    private Matrix biasGradients;

    public DenseLayer(int inputSize, int outputSize) {
        this.weights = Matrix.randomXavier(inputSize, outputSize);
        this.biases = Matrix.fillValue(outputSize, 1, 0);
    }

    /**
     * Forward pass: computes output = weights × input + bias
     * N = batch size
     *
     * @param input Matrix of shape (inputSize × N)
     * @return Output column vector of shape (outputSize × N)
     */
    @Override
    public Matrix forward(Matrix input) {
        this.input.add(input);

        // (outputSize * inputSize) * (inputSize * N) = (outputSize * N) -> (outputSize * N) + (outputSize * N)
        return weights.transpose().matrixMultiplication(input).addWithBroadcast(biases);
    }

    /**
     * Backward pass: computes gradients.
     * <p>
     * Given gradient from next layer, computes:
     * <ol>
     * <li>Gradient w.r.t input (to pass to previous layer)</li>
     * <li>Gradient w.r.t weights (for updating weights)</li>
     * <li>Gradient w.r.t bias (for updating bias)</li>
     * </ol>
     * Formula for the output of a layer: y = Wx1 + Wx2 + Wx3 + ....
     * </p>
     * @param gradOutput Gradient from next layer, shape (outputSize × batchSize)
     * @return Gradient w.r.t input, shape (inputSize × batchSize)
     */
    @Override
    public Matrix backward(Matrix gradOutput) {
        // Gradient with respect to the input (d(L)/d(x)): W * gradOutput (d(y)/d(x) * d(L)/d(y))
        // (inputSize * outputSize) * (outputSize * 1 ) = (inputSize * 1)
        Matrix gradInput = weights.matrixMultiplication(gradOutput);

        // Gradient with respect to the weight (d(L)/d(W)): input * gradOutput^T (d(y)/d(W) * d(L)/d(y))
        // (inputSize * 1) * (1 * outputSize) = (inputSize * outputSize)
        Matrix currentInput = input.get(timeStep);
        timeStep++;
        Matrix newWeightGrad = currentInput.matrixMultiplication(gradOutput.transpose());

        // For sequence training using the LSTM, the gradients must be accumulated for each time the backward method is
        // called. If it is null, then it is the next epoch so it can be assigned normally
        this.weightGradients = (this.weightGradients == null) ? newWeightGrad :
                this.weightGradients.add(newWeightGrad);

        // Gradient with respect to the bias (d(L)/d(b)): 1 * gradOutput (d(y)/d(b) * d(L)/d(y)
        Matrix newBiasGrad = gradOutput.sumColumns();

        // For sequence training using the LSTM, the gradients must be accumulated for each time the backward method is
        // called. If it is null, then it is the next epoch so it can be assigned normally
        this.biasGradients = (this.biasGradients == null) ? newBiasGrad : this.biasGradients.add(newBiasGrad);

        // Return the gradient for the previous layer
        return gradInput;
    }

    @Override
    public void updateWeights(double learningRate) {
        // Slightly adjusts the weights and biases to improve the accuracy of the model
        weights = weights.subtract(weightGradients.multiply(learningRate));
        biases = biases.subtract(biasGradients.multiply(learningRate));
    }

    @Override
    public Matrix getWeights() {
        return weights;
    }

    @Override
    public void setWeights(Matrix weights) {
        this.weights = weights;
    }

    @Override
    public Matrix getWeightGradients() {
        return weightGradients;
    }

    @Override
    public Matrix  getBiases() {
        return biases;
    }

    @Override
    public void setBiases(Matrix biases) {
        this.biases = biases;
    }

    @Override
    public Matrix getBiasGradients() {
        return biasGradients;
    }

    /**
     * Clears the accumulated gradients for both weights and biases.
     * <p>
     * This should be called at the start of a new training iteration to ensure
     * that gradients from previous batches or sequences do not affect the current
     * update.
     * </p>
     */
    public void resetGradients() {
        this.weightGradients = null;
        this.biasGradients = null;
    }

    /**
     * Resets the temporal state of the layer.
     * <p>
     * Since this layer is often used with sequences, it caches inputs to facilitate
     * Backpropagation Through Time (BPTT). This method clears that cache and
     * resets the time step counter.
     * </p>
     */
    public void resetState() {
        this.input.clear();
        this.timeStep = 0;
    }

    /**
     * Scales all accumulated gradients by a scalar factor.
     * <p>
     * Typically used for gradient averaging ($1/N$) to prevent large sequence
     * lengths from causing massive weight updates.
     * </p>
     *
     * @param factor The multiplier to apply to the gradients.
     */
    public void scaleGradients(double factor) {
        this.weightGradients = this.weightGradients.multiply(factor);
        this.biasGradients = this.biasGradients.multiply(factor);
    }
}
