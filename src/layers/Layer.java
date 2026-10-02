package layers;

import core.Matrix;

public interface Layer {
    /**
     * Forward pass: transforms input to output.
     *
     * This is where the layer performs its computation during prediction
     * or training. Each layer type implements its own transformation
     * (e.g., matrix multiplication for dense layers, lookup for embeddings).
     *
     * @param input Input matrix from previous layer or network input
     * @return Output matrix to pass to next layer
     */
    Matrix forward(Matrix input);

    /**
     * Backward pass: computes gradients for backpropagation.
     *
     * Given the gradient flowing back from the next layer, this method
     * computes gradients for this layer's parameters (weights/biases)
     * and returns the gradient to pass to the previous layer.
     *
     * @param gradOutput Gradient from the next layer (or loss function)
     * @return Gradient to pass to the previous layer
     */
    Matrix backward(Matrix gradOutput);

    /**
     * Updates the layer's parameters using computed gradients.
     *
     * After backward() computes gradients, this method uses those gradients
     * to update weights and biases. The learning rate controls the step size.
     *
     * @param learningRate How much to adjust weights (typically 0.001 - 0.1)
     */
    void updateWeights(double learningRate);

    Matrix getWeights();
    void setWeights(Matrix weights);
    Matrix getWeightGradients();

    default Matrix getBiases() {
        return null;
    }
    default void setBiases(Matrix biases) {}
    default Matrix getBiasGradients() {
        return null;
    }

    default boolean hasBiases() {
        return getBiases() != null;
    }
}
