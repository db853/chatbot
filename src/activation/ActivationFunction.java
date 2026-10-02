package activation;

import core.Matrix;

/**
 * Interface for neural network activation functions.
 * <p>
 * Activation functions introduce non-linearity into the network, allowing it to learn
 * complex patterns. Each implementation must provide a forward pass (to transform data)
 * and a backward pass (to compute gradients for learning).
 * </p>
 */
public interface ActivationFunction {
    /**
     * Forward pass: applies the activation function element-wise to the input.
     * <p>
     * This transforms the raw weighted sum from a layer (logits) into an activated output.
     * </p>
     *
     * @param input Matrix of values to activate.
     * @return Matrix with activation function applied.
     */
    Matrix forward(Matrix input);

    /**
     * Backward pass: computes the gradient of the activation function with respect to its input.
     * <p>
     * This is essential for backpropagation. It uses the chain rule:
     * {@code gradInput = gradOutput * derivative(input)}.
     * </p>
     *
     * @param input The original input to the activation function from the forward pass.
     * @param gradOutput The gradient flowing back from the subsequent layer.
     * @return Gradient to pass to the previous layer.
     */
    Matrix backward(Matrix input, Matrix gradOutput);
}
