package activation;

import core.MathUtils;
import core.Matrix;

/**
 * Rectified Linear Unit (ReLU) activation function.
 * <p>
 * ReLU is a non-linear activation function that outputs the input directly if it is positive,
 * otherwise, it outputs zero. It is widely used in deep learning to mitigate the vanishing
 * gradient problem.
 * </p>
 * Mathematical formula: {@code f(x) = max(0, x)}
 */
public class ReLU implements ActivationFunction {
    /**
     * Applies the ReLU function element-wise to the input matrix.
     * @param input Matrix of raw values (logits).
     * @return Matrix where all negative values are replaced by 0.
     */
    @Override
    public Matrix forward(Matrix input) {
        Matrix output = new Matrix(input.getRows(), input.getCols());

        // Apply the relu function to the matrix element-wise
        for (int i = 0; i < input.getCols(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                double value = input.getData()[i][j];
                output.set(i, j, MathUtils.relu(value));
            }
        }
        return output;
    }

    /**
     * Computes the gradient of the ReLU function for backpropagation.
     * <p>
     * The gradient is 1 for positive inputs and 0 for negative inputs.
     * </p>
     * @param input The original input matrix from the forward pass.
     * @param gradOutput The gradient flowing back from the subsequent layer.
     * @return The gradient to be passed back to the previous layer.
     */
    @Override
    public Matrix backward(Matrix input, Matrix gradOutput) {
        Matrix output = new Matrix(input.getRows(), input.getCols());

        // Derivative of the relu function: 1 if x > 0, else 0
        // Chain rule: gradInput = gradOutput * derivative
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                double value = input.getData()[i][j];
                double derivative = MathUtils.reluDerivative(value);
                output.set(i, j, gradOutput.getData()[i][j] * derivative);
            }
        }
        return output;
    }
}
