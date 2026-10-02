package activation;

import core.Matrix;
import core.MathUtils;

/**
 * Sigmoid activation function.
 * <p>
 * The sigmoid function maps any real-valued number into a range between 0 and 1.
 * It is commonly used in LSTMs for gate control (forget, input, and output gates)
 * because it acts as a "squashing" function that can represent probabilities or "levels of flow."
 * </p>
 * Mathematical formula: {@code f(x) = 1 / (1 + exp(-x))}
 */
public class Sigmoid implements ActivationFunction {
    /**
     * Applies the Sigmoid function element-wise to the input matrix.
     * * @param input Matrix of raw values.
     * @return Matrix with values squashed between 0 and 1.
     */
    @Override
    public Matrix forward(Matrix input) {
        Matrix output = new Matrix(input.getRows(), input.getCols());

        // Apply the sigmoid function to all values in the matrix
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                double value = input.getData()[i][j];
                output.set(i, j, MathUtils.sigmoid(value));
            }
        }
        return output;
    }

    /**
     * Computes the gradient of the Sigmoid function.
     * <p>
     * The derivative is calculated as {@code sigmoid(x) * (1 - sigmoid(x))}.
     * </p>
     * @param input The original input matrix from the forward pass.
     * @param gradOutput The gradient flowing back from the subsequent layer.
     * @return The gradient to be passed back to the previous layer.
     */
    @Override
    public Matrix backward(Matrix input, Matrix gradOutput) {
        Matrix gradInput = new Matrix(input.getRows(), input.getCols());

        // Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        // Chain rule: gradInput = gradOutput * derivative
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                double value = input.getData()[i][j];
                double derivative = MathUtils.sigmoidDerivative(value);
                gradInput.set(i, j, gradOutput.getData()[i][j] * derivative);
            }
        }
        return gradInput;
    }
}
