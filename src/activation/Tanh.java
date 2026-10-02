package activation;

import core.MathUtils;
import core.Matrix;

/**
 * Hyperbolic Tangent (Tanh) activation function.
 * <p>
 * Tanh maps input values to a range between -1 and 1. It is zero-centered, which often
 * helps the model converge faster than the sigmoid function. In LSTMs, it is used to
 * transform the cell candidate and the final cell state.
 * </p>
 * Mathematical formula: {@code f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))}
 */
public class Tanh implements ActivationFunction {
    /**
     * Applies the Tanh function element-wise to the input matrix.
     * @param input Matrix of raw values.
     * @return Matrix with values squashed between -1 and 1.
     */
    @Override
    public Matrix forward(Matrix input) {
        Matrix output = new Matrix(input.getRows(), input.getCols());

        // Apply the tanh function to the matrix element-wise
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                double value = input.getData()[i][j];
                output.set(i, j, MathUtils.tanh(value));
            }
        }
        return output;
    }

    /**
     * Computes the gradient of the Tanh function.
     * <p>
     * The derivative is calculated as {@code 1 - tanh(x)^2}.
     * </p>
     * @param input The original input matrix from the forward pass.
     * @param gradOutput The gradient flowing back from the subsequent layer.
     * @return The gradient to be passed back to the previous layer.
     */
    @Override
    public Matrix backward(Matrix input, Matrix gradOutput) {
        Matrix output = new Matrix(input.getRows(), input.getCols());

        // Derivative of the tanh function: 1 - (tanh)^2
        // Chain rule: gradInput = gradOutput * derivative
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                double value = input.getData()[i][j];
                double derivative = MathUtils.tanhDerivative(value);
                output.set(i, j, gradOutput.getData()[i][j] * derivative);
            }
        }
        return output;
    }
}
