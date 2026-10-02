package activation;

import core.MathUtils;
import core.Matrix;

/**
 * Softmax activation function.
 * <p>
 * Softmax is typically used in the final layer of a classification network. It turns a
 * vector of K real values (logits) into a vector of K real values that sum to 1.0,
 * representing a probability distribution over classes.
 * </p>
 * Mathematical formula for element {@code i}: {@code exp(x_i) / sum(exp(x_j))}
 */
public class Softmax implements  ActivationFunction {
    /**
     * Applies the Softmax function to each column of the input matrix.
     * Each column is treated as a separate set of scores for a single sample.
     * @param input Matrix of logits.
     * @return Matrix where each column sums to 1.0.
     */
    @Override
    public Matrix forward(Matrix input) {
        Matrix output = new Matrix(input.getRows(), input.getCols());

        // Apply the softmax function to each column in the input
        for (int i = 0; i < input.getCols(); i++) {
            double[] column = input.getColumn(i);
            double[] softmaxColumn = MathUtils.softmax(column);
            output.setColumn(i, softmaxColumn);
        }

        return output;
    }

    /**
     * Computes the gradient of the Softmax function.
     * <p>
     * Unlike element-wise activations, Softmax has a Jacobian derivative because each
     * output depends on every input in the vector. This implementation multiplies the
     * incoming gradient by the Jacobian of the softmax.
     * </p>
     * @param input The original input matrix from the forward pass.
     * @param gradOutput The gradient (usually from a Loss function like Cross-Entropy).
     * @return The gradient to be passed back to the previous layer.
     */
    @Override
    public Matrix backward(Matrix input, Matrix gradOutput) {
        Matrix gradInput = new Matrix(input.getRows(), input.getCols());

        for (int i = 0; i < input.getCols(); i++) {
            // Extract the softmax outputs and generate the jacobian for that column
            double[] softmaxOutput = MathUtils.softmax(input.getColumn(i));
            double[][] jacobian = MathUtils.softmaxDerivative(softmaxOutput);
            Matrix jacobianInput = new Matrix(jacobian);

            // Extract corresponding loss vector from CCE and multiply them together
            double[] lossGrad = gradOutput.getColumn(i);

            // Chain rule: gradInput_i = gradOut_i * jacobianDerivative_i
            double[] lossJacobianMul = jacobianInput.matrixVectorMultiplication(lossGrad);
            gradInput.setColumn(i, lossJacobianMul);

        }
        return gradInput;
    }
}
