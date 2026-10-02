package loss;

import core.Matrix;

public interface LossFunction {
    /**
     * Computes the scalar loss value (for logging/tracking).
     */
    double compute(Matrix prediction, Matrix target);

    /**
     * Computes the gradient of the loss with respect to the prediction.
     * This is the starting point for backpropagation.
     */
    Matrix gradient(Matrix prediction, Matrix target);
}
