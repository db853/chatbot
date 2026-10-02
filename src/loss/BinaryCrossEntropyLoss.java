package loss;

import core.Matrix;

/**
 * Binary Cross-Entropy Loss (BCE).
 * Used for binary classification tasks where the output is a single probability.
 * Pairs best with a Sigmoid activation layer.
 */
public class BinaryCrossEntropyLoss implements LossFunction {
    private final double epsilon = 1e-15;

    @Override
    public double compute(Matrix prediction, Matrix target) {
        // L = -(ylog(p) + (1 - y)log(1 - p))
        double pred = prediction.getData()[0][0];
        double actual =  target.getData()[0][0];

        // Small value is add in the log to prevent a log of zero
        return -(actual * Math.log(pred + epsilon) + (1 - actual) * Math.log(1 - pred + epsilon));
    }

    @Override
    public Matrix gradient(Matrix prediction, Matrix target) {
        double pred = prediction.getData()[0][0];
        double actual =  target.getData()[0][0];

        Matrix gradient = new Matrix(prediction.getRows(), prediction.getCols());

        // Derivative of cross-entropy with respect to prediction: (p - y) / (p * (1 - p))
        double value = (pred - actual) / (pred * (1 - pred + epsilon));
        gradient.set(0, 0, value);
        return gradient;
    }
}
