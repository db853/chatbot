package loss;

import core.Matrix;

public class MeanSquaredError implements LossFunction {
    @Override
    public double compute(Matrix prediction, Matrix target) {
        // L = (pred - target)^2
        double error = prediction.getData()[0][0] -  target.getData()[0][0];
        return error * error;
    }

    @Override
    public Matrix gradient(Matrix prediction, Matrix target) {
        // d(L)/d(pred) = 2 * (pred - target)
        double error = prediction.getData()[0][0] -  target.getData()[0][0];
        Matrix gradient = new Matrix(prediction.getRows(), prediction.getCols());
        gradient.set(0, 0, 2 * error);
        return gradient;
    }
}
