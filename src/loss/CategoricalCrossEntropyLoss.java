package loss;

import core.Matrix;

/**
 * Categorical Cross-Entropy Loss (CCE).
 * Used for multi-class classification where outputs sum to 1.
 * Pairs best with a Softmax activation layer and an explicit Jacobian matrix.
 */
public class CategoricalCrossEntropyLoss implements LossFunction {
    private final double epsilon = 1e-15;
    private final int padIndex;
    private final boolean useMasking;

    public CategoricalCrossEntropyLoss() {
        this.padIndex = -1;
        this.useMasking = false;
    }

    public CategoricalCrossEntropyLoss(int padIndex, boolean useMasking) {
        this.padIndex = padIndex;
        this.useMasking = useMasking;
    }

    /**
     * Computes the loss: L = -sum(target_i * log(pred_i))
     * * Logic:
     * 1. Iterate through the columns of the prediction matrix (the classes).
     * 2. Accumulate the sum of (target[i] * Math.log(prediction[i] + epsilon)).
     * 3. Multiply the final total by -1 and divide by number of columns to get average loss.
     */
    @Override
    public double compute(Matrix prediction, Matrix target) {
        double totalSum = 0.0;
        int validTokens = 0;

        for (int j = 0; j < prediction.getCols(); j++) {
            double[] targetColumn = target.getColumn(j);

            if (useMasking && targetColumn[padIndex] == 1.0) {
                continue;
            }

            validTokens++;

            double[] predictionColumn = prediction.getColumn(j);
            for (int i = 0; i < targetColumn.length; i++) {
                totalSum += targetColumn[i] * Math.log(predictionColumn[i] + epsilon);
            }
        }

        if (Double.isNaN(- totalSum / validTokens)) {
            System.out.println("Not a number");
        }
        return - totalSum / validTokens;
    }

    /**
     * Computes the gradient vector: dL/dy_i = -target_i / (prediction_i + epsilon)
     * Logic for Column Vectors:
     * 1. Create a new Matrix/vector of the same size as the prediction (vocabSize x batchSize).
     * 2. Loop through each row 'i' (each class) for the current column.
     * 3. Calculate: -target.get(i, 0) / (prediction.get(i, 0) + epsilon).
     * 4. This resulting vocabSize x batchSize matrix is your "Loss Gradient" (dL/dy).
     * * NOTE: To complete backpropagation with column vectors, you will multiply
     * the Softmax Jacobian Matrix by this vector: [vocabSize x vocabSize] * [vocabSize x batchSize] = [vocabSize x batchSize]
     */
    @Override
    public Matrix gradient(Matrix prediction, Matrix target) {
        // Create a matrix same size as prediction
        Matrix gradient = new Matrix(prediction.getRows(), prediction.getCols());
        int validTokens = 0;

        if (useMasking) {
            for (int j = 0; j < prediction.getCols(); j++) {
                double[] targetColumn = target.getColumn(j);
                if (targetColumn[padIndex] != 1.0) {
                    validTokens++;
                }
            }
        } else {
            validTokens = prediction.getCols();
        }

        for (int j = 0; j < prediction.getCols(); j++) {
            double[] targetColumn = target.getColumn(j);

            if (useMasking && targetColumn[padIndex] == 1.0) {
                continue;
            }

            double[] predictionColumn = prediction.getColumn(j);
            for (int i = 0; i < targetColumn.length; i++) {
                double value = (- targetColumn[i] / (predictionColumn[i] + epsilon)) / validTokens;
                if (Double.isNaN(value)) {
                    System.out.println("Not a number");
                }
                gradient.set(i, j, value);
            }
        }
        return gradient;
    }
}
