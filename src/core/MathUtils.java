package core;

public class MathUtils {
    /**
     * Sigmoid activation function: 1 / (1 + e^(-x))
     * Squashes any value to range (0, 1).
     *
     * @param x Input value
     * @return Output between 0 and 1
     */
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Derivative of sigmoid function.
     * Used in backpropagation: sigmoid(x) * (1 - sigmoid(x))
     *
     * @param x Input value (the activated output from forward pass)
     * @return Derivative value
     */
    public static double sigmoidDerivative(double x) {
        double sig = sigmoid(x);
        return sig * (1 - sig);
    }

    /**
     * Tanh activation function: (e^x − e^(−x))/(e^x + e^(−x))
     * Squashes any value to range (-1, 1).
     * Often used within the hidden layers
     *
     * @param x Input value
     * @return Output between -1 and 1
     */
    public static double tanh(double x) {
        return Math.tanh(x);
    }

    /**
     * Derivative of the tanh function
     * Used in backpropagation: 1 - (tanh)^2
     *
     * @param x Derivative value between 0 and 1
     * @return Derivative value between 0 and 1
     */
    public static double tanhDerivative(double x) {
        double t = Math.tanh(x);
        return 1.0 - (t * t);
    }

    /**
     * ReLU (Rectified Linear Unit) activation function.
     * Zeros out negative values and returns positive values unchanged
     * @param x Input value
     * @return x if x > 0, otherwise 0
     */
    public static double relu(double x) {
        return Math.max(0.0, x);
    }

    /**
     * Derivative of ReLU function.
     * Used in backpropagation: 1 if x > 0, else 0
     * @param x Input value that was passed through ReLU in the forward pass
     * @return 1.0 if x > 0, otherwise 0.0
     */
    public static double reluDerivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    /**
     * Softmax (soft argmax) activation function.
     * Transformations a vector into a normalised distribution where each value represents the probability of an
     * outcome.
     * Formula: softmax(x_i) = e^(x_i) / sum(e^(x_j)) for all j
     * @param x Column vector of un-normalised values
     * @return A normal distribution of the values
     */
     public static double[] softmax(double[] x) {
         double max = 0;

         for (double v : x) {
             if (v > max) {
                 max = v;
             }
         }

         // Create a new array by computing e^value
        double[] expX = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            expX[i] = Math.exp(x[i] - max);
        }

        // Sum the new array to ise in normalisation
        double expSum = 0.0;
        for (int i = 0; i < x.length; i++) {
            expSum += expX[i];
        }

        // Normalise all values
        double[] normalised =  new double[x.length];
        for (int i = 0; i < x.length; i++) {
            normalised[i] = expX[i] / expSum;
        }

        return normalised;
     }

    /**
     * Computes the Jacobian matrix of the Softmax function.
     * * The Jacobian represents the partial derivatives of each output y_i with respect
     * to each input x_j. Since Softmax is a vector-valued function where every
     * output depends on every input, the result is a square n x n matrix.
     * * Mathematical Logic:
     * - If i = j (Diagonal): The derivative is y_i(1 - y_i)
     * - If i != j (Off-diagonal): The derivative is -y_i * y_j
     * @param predictions The output vector (probabilities) from the Softmax forward pass.
     * @return A 2d array representing an n x n Jacobian matrix, where n is the length of the predictions array.
     */
     public static double[][] softmaxDerivative(double[] predictions) {
         double[][] jacobian = new double[predictions.length][predictions.length];

         for (int i = 0; i < predictions.length; i++) {
             for (int j = 0; j < predictions.length; j++) {
                double gradient;

                if (i == j) {
                    gradient = predictions[i] * (1 - predictions[i]);
                } else {
                    gradient = - predictions[i] * predictions[j];
                }
                jacobian[i][j] =  gradient;

             }
         }
         return jacobian;
     }

    /**
     * Calculates the index of the maximum value in a list, used to determine the highest probability in a softmax
     * array.
     * @param x The softmax output
     * @return The index of the largest element
     */
     public static int argMax(double[] x) {
        double max = x[0];
        int maxIndex = 0;

        for (int i = 1; i < x.length; i++) {
            if (x[i] > max) {
                max = x[i];
                maxIndex = i;
            }
        }
        return maxIndex;
     }
}
