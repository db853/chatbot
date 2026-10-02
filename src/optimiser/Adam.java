package optimiser;


import core.Matrix;
import layers.LSTMLayer;
import layers.Layer;

import java.util.HashMap;
import java.util.Map;

/**
 * Adam (Adaptive Moment Estimation) optimizer.
 *
 * Purpose: The most popular optimizer in modern deep learning. Combines the
 * benefits of momentum and adaptive learning rates (RMSprop) to provide fast,
 * stable convergence with minimal hyperparameter tuning.
 *
 * How it works:
 * Adam maintains two moving averages for each parameter:
 * 1. First moment (m): Exponential moving average of gradients (like momentum)
 * 2. Second moment (v): Exponential moving average of squared gradients (for adaptive learning rates)
 *
 * It then uses these to compute adaptive learning rates for each parameter.
 * Parameters with large gradients get smaller effective learning rates, and
 * parameters with small gradients get larger effective learning rates.
 *
 * Formula (simplified):
 * m = beta1 × m + (1 - beta1) × gradient         // Update first moment
 * v = beta2 × v + (1 - beta2) × gradient²        // Update second moment
 * m_corrected = m / (1 - beta1^t)                 // Bias correction
 * v_corrected = v / (1 - beta2^t)                 // Bias correction
 * weight = weight - learningRate × m_corrected / (√v_corrected + epsilon)
 *

 * Typical hyperparameters (usually work well without tuning):
 * - Learning rate: 0.001 (default) or 0.0001 for fine-tuning
 * - beta1: 0.9 (decay rate for first moment)
 * - beta2: 0.999 (decay rate for second moment)
 * - epsilon: 1e-8 (small constant for numerical stability)
 */
public class Adam implements Optimiser {
    private final double beta1; // Decay rate for first moment
    private final double beta2; // Decay rate for second moment
    private final double epsilon; // Small constant for numerical stability
    private final double learningRate;

    // First moments (moving average of gradients) - like momentum
    private final Map<Layer, Matrix> weightM;
    private final Map<Layer, Matrix> biasM;

    // Second moments (moving average of squared gradients) - for adaptive learning rates
    private final Map<Layer, Matrix> weightV;
    private final Map<Layer, Matrix> biasV;

    // Time step counter per layer (for bias correction)
    // Adam's moments are biased toward zero initially, especially for small beta values
    // This is corrected by dividing by (1 - beta^t)
    private final Map<Layer, Integer> timeSteps;

    public Adam(double learningRate) {
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.epsilon = 1e-8;
        this.learningRate = learningRate;
        this.weightM = new HashMap<>();
        this.biasM = new HashMap<>();
        this.weightV = new HashMap<>();
        this.biasV = new HashMap<>();
        this.timeSteps = new HashMap<>();
    }


    /**
     * Updates layer parameters using Adam's adaptive moment estimation.
     *
     * Process for each parameter:
     * 1. Increment time step counter
     * 2. Update first moment (exponential moving average of gradients)
     * 3. Update second moment (exponential moving average of squared gradients)
     * 4. Compute bias-corrected moments (important for early training steps)
     * 5. Compute adaptive learning rate for each parameter
     * 6. Update parameter
     *
     * The key insight of Adam is that each parameter gets its own effective
     * learning rate based on the history of its gradients. Parameters with
     * consistently large gradients get smaller updates (preventing oscillation),
     * while parameters with small gradients get larger updates (accelerating learning).
     *
     * The bias correction ensures the optimiser works well from the very first
     * update, even though the moment estimates start at zero.
     *
     * Handles layers with and without bias appropriately - only updates
     * bias if the layer has one.
     *
     * @param layer The layer whose parameters to update
     */
    @Override
    public void update(Layer layer) {
        // Add initial values to the weight and bias maps if it is the first time
        if (!weightM.containsKey(layer)) {
            Matrix weights = layer.getWeights();

            weightM.put(layer, Matrix.fillValue(weights.getRows(), weights.getCols(), 0));
            weightV.put(layer, Matrix.fillValue(weights.getRows(), weights.getCols(), 0));

            if (layer.hasBiases()) {
                Matrix bias = layer.getBiases();
                biasM.put(layer, Matrix.fillValue(bias.getRows(), bias.getCols(), 0));
                biasV.put(layer, Matrix.fillValue(bias.getRows(), bias.getCols(), 0));
            }

            // Initialise time step
            timeSteps.put(layer, 0);
        }

        // Increment time step
        int t = timeSteps.get(layer) + 1;
        timeSteps.put(layer, t);

        // Compute bias correction factors (same for weights and bias)
        double beta1Correction = 1.0 - Math.pow(beta1, t);
        double beta2Correction = 1.0 - Math.pow(beta2, t);

        // Get current moments
        Matrix wM = weightM.get(layer);
        Matrix wV = weightV.get(layer);

        Matrix weightGrad = layer.getWeightGradients();
        weightGrad = weightGrad.clipByL2Norm(1.0);

        // Update first moment: m_t = m_{t-1} * beta1 + gradient * (1 - beta1)
        wM = wM.multiply(beta1).add(weightGrad.multiply(1 - beta1));

        // Update second moment: v_t = v_{t-1} * beta2 + gradient * (1 - beta2)
        Matrix weightGradSquared = weightGrad.hadamardProduct(weightGrad);
        wV = wV.multiply(beta2).add(weightGradSquared.multiply(1 - beta2));

        // Bias corrected moments - prevents moments from tending to 0
        Matrix wMCorrected = wM.multiply(1.0 / beta1Correction);
        Matrix wVCorrected = wV.multiply(1.0 / beta2Correction);

        // Commute adaptive update
        Matrix wUpdate = computeAdaptiveUpdate(wMCorrected, wVCorrected, learningRate);

        // Update weights
        Matrix weights = layer.getWeights();
        layer.setWeights(weights.subtract(wUpdate));

        // Store updated moments
        weightM.put(layer, wM);
        weightV.put(layer, wV);

        if (layer.hasBiases()) {
            Matrix bM = biasM.get(layer);
            Matrix bV = biasV.get(layer);

            Matrix biasGrad = layer.getBiasGradients().sumColumns();
            biasGrad = biasGrad.clipByL2Norm(1.0);

            // Update first moment: m_t = beta1 * m_{t-1} + (1-beta1) * gradient
            bM = bM.multiply(beta1).add(biasGrad.multiply(1 - beta1));

            // Update second moment: v_t = beta2 * v_{t-1}
            Matrix biasGradSquared = biasGrad.hadamardProduct(biasGrad);
            bV = bV.multiply(beta2).add(biasGradSquared.multiply(1 - beta2));

            // Bias corrected moments
            Matrix bMCorrected = bM.multiply(1.0 / beta1Correction);
            Matrix bVCorrected = bV.multiply(1.0 / beta2Correction);

            // Commute adaptive update
            Matrix bUpdate = computeAdaptiveUpdate(bMCorrected, bVCorrected, learningRate);

            // Update biases
            Matrix biases = layer.getBiases();
            layer.setBiases(biases.subtract(bUpdate));

            // Store updated moments
            biasM.put(layer, bM);
            biasV.put(layer, bV);
        }
    }

    /**
     * Computes the adaptive parameter update for Adam.
     *
     * Formula: update = learningRate × m / (√v + epsilon)
     *
     * Where:
     * - m is the bias-corrected first moment (average gradient direction)
     * - v is the bias-corrected second moment (average gradient magnitude²)
     * - epsilon prevents division by zero
     *
     * The √v term in the denominator creates the "adaptive" part - parameters
     * with historically large gradients get smaller effective learning rates,
     * and vice versa.
     *
     * @param m First moment (bias-corrected)
     * @param v Second moment (bias-corrected)
     * @param learningRate Base learning rate
     * @return Adaptive update to subtract from parameters
     */
    private Matrix computeAdaptiveUpdate(Matrix m, Matrix v, double learningRate) {
        Matrix update = new Matrix(m.getRows(), m.getCols());

        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                double mVal = m.getData()[i][j];
                double vVal = v.getData()[i][j];

                // Adaptive learning rate for this specific parameter
                double adaptiveStep = learningRate * mVal / (Math.sqrt(vVal) + epsilon);

                update.set(i, j, adaptiveStep);
            }
        }
        return update;
    }
}
