package optimiser;


import core.Matrix;
import layers.Layer;

import java.util.HashMap;
import java.util.Map;

/**
 * Stochastic Gradient Descent with Momentum optimizer.
 *
 * Purpose: An improvement over basic SGD that accelerates convergence and
 * reduces oscillations by accumulating a velocity vector in directions of
 * persistent gradient reduction.
 *
 * How it works:
 * Think of a ball rolling down a hill - it builds up momentum as it goes.
 * Even if it hits a small uphill bump, momentum carries it through. Similarly,
 * this optimizer accumulates gradients from previous steps to build velocity.
 *
 * Formula:
 * velocity = momentum × velocity_previous - learningRate × gradient
 * weight = weight + velocity
 *
 * The momentum parameter (typically 0.9) determines how much of the previous
 * velocity to retain. Higher values mean more smoothing and acceleration.
 */
public class SGDMomentum implements Optimiser {
    private final double momentum;
    private final double learningRate;

    Map<Layer, Matrix> weightVelocity;
    Map<Layer, Matrix> biasVelocity;

    public SGDMomentum(double momentum, double learningRate) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.weightVelocity = new HashMap<>();
        this.biasVelocity = new HashMap<>();
    }

    /**
     * Updates layer parameters using momentum-accelerated gradient descent.
     *
     * On first call for a layer, initialises velocity to zero. On subsequent
     * calls, accumulates gradient into velocity using the momentum coefficient,
     * then applies the velocity to update parameters.
     *
     * This creates a "rolling average" effect that smooths out gradient noise
     * and accelerates movement in consistent directions.
     *
     * @param layer The layer whose parameters to update
     */
    @Override
    public void update(Layer layer) {
        // Initialise velocities if first time
        if (!weightVelocity.containsKey(layer)) {
            Matrix weights = layer.getWeights();
            weightVelocity.put(layer, Matrix.fillValue(weights.getRows(), weights.getCols(), 0));

            if (layer.hasBiases()) {
                Matrix biases = layer.getBiases();
                biasVelocity.put(layer, Matrix.fillValue(biases.getRows(), biases.getCols(), 0));
            }
        }

        Matrix weightVel = weightVelocity.get(layer);
        Matrix weightGrad = layer.getWeightGradients();

        // Update velocity: v = momentum * v_old - learningRate * gradient
        weightVel = weightVel.multiply(momentum).subtract(weightGrad.multiply(learningRate));

        // Update weights: weight = weight + velocity
        Matrix weights = layer.getWeights();
        layer.setWeights(weights.add(weightVel));

        // Store updated velocity
        weightVelocity.put(layer, weightVel);

        if (layer.hasBiases()) {
            Matrix biasVel = biasVelocity.get(layer);
            Matrix biasGrad = layer.getBiasGradients();

            biasVel = biasVel.multiply(momentum).subtract(biasGrad.multiply(learningRate));

            Matrix bias = layer.getBiases();
            layer.setBiases(bias.add(biasVel));

            biasVelocity.put(layer, biasVel);

        }

    }
}
