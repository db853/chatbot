package optimiser;

import layers.Layer;

/**
 * Stochastic Gradient Descent (SGD) optimizer.
 *
 * Purpose: The simplest optimization algorithm for training neural networks.
 * Updates weights by moving them in the opposite direction of the gradient,
 * scaled by the learning rate.
 *
 * How it works:
 * For each parameter (weight): weight = weight - learningRate × gradient
 *
 * The "stochastic" part means we update after each batch rather than after
 * seeing all training data. This introduces noise but often helps escape
 * local minima and trains faster.
 */
public class SGD implements Optimiser {
    private final double learningRate;

    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void update(Layer layer) {
        layer.updateWeights(learningRate);
    }
}
