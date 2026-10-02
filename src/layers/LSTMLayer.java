package layers;

import core.Matrix;
import layers.cells.LSTMBackwardResult;
import layers.cells.LSTMCell;
import layers.cells.LSTMForwardResult;
import layers.cells.LSTMHistory;
import lombok.Getter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Represents a high-level LSTM (Long Short-Term Memory) layer capable of processing sequences of data.
 * <p>
 * This layer manages the temporal iteration over input sequences, maintains the internal hidden
 * and cell states across time steps, and facilitates the accumulation of gradients during
 * Backpropagation Through Time (BPTT).
 * </p>
 * <h3>Key Responsibilities:</h3>
 * <ul>
 * <li><b>State Management:</b> Automatically initialises and tracks the hidden state (h_t)
 * and cell state (C_t) across the entire input sequence.</li>
 * <li><b>Temporal Processing:</b> Handles the "unrolling" of the sequence by passing
 * individual time steps to the {@link LSTMCell}.</li>
 * <li><b>Gradient Accumulation:</b> Stores and sums gradients for weights and biases
 * calculated at each time step during the backward pass to enable batch-level updates.</li>
 * <li><b>Optimization Interface:</b> Provides methods for external optimizers (like Adam or SGD)
 * to access combined weight and bias parameters and their respective gradients.</li>
 * </ul>
 * <h3>BPTT Implementation:</h3>
 * <p>
 * During the {@code backward} pass, the layer iterates through cached histories in reverse
 * chronological order. It computes the local gradients at each step and propagates the
 * error signal back through time to previous states and inputs.
 * </p>
 * @see LSTMCell
 * @see LSTMHistory
 * @see SequenceLayer
 */
public class LSTMLayer implements SequenceLayer {
    private final int inputSize;
    private final int hiddenSize;
    private final LSTMCell cell;

    private Matrix initialHiddenState;
    private Matrix initialCellState;

    @Getter
    private Matrix lastHiddenState;
    @Getter
    private Matrix lastCellState;

    @Getter
    private Matrix initialHiddenStateGrad;
    @Getter
    private Matrix initialCellStateGrad;

    @Getter
    private final List<LSTMHistory> histories;

    private Matrix forgetWeightGrad;
    private Matrix inputWeightGrad;
    private Matrix outputWeightGrad;
    private Matrix cellCandidateWeightGrad;

    private Matrix forgetBiasGrad;
    private Matrix inputBiasGrad;
    private Matrix outputBiasGrad;
    private Matrix cellCandidateBiasGrad;

    /**
     * Constructs a new LSTMLayer and initializes the underlying LSTM cell.
     *
     * @param inputSize  The size of the input vector at each time step.
     * @param hiddenSize The size of the hidden state and cell state.
     */
    public LSTMLayer(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.cell = new LSTMCell(inputSize, hiddenSize);
        this.histories = new ArrayList<>();
    }

    /**
     * Assigns initial hidden and cell states before the forward method runs.
     * <p>
     *     This is used in the Decoder class to assign the states to the final states produced by the Encoder (context
     *     state).
     * </p>
     * @param hiddenState Final hidden state from previous section
     * @param cellState Final cell state from previous section
     */
    public void assignInitialStates(Matrix hiddenState, Matrix cellState) {
        this.initialHiddenState = hiddenState;
        this.initialCellState = cellState;
    }

    /**
     * Performs a forward pass over a sequence of input matrices.
     * <p>
     * Iteratively calls the cell's forward method for each time step, maintaining
     * and updating the hidden and cell states while caching history for backpropagation.
     * </p>
     *
     * @param inputs A list of matrices where each matrix represents an input at time step t.
     * @return A list of hidden state matrices, one for each input time step.
     */
    @Override
    public List<Matrix> forward(List<Matrix> inputs) {
        this.histories.clear();

        int currentBatchSize = inputs.getFirst().getCols();

        // Initialise the states if this is the first forward pass or the batch size has changed
        if (initialHiddenState == null || initialCellState.getCols() != currentBatchSize) {
            this.initialCellState = Matrix.fillValue(hiddenSize, currentBatchSize, 0);
            this.initialHiddenState = Matrix.fillValue(hiddenSize, currentBatchSize, 0);
        }

        List<Matrix> hiddenStates = new ArrayList<>();
        Matrix previousHiddenState = this.initialHiddenState;
        Matrix previousCellState = this.initialCellState;

        // Each individual input Matrix represents a time step
        for (Matrix currentInput : inputs) {
            LSTMForwardResult result = cell.forward(currentInput, previousHiddenState, previousCellState);

            this.histories.add(result.history());

            // Adjust the previous hidden and cell states to be passed to the next time step
            previousHiddenState = result.hiddenState();
            previousCellState = result.cellState();

            hiddenStates.add(result.hiddenState());
        }

        // Allows this data from the Encoder to be passed to the decoder
        this.lastHiddenState = previousHiddenState;
        this.lastCellState = previousCellState;

        // Feed the last hidden and cell states back into initial to decode one word at a time
        this.assignInitialStates(this.lastHiddenState, this.lastCellState);

        return hiddenStates;
    }

    /**
     * Performs Backpropagation Through Time (BPTT) over the sequence.
     * <p>
     * Iterates backward through the cached histories to accumulate gradients for all
     * cell parameters and computes gradients for the input sequence.
     * </p>
     *
     * @param gradOutputs The gradients of the loss with respect to the hidden states produced in forward.
     * @return A list of gradients with respect to the input matrices, in original temporal order.
     */
    @Override
    public List<Matrix> backward(List<Matrix> gradOutputs) {
        int currentBatchSize = gradOutputs.getFirst().getCols();

        Matrix nextHiddenStateGrad = Matrix.fillValue(hiddenSize, currentBatchSize, 0);
        Matrix nextCellStateGrad = Matrix.fillValue(hiddenSize, currentBatchSize, 0);

        Matrix totalForgetWeightGrad = Matrix.fillValue(hiddenSize, inputSize + hiddenSize, 0);
        Matrix totalInputWeightGrad = Matrix.fillValue(hiddenSize, inputSize + hiddenSize, 0);
        Matrix totalOutputWeightGrad = Matrix.fillValue(hiddenSize, inputSize + hiddenSize, 0);
        Matrix totalCellCandidateWeightGrad = Matrix.fillValue(hiddenSize, inputSize + hiddenSize, 0);

        Matrix totalForgetBiasGrad = Matrix.fillValue(hiddenSize, 1, 0);
        Matrix totalInputBiasGrad = Matrix.fillValue(hiddenSize, 1, 0);
        Matrix totalOutputBiasGrad = Matrix.fillValue(hiddenSize, 1, 0);
        Matrix totalCellCandidateBiasGrad = Matrix.fillValue(hiddenSize, 1, 0);

        List<Matrix> inputGrads = new ArrayList<>();

        for (int t = gradOutputs.size() - 1; t >= 0; t--) {
            Matrix totalHiddenStateGrad = gradOutputs.get(t).add(nextHiddenStateGrad);

            LSTMHistory cache = this.histories.get(t);
            LSTMBackwardResult result = cell.backward(totalHiddenStateGrad, nextCellStateGrad, cache);

            nextHiddenStateGrad = result.previousHiddenStateGrad();
            nextCellStateGrad = result.previousCellStateGrad();

            // Accumulate the weight and bias gradients across all time steps
            totalForgetWeightGrad = totalForgetWeightGrad.add(result.forgetWeightGrad());
            totalInputWeightGrad = totalInputWeightGrad.add(result.inputWeightGrad());
            totalOutputWeightGrad = totalOutputWeightGrad.add(result.outputWeightGrad());
            totalCellCandidateWeightGrad = totalCellCandidateWeightGrad.add(result.cellCandidateWeightGrad());

            totalForgetBiasGrad = totalForgetBiasGrad.add(result.forgetBiasGrad());
            totalInputBiasGrad = totalInputBiasGrad.add(result.inputBiasGrad());
            totalOutputBiasGrad = totalOutputBiasGrad.add(result.outputBiasGrad());
            totalCellCandidateBiasGrad = totalCellCandidateBiasGrad.add(result.cellCandidateBiasGrad());

            inputGrads.add(result.currentInputGrad());
        }

        this.forgetWeightGrad = totalForgetWeightGrad;
        this.inputWeightGrad = totalInputWeightGrad;
        this.outputWeightGrad = totalOutputWeightGrad;
        this.cellCandidateWeightGrad = totalCellCandidateWeightGrad;
        this.forgetBiasGrad = totalForgetBiasGrad;
        this.inputBiasGrad = totalInputBiasGrad;
        this.outputBiasGrad = totalOutputBiasGrad;
        this.cellCandidateBiasGrad = totalCellCandidateBiasGrad;

        // Store these to be used in the Decoder BPTT
        this.initialHiddenStateGrad = nextHiddenStateGrad;
        this.initialCellStateGrad = nextCellStateGrad;

        // Reverse inputGrads as BPTT inserted input in x_t-1, x_t-2, etc order but other layers need the original order
        Collections.reverse(inputGrads);
        return inputGrads;
    }

    /**
     * Updates the underlying cell's weights and biases using the gradients accumulated during backward. This is used
     * for SDG optimisation.
     *
     * @param learningRate The step size to use for the gradient descent update.
     */
    @Override
    public void updateWeights(double learningRate) {
        cell.updateForgetWeights(this.forgetWeightGrad, learningRate);
        cell.updateInputWeights(this.inputWeightGrad, learningRate);
        cell.updateOutputWeights(this.outputWeightGrad, learningRate);
        cell.updateCellCandidateWeights(this.cellCandidateWeightGrad, learningRate);
        cell.updateForgetBiases(this.forgetBiasGrad, learningRate);
        cell.updateInputBiases(this.inputBiasGrad, learningRate);
        cell.updateOutputBiases(this.outputBiasGrad, learningRate);
        cell.updateCellCandidateBiases(this.cellCandidateBiasGrad, learningRate);
    }

    /**
     * Retrieves the concatenated weights of all LSTM gates from the cell.
     *
     * @return A single matrix containing all gate weight parameters.
     */
    @Override
    public Matrix getWeights() {
        return cell.getWeights();
    }

    /**
     * Sets the weights of the underlying LSTM cell.
     *
     * @param weights A matrix representing the new combined weights for the cell.
     */
    @Override
    public void setWeights(Matrix weights) {
        cell.setWeights(weights);
    }

    /**
     * Retrieves the concatenated gradients of the gate weights.
     *
     * @return A single matrix containing the accumulated weight gradients.
     */
    @Override
    public Matrix getWeightGradients() {
        return forgetWeightGrad.concatenate(inputWeightGrad).concatenate(outputWeightGrad).concatenate(cellCandidateWeightGrad);
    }

    /**
     * Retrieves the concatenated biases of all LSTM gates from the cell.
     *
     * @return A single matrix containing all gate bias parameters.
     */
    @Override
    public Matrix  getBiases() {
        return cell.getBiases();
    }

    /**
     * Sets the biases of the underlying LSTM cell.
     *
     * @param biases A matrix representing the new combined biases for the cell.
     */
    @Override
    public void setBiases(Matrix biases) {
        cell.setBiases(biases);
    }

    /**
     * Retrieves the concatenated gradients of the gate biases.
     *
     * @return A single matrix containing the accumulated bias gradients.
     */
    @Override
    public Matrix getBiasGradients() {
        return forgetBiasGrad.concatenate(inputBiasGrad).concatenate(outputBiasGrad).concatenate(cellCandidateBiasGrad);
    }

    /**
     * Resets the internal temporal memory of the LSTM layer.
     * <p>
     * This method clears the hidden state (h_t) and cell state (C_t) by setting them
     * to null. This ensures that the next call to {@link #forward(List)} will
     * initialise a fresh zero-state.
     * </p>
     * <p><b>Usage:</b> Call this at the start of every training epoch or when
     * switching to a completely new, unrelated data sequence to prevent
     * "state bleeding" from previous time steps.</p>
     */
    public void resetState() {
        this.initialHiddenState = null;
        this.initialCellState = null;

        this.lastHiddenState = null;
        this.lastCellState = null;
    }

    /**
     * Clears all accumulated weight and bias gradients.
     * <p>
     * In Backpropagation Through Time (BPTT), gradients are accumulated (summed)
     * across all time steps in a sequence. This method must be called before
     * starting a new backward pass to ensure the optimiser is working with
     * fresh gradients from the current sequence only.
     * </p>
     */
    public void resetGradients() {
        this.forgetWeightGrad = null;
        this.inputWeightGrad = null;
        this.outputWeightGrad = null;
        this.cellCandidateWeightGrad = null;
        this.forgetBiasGrad = null;
        this.inputBiasGrad = null;
        this.outputBiasGrad = null;
        this.cellCandidateBiasGrad = null;

        this.initialHiddenStateGrad = null;
        this.initialCellStateGrad = null;
    }

    /**
     * Scales all accumulated gradients by a specific numerical factor.
     * <p>
     * This is commonly used for two purposes:
     * <ul>
     * <li><b>Gradient Averaging:</b> Dividing the total gradient by the sequence
     * length (1/N) so that the update step size is independent of the
     * sequence duration.</li>
     * <li><b>Learning Rate Scaling:</b> Applying manual multipliers to specific
     * layers if required by the optimisation strategy.</li>
     * </ul>
     * </p>
     *
     * @param factor The multiplier to apply to every weight and bias gradient matrix.
     */
    public void scaleGradients(double factor) {
        this.forgetWeightGrad = this.forgetWeightGrad.multiply(factor);
        this.inputWeightGrad = this.inputWeightGrad.multiply(factor);
        this.outputWeightGrad = this.outputWeightGrad.multiply(factor);
        this.cellCandidateWeightGrad = this.cellCandidateWeightGrad.multiply(factor);

        this.forgetBiasGrad = this.forgetBiasGrad.multiply(factor);
        this.inputBiasGrad = this.inputBiasGrad.multiply(factor);
        this.outputBiasGrad = this.outputBiasGrad.multiply(factor);
        this.cellCandidateBiasGrad = this.cellCandidateBiasGrad.multiply(factor);
    }
}
