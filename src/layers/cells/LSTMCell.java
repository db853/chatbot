package layers.cells;

import activation.Sigmoid;
import activation.Tanh;
import core.Matrix;
import lombok.Data;

/**
 * Represents a single Long Short-Term Memory (LSTM) cell.
 * <p>
 * This class implements the core mathematical operations of an LSTM unit, utilizing
 * three primary "gates" and a candidate state to manage the flow of information
 * across time steps.
 * </p>
 *
 * <h3>Gate Functions:</h3>
 * <ul>
 * <li><b>Forget Gate (ft):</b> Determines which information from the previous cell state
 * (long-term memory) is no longer relevant and should be discarded.
 * It uses a sigmoid function to output values between 0 (completely forget) and 1 (completely keep).</li>
 * <li><b>Input Gate (it):</b> Controls which new information from the current input and
 * previous hidden state is worthy of being added to the cell state.
 * It acts as a filter for the <b>Cell Candidate</b>.</li>
 * <li><b>Output Gate (ot):</b> Decides what part of the current cell state should be
 * surfaced as the hidden state (output) for the current time step.
 * This hidden state is then passed to the next time step and higher layers of the network.</li>
 * </ul>
 *
 * <h3>Cell Candidate (ĉt):</h3>
 * <p>
 * Unlike the gates, the candidate state uses a Tanh activation to create a vector of
 * "new information" that could potentially be added to the memory.
 * This candidate is scaled by the Input Gate before being merged into the cell state.
 * </p>
 */
@Data
public class LSTMCell {
    private Matrix forgetWeights;
    private Matrix forgetBiases;

    private Matrix inputWeights;
    private Matrix inputBiases;

    private Matrix outputWeights;
    private Matrix outputBiases;

    private Matrix cellCandidateWeights;
    private Matrix cellCandidateBiases;

    private final int inputSize;
    private final int hiddenSize;
    private final Sigmoid sigmoid = new Sigmoid();
    private final Tanh tanh = new Tanh();

    /**
     * Initialises a new LSTM Cell with Xavier-randomised weights and zeroed biases.
     * <p>
     * The weights for all three gates (Forget, Input and Output) and the cell candidate are
     * initialised with a width of {@code inputSize + hiddenSize} to allow for the
     * concatenation of the current input and the previous hidden state.
     * </p>
     * <p>
     * The information is filtered by carrying out element-wise multiplication (Hadamard product) between the gate
     * matrix and previous cell state.
     * </p>
     * @param inputSize  The dimension of the input vector (x_t), such as embedding size.
     * @param hiddenSize The number of units in the LSTM, determining the size of the
     * hidden (h_t) and cell (C_t) states.
     */
    public LSTMCell(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        this.forgetWeights = initGateWeights(inputSize, hiddenSize);
        this.inputWeights = initGateWeights(inputSize, hiddenSize);
        this.outputWeights = initGateWeights(inputSize, hiddenSize);
        this.cellCandidateWeights = initGateWeights(inputSize, hiddenSize);

        this.forgetBiases = Matrix.fillValue(hiddenSize, 1, 1);
        this.inputBiases = Matrix.fillValue(hiddenSize, 1, 0);
        this.outputBiases = Matrix.fillValue(hiddenSize, 1, 0);
        this.cellCandidateBiases = Matrix.fillValue(hiddenSize, 1, 0);
    }

    /**
     * Performs a single forward pass time-step for the LSTM cell.
     * <p>
     * This method processes the current input and previous states through the
     * forget, input, and output gates to update the long-term memory (cell state)
     * and produce the current short-term memory (hidden state).
     * </p>
     * @param currentInput The input matrix (x_t) for the current time step.
     * @param previousHiddenState The hidden state matrix (h_t-1) from the previous time step.
     * @param previousCellState The cell state matrix (C_t-1) from the previous time step.
     * @return An {@link LSTMForwardResult} containing the new hidden state (h_t) and
     * the updated cell state (C_t).
     */
    public LSTMForwardResult forward(Matrix currentInput, Matrix previousHiddenState, Matrix previousCellState) {
        Matrix combinedInput = previousHiddenState.concatenate(currentInput);

        /* FORGET GATE */

        // Calculate the forget gate matrix (ft)
        // ft = sigmoid(Wf * combinedInput + Bf)
        Matrix fLinear = this.forgetWeights.matrixMultiplication(combinedInput).addWithBroadcast(this.forgetBiases);
        Matrix ft = sigmoid.forward(fLinear);

        /* INPUT GATE */

        // Calculate the input gate matrix (it)
        // it = sigmoid(Wi * combinedInput + Bi)
        Matrix iLinear = this.inputWeights.matrixMultiplication(combinedInput).addWithBroadcast(this.inputBiases);
        Matrix it = sigmoid.forward(iLinear);

        // Calculate new candidate cell state matrix
        // cCandidate = tanh(Wc * combinedInput + Bc)
        Matrix cCandidateLinear = this.cellCandidateWeights.matrixMultiplication(combinedInput)
                .addWithBroadcast(this.cellCandidateBiases);
        Matrix cCandidate = tanh.forward(cCandidateLinear);

        // Calculate the new cell state matrix (ct)
        // ct = ft * previousCellState + it * cCandidate
        Matrix ct = ft.hadamardProduct(previousCellState).add(it.hadamardProduct(cCandidate));

        /* OUTPUT GATE */

        // Calculate the output gate matrix (ot)
        // ot = sigmoid(Wo * combinedInput + Bo)
        Matrix oLinear = this.outputWeights.matrixMultiplication(combinedInput).addWithBroadcast(this.outputBiases);
        Matrix ot = sigmoid.forward(oLinear);

        // Calculate the new hidden state matrix (ht)
        // ht = ot * tanh(ct)
        Matrix ht = ot.hadamardProduct(tanh.forward(ct));

        // Return both the cell state and hidden state to be used in the next time step. A LSTM history is also passed
        // so that it can be used in BPTT
        LSTMHistory history = new LSTMHistory(combinedInput, previousCellState, ft, it, ot, cCandidate, ct, fLinear,
                iLinear, oLinear, cCandidateLinear);
        return new LSTMForwardResult(ht, ct, history);
    }

    /**
     * Performs a single backward pass time-step for the LSTM cell (Backpropagation Through Time).
     * <p>
     * This method implements the chain rule to calculate gradients for all learnable parameters
     * (weights and biases) and the error signals to be propagated to the previous time step.
     * </p>
     * <h3>Mathematical Flow:</h3>
     * <ol>
     * <li><b>Cell State Gradient:</b> Combines the incoming error from the future (next time step)
     * with the error from the current hidden state output.</li>
     * <li><b>Gate Gradients:</b> Distributes the cell state error among the forget gate, input gate,
     * and cell candidate based on their contributions during the forward pass.</li>
     * <li><b>Pre-activation Gradients:</b> Reverses the Sigmoid and Tanh activations to find the
     * error at the linear level (z).</li>
     * <li><b>Parameter Gradients:</b> Calculates {@code dW} and {@code dB} for all four gates.
     * Biases are summed across the batch dimension to reverse the forward broadcast.</li>
     * <li><b>Input Propagation:</b> Calculates the gradient for the concatenated input, which is
     * then split into the error for the previous hidden state (h_t-1) and the current input (x_t).</li>
     * </ol>
     *
     * @param hiddenStateGrad  The gradient of the loss with respect to the current hidden state (h_t).
     * Includes error from the layer above and the next time step.
     * @param nextCellStateGrad The gradient of the loss with respect to the cell state of the
     * next time step (C_t+1).
     * @param cache An {@link LSTMHistory} object containing the saved matrices from
     * the forward pass of this specific time step.
     * @return An {@link LSTMBackwardResult} containing the gradients for weights, biases,
     * and the states (h and C) to be passed to the previous time step.
     */
    public LSTMBackwardResult backward(Matrix hiddenStateGrad, Matrix nextCellStateGrad, LSTMHistory cache) {
        /* CELL STATE GRAD */

        /* Calculate the gradient of loss w.r.t the cell state
            dL/dC_t = dL/dC_t (future) + dL/dC_t (current)

            dL/dC_t (current) = dL/dh_t * dh_t/dC_t
            As h_t = o_t * tanh(C_t) => dh_t/dC_t = o_t * tanh`(C_t)
            This means dL/dC_t (current) = dL/dh_t * o_t * tanh`(C_t)
         */
        Matrix currentCellStateGrad = tanh.backward(cache.ct(), hiddenStateGrad.hadamardProduct(cache.ot()));
        Matrix totalCellStateGrad = nextCellStateGrad.add(currentCellStateGrad);

        /* GATE AND CELL CANDIDATE GRADS */

        /* Calculate the gradient of loss w.r.t the forget gate
            dL/df_t = dL/dC_t * dC_t/df_t
            As C_t = f_t * C_t-1 + i_t * C_hat => dC_t/df_t = C_t-1
            This means dL/df_t = dL/dC_t * C_t-1
         */
        Matrix forgetGrad = totalCellStateGrad.hadamardProduct(cache.previousCellState());

        /* Calculate the gradient of loss w.r.t the input gate
            dL/di_t = dL/dC_t * dC_t/di_t
            As C_t = f_t * C_t-1 + i_t * C_hat => dC_t/di_t = C_hat
            This means dL/di_t = dL/dC_t * C_hat
         */
        Matrix inputGrad = totalCellStateGrad.hadamardProduct(cache.cCandidate());

        /* Calculate the gradient of loss w.r.t cell candidate
            dL/dC_hat = dL/dC_t * dC_t/dC_hat
            As C_t = f_t * C_t-1 + i_t * C_hat => dC_t/dC_hat = i_t
            This means dL/dC_hat = dL/dC_t * i_t
         */
        Matrix cellCandidateGrad = totalCellStateGrad.hadamardProduct(cache.it());

        /* Calculate the gradient of loss w.r.t the output gate
            dL/do_t = dL/dh_t * dh_t/do_t
            As h_t = o_t * tanh(C_t) => dh_t/d_ot = tanh(C_t).
            This means dL/do_t = dL/dh_t * tanh(C_t)
        */
        Matrix outputGrad = hiddenStateGrad.hadamardProduct(tanh.forward(cache.ct()));

        /* PRE-ACTIVATION GRADS */

        /* Calculate the gradient of loss w.r.t the linear forget gate
            dL/dz_f = dL/df_t * df_t/dz_f
            As f_t = sigmoid(z_f) => df_t/dz_f = sigmoid`(z_f)
            This means dL/dz_f = dL/df_t * sigmoid`(z_f)
         */
        Matrix forgetLinearGrad = sigmoid.backward(cache.fLinear(), forgetGrad);

        /* Calculate the gradient of loss w.r.t the linear input gate
            dL/dz_i = dL/di_t * di_t/dz_i
            As i_t = sigmoid(i_f) => di_t/dz_i = sigmoid`(z_i)
            This means dL/dz_i = dL/di_t * sigmoid`(z_i)
         */
        Matrix inputLinearGrad = sigmoid.backward(cache.iLinear(), inputGrad);

        /* Calculate the gradient of loss w.r.t linear cell candidate
            dL/dz_C_hat = dL/dC_hat * dC_hat/dz_C_hat
            As C_hat = tanh(z_C_hat) => dC_hat/dz_C_hat = tanh`(z_C_hat)
            This means dL/dz_C_hat = dL/dC_hat * sigmoid`(z_C_hat)
         */
        Matrix cellCandidateLinearGrad = tanh.backward(cache.cCandidateLinear(), cellCandidateGrad);

        /* Calculate the gradient of loss w.r.t the linear output gate
            dL/dz_o = dL/do_t * do_t/dz_o
            As o_t = sigmoid(z_o) => do_t/dz_o = sigmoid`(z_o)
            This means dL/dz_o = dL/do_t * sigmoid`(z_o)
         */
        Matrix outputLinearGrad = sigmoid.backward(cache.oLinear(), outputGrad);

        /* CALCULATING WEIGHTS AND BIASES */

        /* Calculate the gradient of loss w.r.t the gate and cell candidate weights
            dL/dW_x = dL/dz_x * dz_x/dW_x
            As z_x = W_x * combinedInput + B_x => dz_x/dW_x = combinedInput (transposed)
            This means dW_x = dL/dz_x * combinedInput (transposed)
         */
        Matrix forgetWeightGrad = forgetLinearGrad.matrixMultiplication(cache.combinedInput().transpose());
        Matrix inputWeightGrad = inputLinearGrad.matrixMultiplication(cache.combinedInput().transpose());
        Matrix cellCandidateWeightGrad = cellCandidateLinearGrad.matrixMultiplication(cache.combinedInput().transpose());
        Matrix outputWeightGrad = outputLinearGrad.matrixMultiplication(cache.combinedInput().transpose());

        /* Calculate the gradient of loss w.r.t the gate and cell candidate biases
            dL/dB_x = dL/dz_x * dz_x/dB_x
            As z_x = W_x * combinedInput + B_x => dz_x/dB_x = 1
            This means dB_x = dL/dz_x.
            As addWithBroadcast was used in forward, the values across columns will need summed to collapse back
            into a column vector.
         */
        Matrix forgetBiasGrad = forgetLinearGrad.sumColumns();
        Matrix inputBiasGrad = inputLinearGrad.sumColumns();
        Matrix cellCandidateBiasGrad = cellCandidateLinearGrad.sumColumns();
        Matrix outputBiasGrad = outputLinearGrad.sumColumns();

        /* CALCULATE PASS BACK GRADS */

        /* Calculate the gradient of loss w.r.t the previous cell state to pass back
            dL/dC_t-1 = dL/dC_t * dC_t/dC_t-1
            As C_t = f_t * C_t-1 + i_t * C_hat => dC_t/dC_t-1 = f_t
            This means dL/dC_t-1 = dL/dC_t * f_t
         */
        Matrix previousCellStateGrad = totalCellStateGrad.hadamardProduct(cache.ft());

        /* Calculate the gradient of loss w.r.t the combinedInput to pass back
            dL/dCombinedInput = dL/dz_x * dz_x/dCombinedInput
            As z_x = W_x * combinedInput + B_x => dz_x/dCombinedInput = W_x (transposed).
            This means dL/dCombinedInput = dL/dz_x * W_x (transposed).
            This will be summed for each gate and the cell candidate to get the total gradient for the combined input.
            This is because each of them contribute to the error.
         */
        Matrix combinedInputGrad = this.forgetWeights.transpose().matrixMultiplication(forgetLinearGrad).add(
                this.inputWeights.transpose().matrixMultiplication(inputLinearGrad).add(
                        this.cellCandidateWeights.transpose().matrixMultiplication(cellCandidateLinearGrad).add(
                                this.outputWeights.transpose().matrixMultiplication(outputLinearGrad)
                        )
                )
        );

        /* Split the combined input matrix to get both the gradient of loss w.r.t previous hidden state and gradient of
           loss w.r.t to current input.
         */
        Matrix previousHiddenStateGrad = combinedInputGrad.split(0, this.hiddenSize);
        Matrix currentInputGrad = combinedInputGrad.split(this.hiddenSize, this.hiddenSize + this.inputSize);

        // Return all relevant information back for the previous time step
        return new LSTMBackwardResult(previousHiddenStateGrad, previousCellStateGrad, currentInputGrad,
                forgetWeightGrad, inputWeightGrad, outputWeightGrad, cellCandidateWeightGrad, forgetBiasGrad,
                inputBiasGrad, outputBiasGrad, cellCandidateBiasGrad
        );
    }

    public void updateForgetWeights(Matrix forgetWeightGrad, double learningRate) {
        this.forgetWeights = this.forgetWeights.subtract(forgetWeightGrad.multiply(learningRate));
    }

    public void updateInputWeights(Matrix inputWeightGrad, double learningRate) {
        this.inputWeights = this.inputWeights.subtract(inputWeightGrad.multiply(learningRate));
    }

    public void updateOutputWeights(Matrix outputWeightGrad, double learningRate) {
        this.outputWeights = this.outputWeights.subtract(outputWeightGrad.multiply(learningRate));
    }

    public void updateCellCandidateWeights(Matrix cellCandidateWeightGrad, double learningRate) {
        this.cellCandidateWeights = this.cellCandidateWeights.subtract(cellCandidateWeightGrad.multiply(learningRate));
    }

    public void updateForgetBiases(Matrix forgetBiasGrad, double learningRate) {
        this.forgetBiases = this.forgetBiases.subtract(forgetBiasGrad.multiply(learningRate));
    }

    public void updateInputBiases(Matrix inputBiasGrad, double learningRate) {
        this.inputBiases = this.inputBiases.subtract(inputBiasGrad.multiply(learningRate));
    }

    public void updateOutputBiases(Matrix outputBiasGrad, double learningRate) {
        this.outputBiases = this.outputBiases.subtract(outputBiasGrad.multiply(learningRate));
    }

    public void updateCellCandidateBiases(Matrix cellCandidateBiasGrad, double learningRate) {
        this.cellCandidateBiases = this.cellCandidateBiases.subtract(cellCandidateBiasGrad.multiply(learningRate));
    }

    /**
     * Combines all four gate weight matrices into a single concatenated matrix.
     * Useful for external optimisers or model saving.
     * @return A concatenated Matrix of all weights.
     */
    public Matrix getWeights() {
        return forgetWeights.concatenate(inputWeights).concatenate(outputWeights).concatenate(cellCandidateWeights);
    }

    /**
     * Deconstructs a concatenated weight matrix and updates internal gate weights.
     * @param weights The concatenated weight matrix to split and load.
     */
    public void setWeights(Matrix weights) {
        this.forgetWeights = weights.split(0, hiddenSize);
        this.inputWeights = weights.split(hiddenSize, 2 * hiddenSize);
        this.outputWeights = weights.split(2 * hiddenSize, 3 * hiddenSize);
        this.cellCandidateWeights = weights.split(3 * hiddenSize, 4 * hiddenSize);
    }

    /**
     * Combines all four gate bias vectors into a single concatenated matrix.
     * @return A concatenated Matrix of all biases.
     */
    public Matrix getBiases() {
        return forgetBiases.concatenate(inputBiases).concatenate(outputBiases).concatenate(cellCandidateBiases);
    }

    /**
     * Deconstructs a concatenated bias matrix and updates internal gate biases.
     * @param biases The concatenated bias matrix to split and load.
     */
    public void setBiases(Matrix biases) {
        this.forgetBiases = biases.split(0, hiddenSize);
        this.inputBiases = biases.split(hiddenSize, 2 * hiddenSize);
        this.outputBiases = biases.split(2 * hiddenSize, 3 * hiddenSize);
        this.cellCandidateBiases = biases.split(3 * hiddenSize, 4 * hiddenSize);
    }

    /**
     * Initializes the combined weight matrix for a single LSTM gate using a hybrid
     * Orthogonal and Xavier strategy.
     * <p>
     * To ensure training stability, this method splits the initialization:
     * <ul>
     * <li><b>Recurrent Weights (hidden-to-gate):</b> Initialized using an
     * {@code Orthogonal} matrix. This preserves the gradient norm as it flows
     * through the hidden state over time, preventing vanishing gradients.</li>
     * <li><b>Input Weights (input-to-gate):</b> Initialized using {@code Xavier}
     * (Glorot) initialization to ensure the incoming signal from the character
     * embeddings is scaled appropriately for the gate's activation functions.</li>
     * </ul>
     * </p>
     * <p>
     * The final matrix is concatenated vertically and then transposed to result in a
     * matrix of shape {@code [hiddenSize x (hiddenSize + inputSize)]}.
     * </p>
     *
     * @param inputSize  The dimension of the input vector (x_t).
     * @param hiddenSize The dimension of the hidden state (h_t).
     * @return A {@link Matrix} containing the combined orthogonal and Xavier weights.
     */
    public Matrix initGateWeights(int inputSize, int hiddenSize) {
        // Recurrent part
        Matrix recurrentPart = Matrix.randomOrthogonal(hiddenSize, hiddenSize);

        // Input part
        Matrix inputPart = Matrix.randomXavier(inputSize, hiddenSize);

        return recurrentPart.concatenate(inputPart).transpose();
    }
}
