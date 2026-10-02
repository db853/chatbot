package layers.cells;

import core.Matrix;

/**
 * A container for the gradients calculated during an LSTM cell's backward pass.
 * <p>
 * Contains two types of gradients:
 * <ol>
 * <li><b>State Gradients:</b> Used to propagate the error signal backward to the
 * previous time step (t-1).</li>
 * <li><b>Weight/Bias Gradients:</b> Used to update the cell's parameters to
 * minimise the loss function.</li>
 * </ol>
 * </p>
 * @param previousHiddenStateGrad The gradient with respect to the previous hidden state (h_t-1).
 * @param previousCellStateGrad The gradient with respect to the previous cell state (C_t-1).
 * @param currentInputGrad The gradient with respect to the current input matrix (x_t)
 * @param forgetWeightGrad Gradient for the Forget Gate weights.
 * @param inputWeightGrad Gradient for the Input Gate weights.
 * @param outputWeightGrad Gradient for the Output Gate weights.
 * @param cellCandidateWeightGrad Gradient for the Candidate State weights.
 * @param forgetBiasGrad Gradient for the Forget Gate biases.
 * @param inputBiasGrad Gradient for the Input Gate biases.
 * @param outputBiasGrad Gradient for the Output Gate biases.
 * @param cellCandidateBiasGrad Gradient for the Candidate State biases.
 */
public record LSTMBackwardResult(
        Matrix previousHiddenStateGrad, Matrix previousCellStateGrad, Matrix currentInputGrad,
        Matrix forgetWeightGrad, Matrix inputWeightGrad, Matrix outputWeightGrad, Matrix cellCandidateWeightGrad,
        Matrix forgetBiasGrad, Matrix inputBiasGrad, Matrix outputBiasGrad, Matrix cellCandidateBiasGrad) {}
