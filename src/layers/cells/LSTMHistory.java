package layers.cells;

import core.Matrix;

/**
 * Stores a snapshot of the LSTM cell's internal activations and inputs from a single
 * forward pass time step.
 * <p>
 * This data is cached during the forward pass and retrieved in reverse order during
 * Backpropagation Through Time (BPTT). It contains the intermediate "linear" values
 * required by activation functions to calculate gradients, as well as the gate
 * activations used to distribute error signals.
 * </p>
 * @param combinedInput The concatenated [h_t-1, x_t] matrix used as input to all gates.
 * @param previousCellState The cell state (C_t-1) from the step prior to this one.
 * @param ct The finalised cell state (C_t) after gate updates.
 * @param ft The Forget Gate activation (values 0 to 1).
 * @param it The Input Gate activation (values 0 to 1).
 * @param ot The Output Gate activation (values 0 to 1).
 * @param cCandidate The Candidate Cell State (tanh output, values -1 to 1).
 * @param fLinear The pre-activation (linear) sum for the Forget Gate.
 * @param iLinear The pre-activation (linear) sum for the Input Gate.
 * @param oLinear The pre-activation (linear) sum for the Output Gate.
 * @param cCandidateLinear The pre-activation (linear) sum for the Candidate State.
 */
public record LSTMHistory(Matrix combinedInput, Matrix previousCellState,
                          Matrix ft, Matrix it, Matrix ot, Matrix cCandidate, Matrix ct,
                          Matrix fLinear, Matrix iLinear, Matrix oLinear, Matrix cCandidateLinear) {}
