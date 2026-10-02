package layers.cells;

import core.Matrix;

/**
 * A container for the outputs of an LSTM cell's forward pass at a single time step.
 * <p>
 * This record stores the dual-state output characteristic of LSTMs: the hidden state
 * (short-term memory) and the cell state (long-term memory).
 * </p>
 * @param hiddenState The hidden state matrix produced for the current time step, used as
 * output and as context for the next time step.
 * @param cellState The updated cell state matrix (long-term memory) to be passed
 * to the next time step.
 */
public record LSTMForwardResult(Matrix hiddenState, Matrix cellState, LSTMHistory history) {}
