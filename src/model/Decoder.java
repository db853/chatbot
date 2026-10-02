package model;

import activation.Softmax;
import core.Matrix;
import layers.DenseLayer;
import layers.EmbeddingLayer;
import layers.LSTMLayer;
import optimiser.Adam;

import java.util.ArrayList;
import java.util.List;

/**
 * The Decoder part of the Sequence-to-Sequence (Seq2Seq) model.
 * <p>
 * It takes the final hidden and cell states from the Encoder to start its own LSTM.
 * Then it loops through the target words, turns them into embeddings, passes them
 * through the LSTM, and projects the outputs into vocabulary scores.
 * </p>
 * @see Encoder
 * @see LSTMLayer
 * @see DenseLayer
 * @see Softmax
 */
public class Decoder {
    private final EmbeddingLayer embeddingLayer;
    private final LSTMLayer lstmLayer;
    private final DenseLayer denseLayer;
    private final Softmax softmax = new Softmax();

    // Save the logits for the backward pass
    private final List<Matrix> logits = new ArrayList<>();

    /**
     * Constructs a new Decoder with required structural parameters.
     *
     * @param vocabSize    The total number of unique tokens in the target vocabulary.
     * @param embeddingDim The dimensionality of the token embedding space.
     * @param hiddenSize   The number of features inside the LSTM cell hidden state.
     */
    public Decoder(int vocabSize, int embeddingDim, int hiddenSize) {
        this.embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);
        this.lstmLayer = new LSTMLayer(embeddingDim, hiddenSize);
        this.denseLayer = new DenseLayer(hiddenSize, vocabSize);
    }

    /**
     * Performs a forward pass through the Decoder network over a target sequence.
     * <p>
     * This initialises the initial hidden and cell states for the LSTM using the context state from the Encoder. The
     * words at each time interval for each sentence in the batch are converted into a dense embedding. This list is
     * passed to the lstm to produce hidden states. These hidden states are then turned into vocab predictions using
     * a Dense layer and Softmax.
     * </p>
     *
     * @param targetInput A Matrix holding target sequence tokens, structured row-by-row.
     * @return A list of vocabulary token probability distribution matrices, one per sequence step.
     */
    public List<Matrix> forward(Matrix targetInput) {
        // Clear old cached states before processing a new sequence batch
        logits.clear();

        List<Matrix> embeddingList = new ArrayList<>();
        int sequenceLength = targetInput.getRows();

        // Step 1: Loop through the sequence and get embeddings for each word
        for (int t = 0; t < sequenceLength; t++) {
            Matrix wordsAtTimeT = targetInput.split(t, t + 1);
            Matrix embedding = embeddingLayer.forward(wordsAtTimeT);
            embeddingList.add(embedding);
        }

        // Step 2: Feed all word embeddings into the LSTM layer
        List<Matrix> hiddenStates = lstmLayer.forward(embeddingList);

        // Step 3: Turn LSTM outputs into vocabulary predictions
        List<Matrix> predictions = new ArrayList<>();
        for (Matrix hiddenState : hiddenStates) {
            Matrix logit = denseLayer.forward(hiddenState);
            logits.add(logit);
            Matrix prediction = softmax.forward(logit);
            predictions.add(prediction);
        }

        return predictions;
    }

    public Matrix step(Matrix token) {
        Matrix embedding = embeddingLayer.forward(token);

        Matrix hiddenState = lstmLayer.forward(embedding);
        Matrix logit = denseLayer.forward(hiddenState);

        return softmax.forward(logit);
    }

    /**
     * Performs the backward pass for the Decoder module.
     * <p>
     * The loss errors are backproped through the softmax and dense layers. The list of the hidden state gradients for
     * each time step are backproped through the LSTM. The gradients for the embedding and then backproped to the
     * embedding layer. Finally, the gradients for the hidden state and the cell state are returned to pass back to the
     * Encoder.
     * </p>
     *
     * @param lossGrads A list containing error gradients computed by the Loss function for each time-step.
     * @return An object containing computed initial state gradients ready to backpropagate directly into the Encoder.
     */
    public EncoderDecoderOutput backward(List<Matrix> lossGrads) {
        List<Matrix> hiddenStateGrads = new ArrayList<>();

        // Step 1: Backprop through Softmax and Dense layers for each step
        for (int t = 0; t < lossGrads.size(); t++) {
            Matrix softmaxLinearGrad = softmax.backward(logits.get(t), lossGrads.get(t));
            Matrix hiddenStateGrad = denseLayer.backward(softmaxLinearGrad);
            hiddenStateGrads.add(hiddenStateGrad);
        }

        // Step 2: Run the full list of gradients back through the LSTM layer
        List<Matrix> embeddingGrads = lstmLayer.backward(hiddenStateGrads);

        // Step 3: Map individual step gradients back through the Embedding layer
        for (Matrix embeddingGrad : embeddingGrads) {
            embeddingLayer.backward(embeddingGrad);
        }

        // Step 4: Pull the saved initial state gradients out of the LSTM layer
        return new EncoderDecoderOutput(lstmLayer.getInitialHiddenStateGrad(), lstmLayer.getInitialCellStateGrad());
    }

    public void primeDecoder(EncoderDecoderOutput contextState) {
        lstmLayer.assignInitialStates(contextState.finalHiddenState(), contextState.finalCellState());
    }

    public void resetState() {
        this.lstmLayer.resetState();
        this.embeddingLayer.resetState();
        this.denseLayer.resetState();

        this.logits.clear();
    }

    public void resetGradients() {
        this.lstmLayer.resetGradients();
        this.embeddingLayer.resetGradients();
        this.denseLayer.resetGradients();
    }

    public void update(Adam adam) {
        adam.update(embeddingLayer);
        adam.update(lstmLayer);
        adam.update(denseLayer);
    }

}
