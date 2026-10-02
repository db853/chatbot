package model;

import core.Matrix;
import layers.EmbeddingLayer;
import layers.LSTMLayer;
import optimiser.Adam;

import java.util.ArrayList;
import java.util.List;

/**
 * The Encoder part of the Sequence-to-Sequence (Seq2Seq) model.
 * <p>
 * It reads the input text step-by-step, passes it through an LSTM, and
 * saves the very last hidden and cell states. These last states act as
 * a summary context that gets sent to the Decoder.
 * </p>
 * @see Decoder
 * @see LSTMLayer
 * @see EmbeddingLayer
 */
public class Encoder {
    private final EmbeddingLayer embeddingLayer;
    private final LSTMLayer lstmLayer;

    /**
     * Constructs a new Encoder configuration.
     *
     * @param vocabSize The total number of unique tokens in the source language vocabulary.
     * @param embeddingDim The dimensionality of the source token embedding space.
     * @param hiddenSize The number of internal recurrent features used by the underlying LSTM cell.
     */
    public Encoder(int vocabSize, int embeddingDim, int hiddenSize) {
        this.embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);
        this.lstmLayer = new LSTMLayer(embeddingDim, hiddenSize);
    }

    /**
     * Executes the forward pass over a source input sequence.
     * <p>
     * Each input word at a given time step is converted into the dense embedding. These are passed into the lstm and
     * the final temporal hidden and cell states are grabbed and passed as a context state for use by the Decoder.
     * </p>
     *
     * @param input A Matrix holding source sentence sequence indices (sequenceLength x batchSize).
     * @return A data container holding the final hidden and cell context states to prime the Decoder.
     */
    public EncoderDecoderOutput forward(Matrix input) {
        List<Matrix> embeddedingList = new ArrayList<>();

        int sequenceLength = input.getRows();

        // Step 1: Loop through the sequence and get embeddings for each input word
        for (int t = 0; t < sequenceLength; t++) {
            Matrix wordsAtTimeT = input.split(t, t + 1);
            Matrix embedding = embeddingLayer.forward(wordsAtTimeT);
            embeddedingList.add(embedding);
        }


        // Step 2: Pass all inputs through the LSTM layer
        lstmLayer.forward(embeddedingList);

        // Step 3: Grab the very last states to use as the Decoder's context
        Matrix finalHiddenState = lstmLayer.getLastHiddenState();
        Matrix finalCellState = lstmLayer.getLastCellState();

        // Return the context state to be used by the Decoder
        return new EncoderDecoderOutput(finalHiddenState, finalCellState);
    }

    /**
     * Executes Backpropagation Through Time (BPTT) for the Encoder.
     * <p>
     * Receives the accumulated hidden state gradient from the Decoder and adds this to the end of a list. The rest of
     * list is filled with matrices of 0 to match the same size as the sequence length. This list is backproped through
     * the lstm and then finally the embedding layer.
     * </p>
     *
     * @param finalHiddenStateGrad The incoming total accumulated gradient matching the final hidden state context.
     */
    public void backward(Matrix finalHiddenStateGrad) {
        List<Matrix> gradOutputs = new ArrayList<>();
        int sequenceLength = lstmLayer.getHistories().size();

        // Step 1: Fill earlier time steps with zeros because external loss only hits the last step
        for (int i = 0; i < sequenceLength - 1; i++) {
            gradOutputs.add(Matrix.fillValue(finalHiddenStateGrad.getRows(), finalHiddenStateGrad.getCols(), 0));
        }
        // Step 2: Place the incoming decoder gradient into the very last position
        gradOutputs.add(finalHiddenStateGrad);

        // Step 3: Pass the compiled list backward through the LSTM
        List<Matrix> embeddingGrads = lstmLayer.backward(gradOutputs);

        // Step 4: Finish by running gradients backward through the Embedding layer
        for (Matrix embeddingGrad : embeddingGrads) {
            embeddingLayer.backward(embeddingGrad);
        }
    }

    public void resetState() {
        this.lstmLayer.resetState();
        this.embeddingLayer.resetState();
    }

    public void resetGradients() {
        this.lstmLayer.resetGradients();
        this.embeddingLayer.resetGradients();
    }

    public void update(Adam adam) {
        adam.update(embeddingLayer);
        adam.update(lstmLayer);
    }
}
