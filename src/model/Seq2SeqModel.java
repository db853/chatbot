package model;

import core.Matrix;
import optimiser.Adam;

import java.util.List;

/**
 * The master Sequence-to-Sequence (Seq2Seq) model.
 * <p>
 * It connects the Encoder and the Decoder together. It passes the input sequence
 * through the Encoder to get a summary context, and then feeds that context into
 * the Decoder to predict the final output sequence.
 * </p>
 */
public class Seq2SeqModel {
    private final Encoder encoder;
    private final Decoder decoder;

    /**
     * Initialises the entire Seq2Seq model by creating the Encoder and Decoder.
     *
     * @param vocabSize    The total number of unique words in the vocabulary.
     * @param embeddingDim The size of the word embedding vectors.
     * @param hiddenSize   The size of the LSTM hidden and cell states.
     */
    public Seq2SeqModel(int vocabSize, int embeddingDim, int hiddenSize) {
        this.encoder = new Encoder(vocabSize, embeddingDim, hiddenSize);
        this.decoder = new Decoder(vocabSize, embeddingDim, hiddenSize);
    }

    public EncoderDecoderOutput encode(Matrix sourceInput) {
        return encoder.forward(sourceInput);
    }

    public void primeDecoder(EncoderDecoderOutput contextState) {
        decoder.primeDecoder(contextState);
    }

    public List<Matrix> decode(Matrix targetInput) {
        return decoder.forward(targetInput);
    }

    public Matrix step(Matrix token) {
        return decoder.step(token);
    }

    /**
     * Runs the backward pass for the entire model.
     * <p>
     * It takes the gradients from the loss function, runs them backward through
     * the Decoder, and then passes the resulting hidden state gradient back
     * through the Encoder to complete the training step.
     * </p>
     *
     * @param lossGrads The gradients from the loss function for each step.
     */
    public void backward(List<Matrix> lossGrads) {
        // Step 1: Backprop through the Decoder to get the gradients for the connection states
        EncoderDecoderOutput finalStateGrads = decoder.backward(lossGrads);

        Matrix finalHiddenStateGrad = finalStateGrads.finalHiddenState();

        // Step 2: Pass it back through the Encoder to finish the chain
        encoder.backward(finalHiddenStateGrad);
    }

    public void resetState() {
        encoder.resetState();
        decoder.resetState();
    }

    public void resetGradients() {
        encoder.resetGradients();
        decoder.resetGradients();
    }

    public void update(Adam adam) {
        encoder.update(adam);
        decoder.update(adam);
    }
}
