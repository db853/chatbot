package layers;

import core.Matrix;

import java.util.ArrayList;
import java.util.List;

public class EmbeddingLayer implements Layer {
    private Matrix embeddings;
    private final int vocabSize;
    private final int embeddingDim;
    private Matrix embeddingGradients;

    private final List<int[]> lastInputIndices = new ArrayList<>();
    int timeStep = 0;

    public EmbeddingLayer(int vocabSize, int embeddingDim) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;

        this.embeddings = Matrix.randomXavier(vocabSize, embeddingDim);
        this.embeddingGradients = Matrix.fillValue(vocabSize, embeddingDim, 0);
    }

    /**
     * Forward pass: looks up embeddings for input word IDs.
     *
     * @param input Matrix of word IDs (integers), shape: (1 x batchSize). The rows are always 1 as the encoder steps
     *              through the sentences word by word
     * @return Matrix of embeddings, shape: (embeddingDim x batchSize)
     */
    @Override
    public Matrix forward(Matrix input) {
        int batchSize = input.getCols();

        // Store the input for the backward pass
        int[] lastInputIndex = new int[batchSize];

        // Create intermediate output matrix
        Matrix output = new Matrix(batchSize, embeddingDim);

        // Look up the embedding for each word
        for (int i = 0; i < batchSize; i++) {
            int wordID = (int) input.getData()[0][i];
            lastInputIndex[i] = wordID;

            // Copy the embedding vector for this word
            for (int j = 0; j < embeddingDim; j++) {
                output.set(i, j, embeddings.getData()[wordID][j]);
            }
        }
        this.lastInputIndices.add(lastInputIndex);
        return output.transpose();
    }

    /**
     * Backward pass: accumulates gradients for embeddings that were used.
     *
     * Unlike dense layers, embeddings are sparse - only the embeddings
     * that were looked up in forward pass get gradient updates.
     *
     * @param gradOutput Gradient from next layer, shape: (embeddingDim x batchSize)
     * @return null (embedding layer is typically first, no gradient to pass back)
     */
    @Override
    public Matrix backward(Matrix gradOutput) {
        // Reset gradients
        if (embeddingGradients == null) {
            this.embeddingGradients = Matrix.fillValue(vocabSize, embeddingDim, 0);
        }

        int batchSize = gradOutput.getCols();
        int[] lastInputIndex = this.lastInputIndices.get(timeStep);
        timeStep++;

        // Accumulate gradients only for the words that were used
        for (int i = 0; i < batchSize; i++) {
            int wordID = lastInputIndex[i];

            for (int j = 0; j < embeddingDim; j++) {
                double incomingGrad = gradOutput.getData()[j][i];
                double currentGrad = embeddingGradients.getData()[wordID][j];
                embeddingGradients.set(wordID, j, incomingGrad + currentGrad);
            }
        }
        return null;
    }

    /**
     * Updates embeddings using accumulated gradients.
     *
     * @param learningRate Step size for gradient descent
     */
    @Override
    public void updateWeights(double learningRate) {
        // Update embeddings: embeddings = embeddings - learning rate * gradient
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                double currentValue = embeddings.getData()[i][j];
                double gradient = embeddingGradients.getData()[i][j];
                double newValue = currentValue - learningRate * gradient;
                embeddings.set(i, j, newValue);
            }
        }
    }

    @Override
    public Matrix getWeights() {
        return embeddings;
    }

    @Override
    public void setWeights(Matrix weights) {
        this.embeddings = weights;
    }

    @Override
    public Matrix getWeightGradients() {
        return embeddingGradients;
    }

    public void resetState() {
        this.lastInputIndices.clear();
        this.timeStep = 0;
    }

    public void resetGradients() {
        this.embeddingGradients = null;
    }


}
