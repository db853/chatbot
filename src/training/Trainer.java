package training;

import core.MathUtils;
import core.Matrix;
import loss.CategoricalCrossEntropyLoss;
import model.EncoderDecoderOutput;
import model.Seq2SeqModel;
import nlp.DataPreprocessor;
import optimiser.Adam;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Trainer {
    private final Seq2SeqModel model;
    private final DataPreprocessor dp;
    private final CategoricalCrossEntropyLoss cce;
    private final Adam adam;

    public Trainer(Seq2SeqModel model, int batchSize, double learningRate) {
        this.model = model;
        this.dp = new DataPreprocessor(batchSize);
        this.cce = new CategoricalCrossEntropyLoss(1, true);
        this.adam = new Adam(learningRate);
    }

    public void train(String filename, int epochs) {
        dp.loadAndProcessFile(filename);

        List<Matrix> encoderInputs = dp.getEncoderInputs();
        List<Matrix> decoderInputs = dp.getDecoderInputs();
        List<Matrix> decoderOutputs = dp.getDecoderOutputs();

        List<Integer> batchIndices = new ArrayList<>();

        for (int i = 0; i < encoderInputs.size(); i++) {
            batchIndices.add(i);
        }

        for (int epoch = 1; epoch <= epochs; epoch++) {
            // Shuffle order of data to help with training
            Collections.shuffle(batchIndices);

            double totalLoss = 0;

            for (int batch : batchIndices) {
                model.resetState();
                model.resetGradients();

                Matrix encoderInput = encoderInputs.get(batch);
                Matrix decoderInput = decoderInputs.get(batch);
                Matrix decoderOutput = decoderOutputs.get(batch);

                // FORWARD PASS

                EncoderDecoderOutput contextState = model.encode(encoderInput);
                model.primeDecoder(contextState);
                List<Matrix> predictions = model.decode(decoderInput);
                List<Matrix> targets = oneHotEncode(decoderOutput, predictions.getFirst());

                for (int t = 0; t < predictions.size(); t++) {
                    double loss = cce.compute(predictions.get(t), targets.get(t));
                    totalLoss += loss;
                }

                // BACKWARD PASS

                List<Matrix> lossGrads = new ArrayList<>();

                for (int t = 0; t < predictions.size(); t++) {
                    Matrix lossGrad = cce.gradient(predictions.get(t), targets.get(t));
                    lossGrads.add(lossGrad);
                }

                model.backward(lossGrads);
                model.update(adam);

            }

            System.out.print((epoch % 10 == 0) ? "Epoch: " + epoch + " Average Loss: " + totalLoss /
                    encoderInputs.size() + "\n" : "");
        }
    }

    public void evaluate(String prompt) {
        model.resetState();
        Matrix encoderInput = dp.promptToMatrix(prompt);

        Matrix decoderInput = new Matrix(1, 1);
        decoderInput.set(0, 0, dp.getIndexFromToken("<START>"));

        System.out.print("Prompt: " + prompt + "\nResponse: ");
        int maxResponseLength = 15;

        EncoderDecoderOutput contextState = model.encode(encoderInput);
        model.primeDecoder(contextState);

        for (int t = 0; t < maxResponseLength; t++) {
            Matrix nextWord = model.step(decoderInput);

            int nextWordIndex = MathUtils.argMax(nextWord.getColumn(0));

            if (dp.getTokenFromIndex(nextWordIndex).equalsIgnoreCase("<END>")) {
                break;
            }

            String word = dp.getTokenFromIndex(nextWordIndex);
            System.out.print(word + " ");

            decoderInput.set(0, 0, nextWordIndex);
        }
        System.out.println();
    }

    private List<Matrix> oneHotEncode(Matrix decoderOutput, Matrix prediction) {
        List<Matrix> targets = new ArrayList<>();

        for (int i = 0; i < decoderOutput.getRows(); i++) {
            Matrix oneHotEncode = new Matrix(prediction.getRows(), prediction.getCols());

            for (int j = 0; j < decoderOutput.getCols(); j++) {
                double value = decoderOutput.getData()[i][j];
                oneHotEncode.set((int) value, j, 1);
            }
            targets.add(oneHotEncode);
        }

        return targets;
    }
}
