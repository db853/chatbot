package test;

import core.Matrix;
import loss.CategoricalCrossEntropyLoss;
import model.EncoderDecoderOutput;
import model.Seq2SeqModel;
import nlp.DataPreprocessor;
import nlp.Tokeniser;

import java.util.ArrayList;
import java.util.List;

public class Seq2SeqModelTest {
    public static void main(String[] args) {

        DataPreprocessor dp = new DataPreprocessor(1);

        dp.loadAndProcessFile("./test_file");

        Matrix encoderInput = dp.getEncoderInputs().getFirst();
        Matrix decoderInput = dp.getDecoderInputs().getFirst();
        Matrix decoderOutput = dp.getDecoderOutputs().getFirst();

        Seq2SeqModel seq2SeqModel = new Seq2SeqModel(7, 4, 4);

        EncoderDecoderOutput contextState = seq2SeqModel.encode(encoderInput);
        seq2SeqModel.primeDecoder(contextState);
        List<Matrix> predictions = seq2SeqModel.decode(decoderInput);

        List<Matrix> lossGrads = new ArrayList<>();
        CategoricalCrossEntropyLoss cce = new CategoricalCrossEntropyLoss();

        List<Matrix> targets = new ArrayList<>();

        for (int i = 0; i < decoderOutput.getRows(); i++) {
            Matrix oneHotEncode = new Matrix(predictions.getFirst().getRows(), predictions.getFirst().getCols());
            double value = decoderOutput.getData()[i][0];
            oneHotEncode.set((int) value, 0, 1);
            targets.add(oneHotEncode);
        }

        for (int t = 0; t < predictions.size(); t++) {
            Matrix lossGrad = cce.gradient(predictions.get(t), targets.get(t));
            lossGrads.add(lossGrad);
        }

        seq2SeqModel.backward(lossGrads);
    }

}
