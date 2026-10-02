package test;

import activation.Softmax;
import core.MathUtils;
import core.Matrix;
import layers.DenseLayer;
import layers.LSTMLayer;
import loss.CategoricalCrossEntropyLoss;
import optimiser.Adam;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LSTMTest {
    // Create network
    // Architecture: 3 input -> 16 hidden -> 3 output
    private static final LSTMLayer hiddenLayer = new LSTMLayer(3, 16);

    private static final DenseLayer outputLayer = new DenseLayer(16, 3);
    private static final Softmax outputActivation = new Softmax();

    private static final double learningRate = 0.001;
    private static final Adam adam = new Adam(learningRate);

    public static void main(String[] args) {
        System.out.println("=== LSTM Test ===\n");

        Matrix a = new Matrix(new double[][] {
                {1}, {0}, {0}
        });
        Matrix b = new Matrix(new double[][] {
                {0}, {1}, {0}
        });
        Matrix c = new Matrix(new double[][] {
                {0}, {0}, {1}
        });

        List<Matrix> inputs = List.of(a, b, c);

        List<Matrix> targets = List.of(b, c, a);

        double epochs = 5000;

        System.out.println("Network Architecture:");
        System.out.println("  Input: 3 neurons");
        System.out.println("  Hidden: 16 neurons (Sigmoid and Tanh activation)");
        System.out.println("  Output: 3 neurons (Softmax activation)");
        System.out.println("\nTraining Parameters:");
        System.out.println("  Learning Rate: " + learningRate);
        System.out.println("  Epochs: " + epochs);
        System.out.println("\n" + "=".repeat(50) + "\n");

        // Training
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            hiddenLayer.resetState();
            hiddenLayer.resetGradients();
            outputLayer.resetState();
            outputLayer.resetGradients();

            // FORWARD PASS

            List<Matrix> hiddenStates = hiddenLayer.forward(inputs);
            List<Matrix> logits = new ArrayList<>();
            List<Matrix> predictions = new ArrayList<>();

            for (Matrix hiddenState : hiddenStates) {
                Matrix logit = outputLayer.forward(hiddenState);
                Matrix prediction = outputActivation.forward(logit);
                logits.add(logit);
                predictions.add(prediction);
            }

            CategoricalCrossEntropyLoss cce = new CategoricalCrossEntropyLoss();

            for (int t = 0; t < predictions.size(); t++) {
                double loss = cce.compute(predictions.get(t), targets.get(t));
                totalLoss += loss;
            }

            // BACKWARD PASS
            List<Matrix> hiddenStateGrads = new ArrayList<>();

            for (int t = 0; t < predictions.size(); t++) {
                Matrix lossGrad = cce.gradient(predictions.get(t), targets.get(t));
                Matrix softmaxLinearGrad = outputActivation.backward(logits.get(t), lossGrad);
                Matrix hiddenStateGrad = outputLayer.backward(softmaxLinearGrad);
                hiddenStateGrads.add(hiddenStateGrad);
            }

            hiddenLayer.backward(hiddenStateGrads);

            hiddenLayer.scaleGradients(1.0 / inputs.size());
            outputLayer.scaleGradients(1.0 / inputs.size());

            // UPDATE WEIGHTS
            adam.update(hiddenLayer);
            adam.update(outputLayer);

            System.out.print((epoch % 100 == 0) ? "Epoch: " + epoch + " Average Loss: " + totalLoss / inputs.size() +
                    "\n" : "");
        }

        System.out.println("\n" + "=".repeat(50));
        System.out.println("Training Complete!\n");

        // TESTING
        List<Matrix> testing = new ArrayList<>();
        List<Integer> expectedIndices = new ArrayList<>();

        for (int i = 0; i < 333; i++) {
            testing.add(a);
            expectedIndices.add(1);
            testing.add(b);
            expectedIndices.add(2);
            testing.add(c);
            expectedIndices.add(0);
        }

        hiddenLayer.resetState();
        outputLayer.resetState();

        double correctness = 0.0;
        List<Matrix> resultH = hiddenLayer.forward(testing);

        for (int t = 0; t < resultH.size(); t++) {
            Matrix hiddenState =  resultH.get(t);
            int expectedIndex = expectedIndices.get(t);

            Matrix logits = outputLayer.forward(hiddenState);
            Matrix prediction = outputActivation.forward(logits);

            System.out.println(Arrays.toString(prediction.getColumn(0)));
            int predictedIndex = MathUtils.argMax(prediction.getColumn(0));

            if (predictedIndex == expectedIndex) {
                correctness++;
            }
        }

        System.out.println("Correctness: " + (correctness / resultH.size()) * 100.0);
    }
}
