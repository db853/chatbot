package inference;

import model.Seq2SeqModel;
import training.Trainer;
import java.util.Scanner;

public class Chatbot {
    private static final Scanner sc = new Scanner(System.in);

    public static void main(String[] args) {
        Trainer trainer = loadAndTrainModel(94, 24, 96, 1, 0.001,
                500, "./test_file.tsv");

        promptModel(trainer);
    }

    public static Trainer loadAndTrainModel(int vocabSize, int embeddingDim, int hiddenSize, int batchSize,
                                                 double learningRate, int epochs, String filename) {
        Seq2SeqModel model = new Seq2SeqModel(vocabSize, embeddingDim, hiddenSize);
        Trainer trainer = new Trainer(model, batchSize, learningRate);

        System.out.println("TRAINING");
        System.out.println("====================\n\n");

        trainer.train(filename, epochs);
        return trainer;
    }

    public static void promptModel(Trainer trainer) {
        System.out.print("\n\nEnter your prompt: ");
        String prompt = sc.nextLine();

        while (!prompt.equalsIgnoreCase("xxx")) {
            trainer.evaluate(prompt);
            System.out.print("\n\nEnter your prompt: ");
            prompt = sc.nextLine();
        }
    }
}
