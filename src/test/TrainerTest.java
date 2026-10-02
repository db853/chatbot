package test;

import model.Seq2SeqModel;
import training.Trainer;

import java.util.Scanner;

public class TrainerTest {
    public static void main(String[] args) {
        Seq2SeqModel model = new Seq2SeqModel(94, 24, 96);
        Trainer trainer = new Trainer(model, 1, 0.001);

        trainer.train("./test_file.tsv", 500);

        System.out.println("\n".repeat(5));

        Scanner input = new Scanner(System.in);

        while (true) {
            System.out.print("Enter your prompt: ");
            String prompt = input.nextLine();
            trainer.evaluate(prompt);
            System.out.println("\n");
        }

    }
}
