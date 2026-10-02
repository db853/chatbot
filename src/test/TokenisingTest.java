package test;

import core.Matrix;
import nlp.DataPreprocessor;
import nlp.Tokeniser;
import nlp.Vocabulary;

import java.util.List;

public class TokenisingTest {
    public static void main(String[] args) {
        String file = "./test_file.tsv";

        int batchSize = 5;
        DataPreprocessor dataPreprocessor = new DataPreprocessor(batchSize);

        System.out.println(dataPreprocessor.loadAndProcessFile(file));



    }
}
