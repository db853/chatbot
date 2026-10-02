package nlp;

import core.Matrix;
import lombok.Getter;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Handles data preparation pipeline tasks for training a Recurrent Neural Network (LSTM).
 * Utilises a memory-efficient two-pass streaming design:
 * <ol>
 * <li>Pass 1: Blistering-fast initial scan to register unique words and complete the vocabulary registry.</li>
 * <li>Pass 2: Stateful streaming aggregation of characters into sentences, numerical encoding, and dynamic mini-batch
 * generation.</li>
 * </ol>
 */
public class DataPreprocessor {
    private final Tokeniser tokeniser = new Tokeniser();
    private final Vocabulary vocabulary = new Vocabulary();
    private final int batchSize;

    // Stored mini-batch grids prepared for the main execution training loop
    @Getter
    private final List<Matrix> encoderInputs = new ArrayList<>();

    @Getter
    private final List<Matrix> decoderInputs = new ArrayList<>();

    @Getter
    private final List<Matrix> decoderOutputs = new ArrayList<>();

    /**
     * Constructs a DataPreprocessor with necessary component modules and training batch rules.
     *
     * @param batchSize  The fixed number of continuous sentence matrices compiled into a single training step.
     */
    public DataPreprocessor(int batchSize) {
        this.batchSize = batchSize;
    }

    /**
     * Executes the complete ingestion engine on a targeted raw text corpus file.
     *
     * @param filename Path linking to the raw corpus text file on the system.
     * @return The vocab size of the data file
     */
    public int loadAndProcessFile(String filename) {
        buildVocabulary(filename);
        gatherSentences(filename);

        return vocabulary.vocabSize();
    }

    public Matrix promptToMatrix(String prompt) {
        List<Double> encodedPrompt = encodeSentence(prompt, false);
        return convertToMatrix(List.of(encodedPrompt)).transpose();
    }

    /**
     * Pass 1: Scans the target text file purely to discover and index unique tokens.
     * Operates as an unordered, boundary-agnostic flat bag-of-words loop for maximum performance.
     *
     * @param filename Path linking to the raw corpus text file.
     */
    private void buildVocabulary(String filename) {
        List<String> allTokens = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;

            while ((line = br.readLine()) != null) {
                // Instantly bypass empty or unformatted lines to preserve computational overhead
                if (line.trim().isEmpty()) {
                    continue;
                }

                // Slice the raw line into raw text arrays and stockpile them
                List<String> tokens = tokeniser.tokenise(line);
                allTokens.addAll(tokens);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Commit the gathered tokens to establish the text-to-numerical index mapping definitions
        vocabulary.buildFromTokens(allTokens);
    }

    /**
     * Pass 2: Streams lines from the file system, contextually builds structurally intact sentences,
     * encodes tokens using vocabulary index maps, and groups them sequentially into training mini-batches.
     *
     * @param filename Path linking to the raw corpus text file.
     */
    private void gatherSentences(String filename) {
        List<List<Double>> promptBatch = new ArrayList<>();
        List<List<Double>> responseBatch = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;

            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }

                // Configured for TSV files
                String[] promptResponse = line.split("\t");

                String prompt = promptResponse[0];
                List<Double> promptEncoded = encodeSentence(prompt, false);
                promptBatch.add(promptEncoded);

                String response = promptResponse[1];
                List<Double> responseEncoded = encodeSentence(response, true);
                responseBatch.add(responseEncoded);

                if (promptBatch.size() == batchSize) {
                    buildTrainingMatrices(promptBatch, responseBatch);
                    promptBatch.clear();
                    responseBatch.clear();
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Checks if data is still present in the lists if batch size is uneven
        if (!promptBatch.isEmpty()) {
            buildTrainingMatrices(promptBatch, responseBatch);
            promptBatch.clear();
            responseBatch.clear();
        }

    }

    /**
     * Converts an aggregated sentence string into an array list of unique vocabulary sequence ID values.
     * Explicitly inserts mandatory sequence tracking boundary markers.
     *
     * @param sentence A grammatically finalised, isolated line of text.
     * @return A sequential list containing numeric double indices mapping to the vocabulary repository.
     */
    private List<Double> encodeSentence(String sentence, boolean isResponse) {
        List<String> tokenisedSentence = tokeniser.tokenise(sentence);
        List<Double> encodedSentence = new ArrayList<>();

        // Embed strict semantic context markers indicating start/end parameters for sequential prediction
        if (isResponse) {
            tokenisedSentence.addFirst("<START>");
            tokenisedSentence.add("<END>");
        }

        for (String token : tokenisedSentence) {
            double id = vocabulary.getID(token);
            encodedSentence.add(id);
        }
        return encodedSentence;
    }

    /**
     * Manufacturing Engine: Adjusts irregular batches via dynamic padding criteria, transforms
     * standard text grids into column vector matrices, and performs structural time-shift matrix splits.
     *
     * @param promptBatch A list of all encoded prompts in the batch.
     * @param responseBatch A list of all encoded responses in the batch
     *
     */
    private void buildTrainingMatrices(List<List<Double>> promptBatch, List<List<Double>> responseBatch) {
        // Find the max length for each batch for padding
        int maxPromptLength = getBatchMaxSequenceLength(promptBatch);
        int maxResponseLength = getBatchMaxSequenceLength(responseBatch);

        // Normalise batch dimensions based strictly on the current block constraint limits
        padSentences(promptBatch, maxPromptLength);
        padSentences(responseBatch, maxResponseLength);

        // Transpose the matrix 90 degrees to construct vertical column vectors (Sequence Length x Batch Size)
        // This matches the column vector structure of the neural network
        Matrix encoderInput = convertToMatrix(promptBatch).transpose();
        encoderInputs.add(encoderInput);

        Matrix fullResponse = convertToMatrix(responseBatch).transpose();

        // Split the grid down the rows (time-steps)
        // decoderInput keeps <START> but drops <END> (rows 0 to maxResponseLength - 1)
        Matrix decoderInput = fullResponse.split(0, fullResponse.getRows() - 1);

        // decoderTarget drops <START> but keeps <END> (rows 1 to maxResponseLength)
        Matrix decoderTarget = fullResponse.split(1, fullResponse.getRows());

        decoderInputs.add(decoderInput);
        decoderOutputs.add(decoderTarget);

    }

    private Matrix convertToMatrix(List<List<Double>> data) {
        double[][] matrixData = new double[data.size()][data.getFirst().size()];

        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data.get(i).size(); j++) {
                matrixData[i][j] = data.get(i).get(j);
            }
        }

        return new Matrix(matrixData);
    }

    /**
     * Standardizes irregular text length layouts by appending vocabulary pad identifiers
     * onto elements falling short of the computed batch maximum dimension limit.
     *
     * @param tokenisedSentences Collected numeric token arrays allocated for processing.
     * @param sequenceLength    Target size criteria calculated for standard batch grid equalization.
     */
    private void padSentences(List<List<Double>> tokenisedSentences, int sequenceLength) {
        for (List<Double> sentence : tokenisedSentences) {
            while (sentence.size() < sequenceLength) {
                sentence.add(vocabulary.getID("<PAD>"));
            }
        }
    }

    /**
     * Determines the maximum length variation across active rows present inside the current local batch.
     *
     * @param tokenisedSentences Local batch subset array collection containing individual numeric token sequences.
     * @return The highest token length calculated within the evaluated batch boundaries.
     */
    private int getBatchMaxSequenceLength(List<List<Double>> tokenisedSentences) {
        int maxSequenceLength = 0;

        for (List<Double> sentence : tokenisedSentences) {
            maxSequenceLength = Math.max(maxSequenceLength, sentence.size());
        }
        return maxSequenceLength;
    }

    public String getTokenFromIndex(int index) {
        return vocabulary.getToken(index);
    }

    public double getIndexFromToken(String token) {
        return vocabulary.getID(token);
    }

}
