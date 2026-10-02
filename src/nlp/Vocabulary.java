package nlp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Manages the text-to-numerical index codebook for an NLP model.
 * Acts as a bidirectional lookup ledger that maps unique string tokens to individual
 * double numeric keys and vice versa. Automatically initializes mandatory special
 * semantic sequence markers used for sequence boundary and out-of-vocabulary handling.
 */
public class Vocabulary {
    private final String PAD = "<PAD>";
    private final String UNK = "<UNK>";
    private final String START = "<START>";
    private final String END = "<END>";

    private final Map<String, Double> wordToId = new HashMap<>();
    private final List<String> idToWord = new ArrayList<>();

    /**
     * Initialises a default Vocabulary instance.
     * Automatically registers standard padding, out-of-vocabulary, and sequence boundary markers
     * to guarantee they possess fixed, low-value index allocations (IDs 0.0 through 3.0).
     */
    public Vocabulary() {
        addToken(UNK);
        addToken(PAD);
        addToken(START);
        addToken(END);
    }

    /**
     * Builds and builds upon the dictionary index mappings from a provided collection of tokens.
     * Iterates through raw string inputs, filtering out duplicates via inner lookup checks.
     *
     * @param tokens A comprehensive flat list of unique or recurring words harvested from a training corpus.
     */
    public void buildFromTokens(List<String> tokens) {
        for (String token : tokens) {
            addToken(token);
        }
    }

    /**
     * Performs a reverse lookup to find the string token corresponding to a given integer ID.
     *
     * @param id The discrete index position requested from the sequence model's output.
     * @return The text string mapped to that unique integer index; returns the unknown token string
     * ({@code "<UNK>"}) if the index sits outside valid boundary constraints.
     */
    public String getToken(int id) {
        if (id > 0 && id < idToWord.size()) {
            return idToWord.get(id);
        }
        return UNK;
    }

    /**
     * Performs a forward lookup to find the unique numeric double index corresponding to a given text token.
     *
     * @param token The raw text string word being evaluated or encoded.
     * @return The unique double identifier key assigned to that token string; returns the index value
     * associated with the unknown placeholder code ({@code "<UNK>"}) if the word is unrecognised.
     */
    public double getID(String token) {
        if (wordToId.containsKey(token)) {
            return wordToId.get(token);
        }
        return wordToId.get(UNK);
    }

    /**
     * Fetches the total volume of unique elements registered in the lookup dictionary.
     *
     * @return The integer size count representing the total vocabulary capacity.
     */
    public int vocabSize() {
        return idToWord.size();
    }

    /**
     * Safely attempts to append a new unique string token to the internal lookup tables.
     * If the token is already registered, the request is silently bypassed to prevent index collision errors.
     * Assigns continuous ID numbers by referencing the total size count of the current list tracking register.
     *
     * @param token The string token targeting registration into the vocabulary maps.
     */
    private void addToken(String token) {
        if (!wordToId.containsKey(token)) {
            double newId = idToWord.size();
            wordToId.put(token, newId);
            idToWord.add(token);
        }
    }
}
