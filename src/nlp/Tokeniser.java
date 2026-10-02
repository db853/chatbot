package nlp;

import java.util.ArrayList;
import java.util.List;

/**
 * The {@code Tokeniser} class provides functionality to preprocess and split
 * raw English text into a sequence of individual words and punctuation tokens.
 * <p>
 * It utilises a multistep natural language processing (NLP) pipeline to normalise
 * text casing, handle irregular contraction expansions, split regular contractions
 * dynamically, and isolate punctuation marks from alphanumeric words.
 * </p>
 */
public class Tokeniser {
    /**
     * Tokenises a given string of text by passing it through a sequential normalisation
     * and string-cleansing pipeline before splitting it by whitespace.
     *
     * @param text The raw input string containing sentences to be tokenised.
     * @return A {@link List} of lowercase string tokens representing isolated words,
     * contraction fragments, and punctuation marks.
     */
    public List<String> tokenise(String text) {
        // Force the line to lowercase to different cases do not treat the same words as different
        String lowercaseLine = text.toLowerCase();

        // Handle three special contraction instances (can't, won't and ain't)
        lowercaseLine = specialContractionExpansion(lowercaseLine);

        // Handle remaining n't contractions
        lowercaseLine = notContractionSplitting(lowercaseLine);

        // Handle remaining contractions
        lowercaseLine = contractionSplitting(lowercaseLine);

        // Handle all other punctuation
        lowercaseLine = separatingPunctuation(lowercaseLine);

        // Split resulting line by any number of blank spaces to get a list of tokens
        String[] tokenisedLineArray = lowercaseLine.trim().split("\\s+");
        List<String> tokenisedLine = new ArrayList<>();

        // Items must be added one at a time to new ArrayList so add works in the DataPreprocessor class.
        // Arrays converted to Lists makes the List immutable
        for (String token : tokenisedLineArray) {
            tokenisedLine.add(token);
        }

        return tokenisedLine;
    }

    /**
     * Expands highly irregular English contractions into their complete multi-word
     * equivalents to avoid generating non-standard vocabulary root fragments.
     * <p>
     * Specifically handles:
     * <ul>
     * <li>"can't" {@code ->} "can not"</li>
     * <li>"won't" {@code ->} "will not"</li>
     * <li>"ain't" {@code ->} "is not"</li>
     * </ul>
     * </p>
     *
     * @param text The normalised lowercase text string.
     * @return The text string with specific irregular contractions expanded.
     */
    private String specialContractionExpansion(String text) {
        String newLine = text.replaceAll("can't", "can not");
        newLine = newLine.replaceAll("won't", "will not");
        newLine = newLine.replaceAll("ain't", "is not");
        return newLine;
    }

    /**
     * Isolates negative verbal contractions ending in "n't" by injecting a space
     * directly before the 'n'.
     * <p>
     * For example, transforms "don't" into "do n't". This allows downstream splitting
     * to treat the root verb and the negative modifier as distinct structural features.
     * </p>
     *
     * @param text The text string after irregular expansions have been processed.
     * @return The text string with space-separated negative contraction fragments.
     */
    private String notContractionSplitting(String text) {
        return text.replaceAll("n't", " n't");
    }

    /**
     * Isolates standard pronoun and possessive contractions (such as 'm, 'll, 've, 're, 'd, and 's)
     * by inserting a space directly before the apostrophe.
     * <p>
     * This method uses a negative lookbehind regex assertion {@code (?<!n)} to intentionally
     * ignore any apostrophes preceded by the character 'n', preserving the formatting established
     * by the preceding negative contraction step.
     * </p>
     *
     * @param text The text string after negative contraction isolation.
     * @return The text string with spaces inserted before valid, non-negative contraction apostrophes.
     */
    private String contractionSplitting(String text) {
        // Only adds a space in front of apostrophes that do not have the letter 'n' behind them as that contraction
        // was handled in a previous step
        return text.replaceAll("(?<!n)'", " '");
    }

    /**
     * Locates all global Unicode punctuation marks across the text string and wraps
     * them with padding spaces on both sides.
     * <p>
     * This method utilises a regular expression character class intersection/subtraction
     * {@code ([\p{P}&&[^']])} to capture all formal punctuation while strictly protecting
     * apostrophes, ensuring that contraction structures are not broken apart before
     * the final split.
     * </p>
     *
     * @param text The text string after all variations of contractions have been handled.
     * @return The text string with spaces wrapped around every non-apostrophe punctuation character.
     */
    private String separatingPunctuation(String text) {
        // This looks for all punctuation that is not an apostrophe
        String regexPattern = "([\\p{P}&&[^']])";

        // The character found using regex is stored on $1
        return text.replaceAll(regexPattern, " $1 ");
    }
}
