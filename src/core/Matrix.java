package core;

import lombok.Data;

import java.util.Random;

@Data
public class Matrix {
    private final int rows;
    private final int cols;
    private final double[][] data;

    /**
     * Constructs a matrix with specified dimensions, initialized to zero.
     * Use this for creating "placeholder" matrices like bias vectors or
     * accumulating gradients before weight updates.
     * @param rows The number of vertical elements
     * @param cols The number of horizontal elements
     */
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    /**
     * Constructs a matrix from a pre-existing 2D double array.
     * Useful for converting training data or hardcoded weights into a
     * Matrix object for math operations.
     * @param data The 2D array to wrap as a Matrix
     */
    public Matrix (double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = data;
    }

    /**
     * Xavier (Glorot) initialization for weight matrices.
     *
     * Purpose: Keeps activations and gradients at similar scales throughout the network,
     * preventing vanishing or exploding gradients during training.
     *
     * How it works: Scales random values by sqrt(2 / (inputSize + outputSize)) to maintain
     * variance of activations roughly constant as data flows through layers.
     *
     * Best for: Layers using sigmoid or tanh activation functions.
     *
     * Why it matters: Proper initialization is critical for deep networks to train effectively.
     * Without it, early layers might receive tiny gradients (vanishing) or enormous gradients
     * (exploding), preventing the network from learning.
     *
     * @param rows Number of rows (typically input size for weight matrices)
     * @param cols Number of columns (typically output size for weight matrices)
     * @return Matrix filled with random values scaled for stable training
     */
    public static Matrix randomXavier(int rows, int cols) {
        Matrix matrix = new Matrix(rows, cols);
        Random random = new Random();

        double scale = Math.sqrt(2.0 / (rows + cols));

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix.data[i][j] = random.nextGaussian() * scale;
            }
        }

        return matrix;
    }

    /**
     * He (Kaiming) initialization for weight matrices.
     *
     * Purpose: Optimized initialization specifically for ReLU activation functions, which
     * zero out half of their inputs (all negative values become 0).
     *
     * How it works: Scales random values by sqrt(2 / inputSize) to compensate for the
     * variance reduction caused by ReLU killing half the neurons. Uses only input size
     * (not output size) in the calculation because ReLU's behavior depends on the input.
     *
     * Best for: Layers using ReLU, Leaky ReLU, or other ReLU variants.
     *
     * Why it matters: ReLU networks need larger initial weights than sigmoid/tanh networks
     * because ReLU sets negative values to zero. Without He initialization, ReLU networks
     * often suffer from "dying ReLU" problems where neurons stop learning entirely.
     *
     * @param rows Number of rows (typically input size for weight matrices)
     * @param cols Number of columns (typically output size for weight matrices)
     * @return Matrix filled with random values scaled for ReLU activation stability
     */
    public static Matrix randomHe(int rows, int cols) {
        Matrix matrix = new Matrix(rows, cols);
        Random random = new Random();

        double scale = Math.sqrt(2.0 / rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix.data[i][j] = random.nextGaussian() * scale;
            }
        }

        return matrix;
    }

    /**
     * Initializes a matrix with random orthogonal values using the Gram-Schmidt process.
     * <p>
     * Orthogonal initialization is critical for recurrent networks (LSTMs/RNNs) because
     * it ensures the eigenvalues of the weight matrix are approximately 1.0. This allows
     * signals to propagate through many time steps without exploding or vanishing,
     * as the transformation acts as a rotation rather than a scaling of the vector space.
     * </p>
     * <p><b>Implementation Details:</b></p>
     * <ul>
     * <li>Starts with a matrix filled with standard Gaussian noise (mean 0, variance 1).</li>
     * <li>Applies the Gram-Schmidt process to decorrelate columns, making them perpendicular.</li>
     * <li>Normalizes each column to have a Euclidean norm (L2) of 1.0.</li>
     * <li>Includes a small epsilon (1e-8) during division to ensure numerical stability.</li>
     * </ul>
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix. For true orthogonality,
     * the matrix should ideally be square (rows == cols).
     * @return A new {@link Matrix} containing the orthogonal weights.
     */
    public static Matrix randomOrthogonal(int rows, int cols) {
        Matrix matrix = new Matrix(rows, cols);
        Random random = new Random();

        // 1. Start with random Gaussian values
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix.data[i][j] = random.nextGaussian();
            }
        }

        // 2. Gram-Schmidt process to make columns orthogonal
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < j; k++) {
                double dotProduct = 0;
                double normK = 0;
                for (int i = 0; i < rows; i++) {
                    dotProduct += matrix.data[i][j] * matrix.data[i][k];
                    normK += matrix.data[i][k] * matrix.data[i][k];
                }

                // Subtract projection
                double projection = dotProduct / (normK + 1e-18);
                for (int i = 0; i < rows; i++) {
                    matrix.data[i][j] -= projection * matrix.data[i][k];
                }
            }

            // Normalise the column (make it unit length)
            double norm = 0;
            for (int i = 0; i < rows; i++) {
                norm += matrix.data[i][j] * matrix.data[i][j];
            }
            norm = Math.sqrt(norm);
            for (int i = 0; i < rows; i++) {
                matrix.data[i][j] /= (norm + 1e-18);
            }
        }
        return matrix;
    }

    /**
     * Creates a matrix of a specific size filled with a single value.
     * Often used to create constant matrices or to initialize biases
     * to a specific small value (like 0.01).
     * @param rows Number of rows
     * @param cols Number of columns
     * @param value The value to fill every cell with
     * @return A new matrix where every element is the fill value
     */
    public static Matrix fillValue(int rows, int cols, double value) {
        Matrix matrix = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix.data[i][j] = value;
            }
        }

        return matrix;
    }

    /**
     * Sets the value of a specific cell in the matrix.
     * Used for manual data entry or fine-tuning individual weights.
     * @param row The row index
     * @param col The column index
     * @param value The new value for the cell
     */
    public void set(int row, int col, double value) {
        data[row][col] = value;
    }

    /**
     * Performs element-wise addition of two matrices.
     * @param other The matrix to add to the current one
     * @return A new matrix representing the sum
     * @throws IllegalArgumentException if dimensions do not match
     */
    public Matrix add(Matrix other) {
        if (this.rows != other.rows || cols != other.cols) {
            throw new IllegalArgumentException("Matrix rows and columns don't match");
        }

        Matrix addedMatrix = new Matrix(this.rows, this.cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                addedMatrix.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }

        return addedMatrix;
    }

    /**
     * Adds a column vector to every column in this matrix.
     * Crucial for "Minibatch" processing, where a single bias vector needs
     * to be added to multiple inputs in one operation.
     * @param other A matrix with 1 column (column vector)
     * @return A new matrix with the vector added to each column
     */
    public Matrix addWithBroadcast(Matrix other) {
        if (this.rows != other.rows) {
            throw new IllegalArgumentException("Matrix rows don't match");
        }

        Matrix addedMatrix = new Matrix(this.rows, this.cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                addedMatrix.data[i][j] = this.data[i][j] + other.data[i][0];
            }
        }

        return addedMatrix;
    }

    /**
     * Collapses all columns into a single column containing the sum of each row.
     * Used during backpropagation to calculate the total gradient for a bias
     * vector across an entire batch of data.
     * @return A matrix with dimensions (rows x 1)
     */
    public Matrix sumColumns() {
        Matrix columnVector = new Matrix(this.rows, 1);

        for (int i = 0; i < rows; i++) {
            double rowSum = 0;
            for (int j = 0; j < cols; j++) {
                rowSum += data[i][j];
            }
            columnVector.data[i][0] = rowSum;
        }
        return columnVector;
    }

    /**
     * Performs element-wise subtraction of two matrices.
     * @param other The matrix to subtract from the current one
     * @return A new matrix representing the difference
     */
    public Matrix subtract(Matrix other) {
        if (this.rows != other.rows && cols != other.cols) {
            throw new IllegalArgumentException("Matrix rows and columns don't match");
        }

        Matrix subtractedMatrix = new Matrix(this.rows, this.cols);

        for (int i = 0; i < subtractedMatrix.rows; i++) {
            for (int j = 0; j < subtractedMatrix.cols; j++) {
                subtractedMatrix.data[i][j] = this.data[i][j] - other.data[i][j];
            }
        }

        return subtractedMatrix;
    }

    /**
     * Multiplies every element in the matrix by a single scalar value.
     * @param scalar The number to multiply each element by
     * @return A new matrix scaled by the input
     */
    public Matrix multiply(double scalar) {
        Matrix matrix = new Matrix(this.rows, this.cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix.data[i][j] = this.data[i][j] * scalar;
            }
        }
        return matrix;
    }

    /**
     * This performs a Hadamard product (element-wise multiplication) on two matrices of the same size
      * @param other The other matrix in the multiplication
     * @return The resulting matrix of the operation
     */
    public Matrix hadamardProduct(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix rows and columns don't match");
        }

        Matrix hadamardMatrix = new Matrix(this.rows, this.cols);

        for (int i = 0; i < hadamardMatrix.rows; i++) {
            for (int j = 0; j < hadamardMatrix.cols; j++) {
                hadamardMatrix.data[i][j] = this.data[i][j] * other.data[i][j];
            }
        }

        return hadamardMatrix;
    }

    /**
     * Performs matrix multiplication on two matrices
     * @param other The other matrix in the multiplication
     * @return The resulting matrix from the operation
     */
    public Matrix matrixMultiplication(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("The columns of one matrix don't match the rows of the other");
        }

        Matrix mulMatrix = new Matrix(this.rows, other.cols);

        for (int i = 0; i < mulMatrix.rows; i++) {
            for (int j = 0; j < mulMatrix.cols; j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                mulMatrix.data[i][j] = sum;
            }
        }

        return mulMatrix;
    }

    /**
     * Multiplies a matrix by a 1D vector.
     * A specialised, faster version of matrix multiplication used when
     * processing a single input example instead of a batch.
     * @param vector The double array to multiply against
     * @return A 1D array representing the resulting vector
     */
    public double[] matrixVectorMultiplication(double[] vector) {
        if (this.cols != vector.length) {
            throw new IllegalArgumentException("The columns of one matrix don't match the rows of the other");
        }

        double[] result = new double[rows];

        for (int i = 0; i < this.rows; i++) {
            double sum = 0.0;

            for (int j = 0; j < this.cols; j++) {
                sum += this.data[i][j] * vector[j];
            }
            result[i] = sum;
        }
        return result;
    }

    /**
     * Transposes a matrix meaning that the columns and rows are swapped
     * @return The transposed matrix
     */
    public Matrix transpose() {
        Matrix transposedMatrix = new Matrix(this.cols, this.rows);

        for (int i = 0; i < transposedMatrix.rows; i++) {
            for (int j = 0; j < transposedMatrix.cols; j++) {
                transposedMatrix.data[i][j] = this.data[j][i];
            }
        }

        return transposedMatrix;
    }

    /**
     * Outputs the contents of the matrix to the console.
     */
    public void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.print(data[i][j] + " ");
            }
            System.out.println();
        }
    }

    /**
     * Extracts a specific column as a 1D array.
     * @param j The column index to retrieve
     * @return A double array of the column's values
     */
    public double[] getColumn(int j) {
        double[] column = new double[rows];

        for (int i = 0; i < rows; i++) {
            column[i] = data[i][j];
        }
        return column;
    }

    /**
     * Replaces an entire column in the matrix with new values.
     * @param j The column index to replace
     * @param column The new values for that column
     */
    public void setColumn(int j, double[] column) {
        for (int i = 0; i < rows; i++) {
            data[i][j] = column[i];
        }
    }

    /**
     * Scales the matrix to ensure its L2 norm does not exceed a specified threshold.
     * <p>
     * A norm is a mathematical concept used to measure the size or magnitude of a vector.
     * In the context of machine learning, norms are used to constrain the magnitude of model parameters to prevent
     * overfitting and encourage sparsity.
     * </p>
     * <p>
     * This is a technique used to prevent the "Exploding Gradient" problem in LSTMs.
     * Unlike standard value clipping, which clamps individual elements, norm clipping
     * scales the entire matrix uniformly. This preserves the <b>direction</b> of the
     * gradient vector while limiting its magnitude.
     * </p>
     * * <h3>Mathematical Process:</h3>
     * <ol>
     * <li>Calculate the L2 Norm: {@code ||g|| = sqrt(sum(element^2))}</li>
     * <li>If {@code norm > threshold}, multiply every element by {@code threshold / norm}.</li>
     * <li>If {@code norm <= threshold}, return the original matrix.</li>
     * </ol>
     *
     * @param threshold The maximum allowed magnitude (length) of the gradient matrix.
     * Typical values for LSTMs are 1.0 or 5.0.
     * @return A new {@link Matrix} scaled to the threshold if necessary;
     * otherwise, the current matrix.
     */
    public Matrix clipByL2Norm(double threshold) {
        double sumOfSquares = 0.0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sumOfSquares += data[i][j] * data[i][j];
            }
        }

        double l2Norm = Math.sqrt(sumOfSquares);

        if (l2Norm > threshold) {
            double scalar = threshold / l2Norm;
            return this.multiply(scalar);
        }

        return this;
    }

    /**
     * Concatenates two matrices vertically (row-wise stacking).
     * <p>
     * This method creates a new matrix by placing Matrix {@code b} directly
     * underneath Matrix {@code this}. This is used in the LSTM forward pass
     * to combine the previous hidden state (h_t-1) and the current
     * input (x_t) into a single input vector.
     * </p>
     * @param other The bottom matrix (typically the current input x_t)
     * @return A new Matrix with rows equal to {@code a.rows + b.rows} and
     * columns equal to the original column count.
     * @throws IllegalArgumentException If the column counts of the two matrices
     * do not match (e.g., if batch sizes are inconsistent).
     */
    public Matrix concatenate(Matrix other) {
        if (this.cols != other.cols) {
            throw new IllegalArgumentException("Columns must be the same for concatenation");
        }

        int combinedRows = this.rows + other.rows;
        int cols = this.cols;

        Matrix concatenatedMatrix = new Matrix(combinedRows, cols);

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < cols; j++) {
                concatenatedMatrix.data[i][j] = this.data[i][j];
            }
        }

        for (int i = 0; i < other.rows; i++) {
            for (int j = 0; j < cols; j++) {
                concatenatedMatrix.data[i + this.rows][j] = other.data[i][j];
            }
        }

        return concatenatedMatrix;
    }

    public Matrix split(int startRow, int endRow) {
        Matrix splitMatrix = new Matrix(endRow - startRow, this.cols);

        for (int i = startRow; i < endRow; i++) {
            for (int j = 0; j < this.cols; j++) {
                splitMatrix.data[i - startRow][j] = this.data[i][j];
            }
        }

        return splitMatrix;
    }

    public double norm() {
        double sumOfSquares = 0.0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sumOfSquares += data[i][j] * data[i][j];
            }
        }

        return Math.sqrt(sumOfSquares);
    }

}
