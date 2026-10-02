package model;

import core.Matrix;

public record EncoderDecoderOutput(Matrix finalHiddenState, Matrix finalCellState) {}
