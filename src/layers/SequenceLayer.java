package layers;

import core.Matrix;

import java.util.List;

public interface SequenceLayer extends Layer {
    List<Matrix> forward(List<Matrix> inputs);
    List<Matrix> backward(List<Matrix> gradOutputs);

    @Override
    default Matrix forward(Matrix input) {
        List<Matrix> inputs = List.of(input);
        return forward(inputs).getFirst();
    }

    @Override
    default Matrix backward(Matrix gradOutput) {
        List<Matrix> gradOutputs = List.of(gradOutput);
        return backward(gradOutputs).getFirst();
    }
}
