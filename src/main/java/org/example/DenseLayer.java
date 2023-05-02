package org.example;

import java.util.Arrays;

abstract class ActivationFunction {
    private final int featureSize;

    ActivationFunction(int featureSize) {
        this.featureSize = featureSize;
    }

    abstract Vector out(Vector z);
    abstract Vector derivativeAtZ(Vector z);
    int featureSize() {
        return featureSize;
    }
}

public class DenseLayer {
    private final ActivationFunction function;
    private final int neurons;

    private final Vector[] W;
    private final Vector B;

    public DenseLayer(ActivationFunction function, int neurons) {
        this.function = function;
        this.neurons = neurons;

        W = new Vector[neurons];
        B = new Vector(neurons, 0);
        B.randomise();

        for(int i=0;i<W.length;i++) {
            W[i] = new Vector(function.featureSize(), 0);
            W[i].randomise();
        }
    }

    public Pair<Vector, Vector> output(Vector v) {
        if(v.size() != function.featureSize()) {
            throw new RuntimeException("Wrong input dimension!");
        }

        // Calculate z for activation function
        float[] points = new float[neurons];

        for(int i=0;i<neurons;i++) {
            points[i] = W[i].dot(v) + B.x(i);
        }

        Vector z = new Vector(points);
        return new Pair<>(z, function.out(z));
    }

    public int numberOfNeurons() {
        return neurons;
    }

    public int featureSize() {
        return function.featureSize();
    }

    public void update(
            Vector[] firstMomentW,
            Vector[] secondMomentW,
            Vector firstMomentB,
            Vector secondMomentB,
            float learningRate)
            throws Exception {

        for(int i=0;i<firstMomentW.length;i++) {
            W[i].subtract(firstMomentW[i].divide(secondMomentW[i].sqrt(), 10e-8f).scaleBy(learningRate));
        }

        for(int i=0;i<firstMomentB.size();i++) {
            B.subtract(firstMomentB.divide(secondMomentB.sqrt(), 10e-8f).scaleBy(learningRate));
        }
    }

    public Vector derivativeByZ(Vector Z) {
        return function.derivativeAtZ(Z);
    }

    public Matrix transposeOfW() {
        return Matrix.transpose(W);
    }
}