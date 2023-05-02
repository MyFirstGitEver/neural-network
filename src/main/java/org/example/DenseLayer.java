package org.example;

import java.util.Arrays;

abstract class ActivationFunction {
    protected final int featureSize;
    protected final int neurons;

    ActivationFunction(int featureSize, int neurons) {
        this.featureSize = featureSize;
        this.neurons = neurons;
    }

    abstract Vector out(Vector z);
    abstract Vector derivativeAtZ(Vector z, Vector y);
    abstract Vector[] getW();
    abstract Vector getB();
}

public class DenseLayer {
    private final ActivationFunction function;

    private final Vector[] W;
    private final Vector B;

    public DenseLayer(ActivationFunction function) {
        this.function = function;

        W = function.getW();
        B = function.getB();
    }

    public Pair<Vector, Vector> output(Vector v) {
        if(v.size() != W[0].size()) {
            throw new RuntimeException("Wrong input dimension!");
        }

        // Calculate z for activation function
        float[] points = new float[function.neurons];

        for(int i=0;i<function.neurons;i++) {
            points[i] = W[i].dot(v) + B.x(i);
        }

        Vector z = new Vector(points);
        return new Pair<>(z, function.out(z));
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

    public Vector derivativeByZ(Vector Z, Vector y) {
        return function.derivativeAtZ(Z, y);
    }

    public Matrix transposeOfW() {
        return Matrix.transpose(W);
    }

    public Pair<Integer, Integer> wShape() {
        return new Pair<>(W.length, W[0].size());
    }

    public int bShape() {
        return B.size();
    }
}