package org.example;

public class LinearActivation extends MatrixActivation {
    LinearActivation(int featureSize, int neurons) {
        super(featureSize, neurons);
    }

    @Override
    public Vector out(Vector z) {
        return z;
    }

    @Override
    public Vector derivativeAtZ(Vector z, Vector y) {
        return new Vector(z.size(), 1);
    }

}