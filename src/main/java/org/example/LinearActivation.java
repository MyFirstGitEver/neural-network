package org.example;

public class LinearActivation extends ActivationFunction {
    LinearActivation(int featureSize) {
        super(featureSize);
    }

    @Override
    public Vector out(Vector z) {
        return z;
    }

    @Override
    public Vector derivativeAtZ(Vector z) {
        return new Vector(z.size(), 1);
    }
}