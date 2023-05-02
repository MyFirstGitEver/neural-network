package org.example;

public class ReluActivation extends MatrixActivation {
    ReluActivation(int featureSize, int neurons) {
        super(featureSize, neurons);
    }

    @Override
    public Vector out(Vector z) {
        Vector a = new Vector(z.size());

        for(int i=0;i<a.size();i++) {
            a.setX(i, Math.max(z.x(i), 0));
        }

        return a;
    }

    @Override
    public Vector derivativeAtZ(Vector z, Vector y) {
        Vector a = new Vector(z.size());

        for(int i=0;i<a.size();i++) {
            a.setX(i, z.x(i) <= 0 ? 0 : 1);
        }

        return a;
    }
}
