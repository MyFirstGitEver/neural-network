package org.example;

public class ReluActivation extends ActivationFunction {
    ReluActivation(int featureSize) {
        super(featureSize);
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
    public Vector derivativeAtZ(Vector z) {
        Vector a = new Vector(z.size());

        for(int i=0;i<a.size();i++) {
            a.setX(i, z.x(i) <= 0 ? 0 : 1);
        }

        return a;
    }
}
