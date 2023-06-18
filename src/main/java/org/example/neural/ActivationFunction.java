package org.example.neural;

import org.example.Vector;

import java.util.Random;

public abstract class ActivationFunction {
    protected final int featureSize;
    protected final int neurons;

    ActivationFunction(int featureSize, int neurons) {
        this.featureSize = featureSize;
        this.neurons = neurons;
    }

    abstract Vector out(Vector z);

    public abstract Vector derivativeByZ(Vector z, int aPos);

    Vector[] getW() {
        Vector[] W = new Vector[neurons];

        for (int i = 0; i < W.length; i++) {
            W[i] = new Vector(featureSize);

            for(int j=0;j<featureSize;j++) {
                W[i].setX(j, Math.random() * 0.01);
            }
        }

        return W;
    }

    Vector getB() {
        Vector b = new Vector(neurons);

        for(int j=0;j<neurons;j++) {
            b.setX(j, Math.random());
        }

        return b;
    }

    Vector z(Vector[] W, Vector B, Vector v) {
        double[] points = new double[neurons];

        for (int i = 0; i < neurons; i++) {
            points[i] = W[i].dot(v) + B.x(i);
        }

        return new Vector(points);
    }
}
