package org.example;

abstract class MatrixActivation extends ActivationFunction{
    MatrixActivation(int featureSize, int neurons) {
        super(featureSize, neurons);
    }

    @Override
    Vector[] getW() {
        Vector[] W = new Vector[neurons];

        for(int i=0;i<W.length;i++) {
            W[i] = new Vector(featureSize);
            W[i].randomise();
        }

        return W;
    }

    @Override
    Vector getB() {
        Vector b = new Vector(neurons);
        b.randomise();

        return b;
    }
}
