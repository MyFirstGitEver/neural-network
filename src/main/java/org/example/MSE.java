package org.example;

public class MSE implements SimpleNeuralNetwork.Loss {
    @Override
    public float loss(Vector v, Vector y) {
        float loss = 0;

        for(int i=0;i<v.size();i++) {
            float term = v.x(i) - y.x(i);

            loss += term * term;
        }

        return loss;
    }

    @Override
    public Vector derivativeByA(Vector A, Vector Y) {
        Vector v = new Vector(A.size());

        for(int i=0;i<v.size();i++) {
            v.setX(i, 2 * (A.x(i) - Y.x(i)));
        }

        return v;
    }
}
