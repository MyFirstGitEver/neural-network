package org.example;

public class CrossEntropy implements SimpleNeuralNetwork.Loss {
    @Override
    public float loss(Vector v, Vector y) throws Exception {
        for(int i=0;i<y.size();i++) {
            if(y.x(i) == 1) {
                return (float) -Math.log(v.x(i) + 0.00001f);
            }
        }

        throw new Exception("Label has some problem with it!");
    }

    @Override
    public Vector derivativeByA(Vector A, Vector Y) {
        Vector derivative = new Vector(A.size());

        for(int i=0;i<Y.size();i++) {
            if(Y.x(i) == 1) {
                if(Double.isNaN(-1 / (A.x(i) + 10e-5f))) {
                    int m = 3;
                }

                derivative.setX(i, -1 / (A.x(i) + 10e-5f));
            }
            else{
                derivative.setX(i, 0);
            }
        }

        return derivative;
    }
}