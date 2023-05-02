package org.example;

public class SoftMaxActivation extends ActivationFunction {
    SoftMaxActivation(int featureSize, int neurons) {
        super(featureSize, neurons);
    }

    @Override
    Vector out(Vector z) {
        float total = 0;
        float max = Float.MIN_VALUE;

        for(int i=0;i<z.size();i++) {
            max = Math.max(max, z.x(i));
        }

        for(int i=0;i<z.size();i++) {
            total += Math.exp(z.x(i) - max);
        }

        Vector a = new Vector(z.size());
        for(int i=0;i<z.size();i++) {
            a.setX(i, (float) (Math.exp(z.x(i) - max) / total));
        }

        return a;
    }

    @Override
    Vector derivativeAtZ(Vector z, Vector y) {
        int subscript = -1;
        float mainA = -1;

        Vector a = out(z);

        for(int i=0;i<y.size();i++) {
            if(y.x(i) == 1) {
                subscript = i;
                mainA = a.x(i);
                break;
            }
        }

        for(int i=0;i<a.size();i++) {
            if(i == subscript) {
                a.setX(i, mainA * (1 - mainA));
            }
            else {
                a.setX(i, - mainA * a.x(i));
            }
        }

        return a;
    }

    @Override
    Vector[] getW() {
        Vector[] W  = new Vector[1];
        W[0] = new Vector(featureSize);
        W[0].randomise();

        return W;
    }

    @Override
    Vector getB() {
        Vector b = new Vector(1);

        b.randomise();
        return b;
    }
}