package org.example;

public class SoftMaxActivation extends ActivationFunction {
    SoftMaxActivation(int featureSize, int neurons) {
        super(featureSize, neurons);
    }

    @Override
    Vector out(Vector z) {
        double total = 0;
        double max = Double.MIN_VALUE;

        for(int i=0;i<z.size();i++) {
            max = Math.max(max, z.x(i));
        }

        for(int i=0;i<z.size();i++) {
            total += Math.exp(z.x(i) - max);
        }

        Vector a = new Vector(z.size());
        for(int i=0;i<z.size();i++) {
            a.setX(i, (double) (Math.exp(z.x(i) - max) / total));
        }

        return a;
    }

    @Override
    Vector derivativeByZ(Vector z, Vector y) {
        int subscript = -1;
        double mainA = -1;

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
}