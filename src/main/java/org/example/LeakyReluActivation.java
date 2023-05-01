package org.example;

public class LeakyReluActivation extends ActivationFunction{
    LeakyReluActivation(int featureSize) {
        super(featureSize);
    }

    @Override
    Vector out(Vector z) {
        Vector a = new Vector(z.size());

        for(int i=0;i<a.size();i++) {
            float answer;

            if(z.x(i) >= 0) {
                answer = z.x(i);
            }
            else {
                answer = 0.01f * z.x(i);
            }

            a.setX(i, answer);
        }

        return a;
    }

    @Override
    Vector derivativeAtZ(Vector z) {
        Vector a = new Vector(z.size());

        for(int i=0;i<a.size();i++) {
            a.setX(i, z.x(i) > 0 ? 1 : 0.01f);
        }

        return a;
    }
}
