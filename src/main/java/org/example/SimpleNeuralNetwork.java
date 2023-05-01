package org.example;

public class SimpleNeuralNetwork {
    interface Loss {
        float loss(Vector v, Vector y);
        Vector derivativeByA(Vector A, Vector Y);
    }

    private final DenseLayer[] layers;
    private final Loss loss;
    private final Pair<Vector, Float>[] dataset;

    public SimpleNeuralNetwork(DenseLayer[] layers, Loss loss, Pair<Vector, Float>[] dataset) {
        this.layers = layers;
        this.loss = loss;
        this.dataset = dataset;
    }

    public Vector predict(Vector v) {
        Vector curr = v;

        for (DenseLayer layer : layers) {
            curr = layer.output(curr).second;
        }

        return curr;
    }

    public void train(float learningRate, int iter) throws Exception {
        float cost;
        int iteration = 0;

        // dW and dB holds current gradient update
        Matrix[] dW = new Matrix[layers.length];
        Vector[] dB = new Vector[layers.length];

        for(int i=0;i<layers.length;i++) {
            dW[i] = new Matrix(layers[i].numberOfNeurons(), layers[i].featureSize());
            dB[i] = new Vector(layers[i].numberOfNeurons());
        }

        while((cost = cost()) > 0.001f && iteration < iter) {
            if(iteration % 30 == 0) {
                System.out.println(iteration + " iterations have passed. Cost: " + cost);
            }

            for (Matrix matrix : dW) {
                matrix.reset();
            }

            for(Pair<Vector, Float> point : dataset) {
                Pair<Vector, Vector>[] zAndA = new Pair[layers.length];

                Vector curr = point.first;
                // Forward propagation
                for (int i=0;i<layers.length;i++) {
                    zAndA[i] = layers[i].output(curr);
                    curr = zAndA[i].second;
                }

                Vector currError = lastError(zAndA[zAndA.length - 1], point.first);

                for(int i=zAndA.length - 1;i>=0;i--) {
                    Vector a;

                    if(i == 0) {
                        a = point.first;
                    }
                    else {
                        a = zAndA[i - 1].second;
                    }

                    Matrix matError = new Matrix(currError, true);
                    Vector[] gradientW = matError.mul(new Matrix(a, false));

                    Vector nextError = null;
                    if(i != 0) {
                        // W^T . lastError * g'(z)
                        // only makes sense when we're not finished
                        nextError = new Vector(layers[i].transposeOfW().mul(matError))
                                .hadamard(layers[i - 1].derivativeByZ(zAndA[i - 1].first));

                    }

                    dW[i].add(gradientW);
                    dB[i].add(currError);

                    currError = nextError;
                }
            }

            for(int i=0;i<layers.length;i++) {
                dW[i].scale(1.0f / dataset.length * learningRate);
                dB[i].scaleBy(1.0f / dataset.length * learningRate);

                layers[i].update(dW[i].vectorize(true), dB[i]);
            }

            iteration++;

            if(cost < cost()) {
                int m = 3;
            }
        }
    }

    public float cost() {
        float total = 0;

        for(Pair<Vector, Float> p : dataset) {
            total += loss.loss(predict(p.first), new Vector(p.second));
        }

        return total;
    }

    private Vector lastError(Pair<Vector, Vector> lastZAndA, Vector y) {
        Vector z = lastZAndA.first;
        Vector a = lastZAndA.second;

        return loss.derivativeByA(a, y).hadamard(layers[layers.length - 1].derivativeByZ(z));
    }
}