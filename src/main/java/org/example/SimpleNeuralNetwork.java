package org.example;

import java.io.*;

public class SimpleNeuralNetwork {
    interface Loss {
        double loss(Vector v, Vector y) throws Exception;
        Vector derivativeByA(Vector A, Vector Y);
    }

    private final DenseLayer[] layers;
    private final Loss loss;
    private final Pair<Vector, Vector>[] dataset;
    private final static double beta = 0.9f;
    private final static double beta2 = 0.999f;

    public SimpleNeuralNetwork(DenseLayer[] layers, Loss loss, Pair<Vector, Vector>[] dataset) {
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

    public void train(double learningRate, int iter, int batchSize) throws Exception {
        double cost;
        int iteration = 0;

        // dW and dB holds current gradient update
        Matrix[] dW = new Matrix[layers.length];
        Vector[] dB = new Vector[layers.length];

        Matrix[] firstMomentW = new Matrix[layers.length];
        Matrix[] secondMomentW = new Matrix[layers.length];
        Vector[] firstMomentB = new Vector[layers.length];
        Vector[] secondMomentB = new Vector[layers.length];

        for(int i=0;i<layers.length;i++) {
            Pair<Integer, Integer> wShape = layers[i].wShape();

            dW[i] = new Matrix(wShape.first, wShape.second);
            dB[i] = new Vector(layers[i].bShape());

            firstMomentW[i] = new Matrix(wShape.first, wShape.second);
            secondMomentW[i] = new Matrix(wShape.first, wShape.second);

            firstMomentB[i] = new Vector(layers[i].bShape());
            secondMomentB[i] = new Vector(layers[i].bShape());
        }

        while((cost = cost()) > 0.001f && iteration < iter) {
            if(iteration % 30 == 0) {
                System.out.println(iteration + " iterations have passed. Cost: " + cost);
            }

            for(int j=0;j<dataset.length;j+= Math.min(batchSize, dataset.length - j)) {
                for (Matrix matrix : dW) {
                    matrix.reset();
                }

                for(Vector v : dB) {
                    v.reset();
                }

                computeGradient(j, j + Math.min(batchSize - 1, dataset.length - 1 - j), dW, dB);

                int m = 3;

                for(int i=0;i<layers.length;i++) {
                    firstMomentW[i].scale(beta).add(dW[i].scale(1 - beta));
                    secondMomentW[i].scale(beta2).add(dW[i].square().scale(1 - beta2));

                    firstMomentB[i].scaleBy(beta).add(dB[i].scaleBy(1 - beta));
                    secondMomentB[i].scaleBy(beta2).add(dB[i].square().scaleBy(1 - beta2));

                    layers[i].update(
                            firstMomentW[i].vectorize(true),
                            secondMomentW[i].vectorize(true),
                            firstMomentB[i],
                            secondMomentB[i], learningRate);
                }
            }

            iteration++;
        }
    }

    public double cost() throws Exception {
        double total = 0;

        for(Pair<Vector, Vector> p : dataset) {
            total += loss.loss(predict(p.first), p.second);
        }

        return total / dataset.length;
    }

    private Vector lastError(Pair<Vector, Vector> lastZAndA, Vector y) {
        Vector z = lastZAndA.first;
        Vector a = lastZAndA.second;

        return loss.derivativeByA(a, y).hadamard(layers[layers.length - 1].derivativeByZ(z, y));
    }

    private void computeGradient(int from, int to, Matrix[] dW, Vector[] dB) throws Exception {
        for(int iteration=from;iteration<=to;iteration++) {
            Pair<Vector, Vector> point = dataset[iteration];

            Pair<Vector, Vector>[] zAndA = new Pair[layers.length];

            Vector curr = point.first;
            // Forward propagation
            for (int i=0;i<layers.length;i++) {
                zAndA[i] = layers[i].output(curr);
                curr = zAndA[i].second;
            }

            Vector currError = lastError(zAndA[zAndA.length - 1], point.second);

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
                            .hadamard(layers[i - 1].derivativeByZ(zAndA[i - 1].first, point.second));

                }

                dW[i].add(gradientW);
                dB[i].add(currError);

                currError = nextError;
            }
        }
    }

    public void saveParams() throws IOException {
        File f = new File("layers");
        f.mkdir();

        for(int i=0;i<layers.length;i++) {
            File dir = new File(f, "layer " + (i + 1));
            dir.mkdir();

            BufferedWriter wWriter = new BufferedWriter(new FileWriter(dir.getPath() + "\\w"));
            BufferedWriter bWriter = new BufferedWriter(new FileWriter(dir.getPath() + "\\b"));

            Vector[] w = layers[i].getW();

            for(Vector v : w) {
                wWriter.write(v.toString());
                wWriter.newLine();
            }

            Vector b = layers[i].getB();
            bWriter.write(b.toString());

            wWriter.close();
            bWriter.close();
        }
    }

    public boolean loadParams() throws IOException {
        File f = new File("layers");

        if(!f.exists()) {
            return false;
        }

        int layerNum = f.listFiles().length;
        for(int i=0;i<layerNum;i++) {
            File dir = new File(f, "layer " + (i + 1));

            BufferedReader wReader = new BufferedReader(new FileReader(dir.getPath() + "\\w"));
            BufferedReader bReader = new BufferedReader(new FileReader(dir.getPath() + "\\b"));

            Vector[] w = layers[i].getW();

            String line;

            int index = 0;
            while((line = wReader.readLine()) != null) {
                w[index] = new Vector(line);
                index++;
            }

            layers[i].setB(new Vector(bReader.readLine()));

            wReader.close();
            bReader.close();
        }

        return true;
    }
}