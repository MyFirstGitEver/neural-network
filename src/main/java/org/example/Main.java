package org.example;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws Exception {
        ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Plant\\plant.xlsx");

        Pair<Vector, Vector>[] dataset = new Pair[reader.getRowCount() - 1]; // exclude the first row
        Pair<Vector, Float>[] dataset2 = new Pair[reader.getRowCount() - 1]; // exclude the first row

        for(int i=1;i<=dataset.length;i++) {
            Object[] data = reader.getRow(i, 0);
            Object[] x = Arrays.copyOfRange(data, 0, data.length - 1);

            dataset[i - 1] = new Pair<>(new Vector(x), new Vector(((Double)data[data.length - 1]).floatValue()));
            dataset2[i - 1] = new Pair<>(new Vector(x), ((Double)data[data.length - 1]).floatValue());
        }

        Pair<Vector, Vector>[] xTrain = Arrays.copyOfRange(dataset, 0, (int) (0.8 * dataset.length));
        Pair<Vector, Vector>[] xTest = Arrays.copyOfRange(dataset,
                (int) (0.8 * dataset.length), dataset.length);

        Pair<Vector, Float>[] xTrain2 = Arrays.copyOfRange(dataset2, 0, (int) (0.8 * dataset.length));
        Pair<Vector, Float>[] xTest2 = Arrays.copyOfRange(dataset2,
                (int) (0.8 * dataset.length), dataset.length);

        //normalise(xTrain, xTest);

        SimpleNeuralNetwork model = new SimpleNeuralNetwork(construct(xTrain[0].first.size()), new MSE(), xTrain);
        //0.000001f
        model.train(0.0001f, 1_000);

        neuralTest(xTest, model);
        normalRegression(xTrain2, xTest2);
    }

    static void normalise(Pair<Vector, Float>[] xTrain, Pair<Vector, Float>[] xTest) {
        float[] mean = new float[xTrain[0].first.size()];
        float[] std = new float[xTrain[0].first.size()];

        for(Pair<Vector, Float> p : xTrain) {
            for(int i=0;i<p.first.size();i++) {
                mean[i] += p.first.x(i);
            }
        }

        for(int i=0;i<mean.length;i++) {
            mean[i] /= xTrain.length;
        }

        for(Pair<Vector, Float> p : xTrain) {
            for(int i=0;i<p.first.size();i++) {
                float term = (p.first.x(i) - mean[i]);
                std[i] += term * term;
            }
        }

        for(int i=0;i<std.length;i++) {
            std[i] = (float) Math.sqrt(std[i]);
        }

        // Normalise
        for(Pair<Vector, Float> p : xTrain) {
            for(int i=0;i<std.length;i++) {
                p.first.setX(i, (p.first.x(i) - mean[i]) / std[i]);
            }
        }

        for(Pair<Vector, Float> p : xTest) {
            for(int i=0;i<std.length;i++) {
                p.first.setX(i, (p.first.x(i) - mean[i]) / std[i]);
            }
        }
    }

    static DenseLayer[] construct(int inputSize) {
        return new DenseLayer[] {
                new DenseLayer(new ReluActivation(inputSize), 10),
                new DenseLayer(new LinearActivation(10), 1),
        };
    }

    static void normalRegression(Pair<Vector, Float>[] xTrain, Pair<Vector, Float>[] xTest) {
        System.out.println("_______\n");
        LinearRegression model = new LinearRegression(new PolynomialPredictor(), xTrain);
        model.train(0.000001f, 8_000);
        System.out.println("Error on train set: " + model.cost());

        float l2Error = 0.0f;
        for(Pair<Vector, Float> point : xTest) {
            float term = model.predict(point.first) - point.second;

            l2Error += term * term;
        }

        System.out.println("Error on test set: " + (l2Error) / (2 * xTest.length));
    }

    static void neuralTest(Pair<Vector, Vector>[] xTest, SimpleNeuralNetwork model) {
        System.out.println("Error on train set: " + model.cost());

        float l2Error = 0.0f;
        for(Pair<Vector, Vector> point : xTest) {
            float term = model.predict(point.first).x(0) - point.second.x(0);

            l2Error += term * term;
        }

        System.out.println("Error on test set: " + (l2Error) / (2 * xTest.length));
    }
}