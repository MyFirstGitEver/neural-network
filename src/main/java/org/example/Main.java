package org.example;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws Exception {
        ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Sales\\advertising.xlsx");

        Pair<Vector, Float>[] dataset = new Pair[reader.getRowCount() - 1]; // exclude the first row

        for(int i=1;i<dataset.length;i++) {
            Object[] data = reader.getRow(i, 0);
            Object[] x = Arrays.copyOfRange(data, 0, data.length - 1);

            dataset[i - 1] = new Pair<>(new Vector(x), ((Double)data[data.length - 1]).floatValue());
        }

        Pair<Vector, Float>[] xTrain = Arrays.copyOfRange(dataset, 0, (int) (0.7 * dataset.length));
        Pair<Vector, Float>[] xTest = Arrays.copyOfRange(dataset, 0, (int) (0.3 * dataset.length));

        SimpleNeuralNetwork model = new SimpleNeuralNetwork(construct(xTrain[0].first.size()), new MSE(), xTrain);
        //0.000001f
        model.train(0.00001f, 1_000_000);

        int i = 3;
    }

    static DenseLayer[] construct(int inputSize) {
        return new DenseLayer[] {
                new DenseLayer(new ReluActivation(inputSize), 30),
                new DenseLayer(new ReluActivation(30), 1),
        };
    }

    static void normalRegression(Pair<Vector, Float>[] xTrain, Pair<Vector, Float>[] xTest) {
        LinearRegression model = new LinearRegression(new PolynomialPredictor(), xTrain);
        model.train(0.000000001f, 100_000);
        System.out.println("Error on train set: " + model.cost());

        float l2Error = 0.0f;
        for(Pair<Vector, Float> point : xTest) {
            float term = model.predict(point.first) - point.second;

            l2Error += term * term;
        }

        System.out.println("Error on test set: " + (l2Error) / (2 * xTest.length));
    }
}