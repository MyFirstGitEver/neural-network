package org.example;

import org.example.neural.*;

import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Random;

//        ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Plant\\plant.xlsx");
//
//                Pair<Vector, Vector>[] dataset = new Pair[reader.getRowCount() - 1]; // exclude the first row
//        Pair<Vector, Double>[] dataset2 = new Pair[reader.getRowCount() - 1]; // exclude the first row
//
//        for(int i=1;i<=dataset.length;i++) {
//        Object[] data = reader.getRow(i, 0);
//        Object[] x = Arrays.copyOfRange(data, 0, data.length - 1);
//
//        dataset[i - 1] = new Pair<>(new Vector(x), new Vector(((Double)data[data.length - 1]).DoubleValue()));
//        dataset2[i - 1] = new Pair<>(new Vector(x), ((Double)data[data.length - 1]).DoubleValue());
//        }
//
//        Pair<Vector, Vector>[] xTrain = Arrays.copyOfRange(dataset, 0, (int) (0.8 * dataset.length));
//        Pair<Vector, Vector>[] xTest = Arrays.copyOfRange(dataset,
//        (int) (0.8 * dataset.length), dataset.length);
//
//        Pair<Vector, Double>[] xTrain2 = Arrays.copyOfRange(dataset2, 0, (int) (0.8 * dataset.length));
//        Pair<Vector, Double>[] xTest2 = Arrays.copyOfRange(dataset2,
//        (int) (0.8 * dataset.length), dataset.length);
//
//        //normalise(xTrain, xTest);
//
//        SimpleNeuralNetwork model = new SimpleNeuralNetwork(construct(xTrain[0].first.size()), new MSE(), xTrain);
//        //0.000001f
//        model.train(0.0001f, 1_000);
//
//        neuralTest(xTest, model);
//        normalRegression(xTrain2, xTest2);
// --------------------------------------------

//        ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Titanic\\train.xlsx");
//        Pair<Vector, Vector>[] dataset = reader.createLabeledDataset(0, 0, 0);
//        reader = new ExcelReader("D:\\Source code\\Data\\Titanic\\test.xlsx");
//        Pair<Vector, Vector>[] testSet = reader.createLabeledDataset(0, 0, 0);
//
//        SimpleNeuralNetwork model = new SimpleNeuralNetwork(new DenseLayer[] {
//                new DenseLayer(new ReluActivation(dataset[0].first.size(), 10)),
//                new DenseLayer(new ReluActivation(10, 5)),
//                new DenseLayer(new ReluActivation(5, 10)),
//                new DenseLayer(new SoftMaxActivation(10, 2)),
//        }, new CrossEntropy(), dataset);
//
//        //0.00001f
//        if(!model.loadParams()) {
//            model.train(0.00001f, 10_000, 60);
//            model.saveParams();
//        }
//
//        System.out.println("Cost of training set is: " + model.cost());
//
//        int hit = 0;
//
//        for (Pair<Vector, Vector> vectorVectorPair : testSet) {
//            Vector confidence = model.predict(vectorVectorPair.first);
//
//            if (confidence.x(0) >= confidence.x(1) && vectorVectorPair.second.x(0) == 1) {
//                hit++;
//            } else if (confidence.x(0) < confidence.x(1) && vectorVectorPair.second.x(1) == 1) {
//                hit++;
//            }
//        }
//
//        System.out.println("Accuracy reached: " + (double)hit / testSet.length * 100 + " %");
public class Main {
    public static void main(String[] args) throws Exception {
        int total = 0;

        for(int i=0;i<=9;i++) {
            File folder = new File("D:\\Source code\\Data\\MNIST\\training\\" + i);
            total += folder.listFiles().length;
        }

        Pair<Vector, Vector>[] dataset = new Pair[total];

        int index = 0;
        for(int i=0;i<=9;i++) {
            File folder = new File("D:\\Source code\\Data\\MNIST\\training\\" + i);

            for(File f : folder.listFiles()) {
                Vector label = new Vector(10);
                label.setX(i, 1);

                Vector v = new ImageProcessing(f.getAbsolutePath()).hog(4);
                dataset[index] = new Pair<>(v, label);
                index++;
            }
        }

        DataGetter<Vector> xGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                return dataset[i].first;
            }

            @Override
            public int size() {
                return dataset.length;
            }
        };

        DataGetter<Vector> yGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                return dataset[i].second;
            }

            @Override
            public int size() {
                return dataset.length;
            }
        };

        SimpleNeuralNetwork model = new SimpleNeuralNetwork(
                new DenseLayer[] {
                        new DenseLayer(new ReluActivation(dataset[0].first.size() , 10)),
                        new DenseLayer(new SoftMaxActivation(10, 10))
                }, new org.example.neural.CrossEntropy(), xGetter, yGetter);

        //model.loadParams("MNIST");
        model.train(0.0001, 300, 100, 5, "MNIST", true);
        //model.saveParams("MNIST");

        for(int i=0;i<=9;i++) {
            testDigit(model, i);
        }
    }

    static void softmax() throws Exception {
        ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Titanic\\train.xlsx");
        Pair<Vector, Vector>[] dataset = reader.createLabeledDataset(0, 0, 0);
        reader = new ExcelReader("D:\\Source code\\Data\\Titanic\\test.xlsx");
        Pair<Vector, Vector>[] testSet = reader.createLabeledDataset(0, 0, 0);

        DataGetter<Vector> xGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                return dataset[i].first;
            }

            @Override
            public int size() {
                return dataset.length;
            }
        };

        DataGetter<Vector> yGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                return dataset[i].second;
            }

            @Override
            public int size() {
                return dataset.length;
            }
        };

        SimpleNeuralNetwork model = new SimpleNeuralNetwork(
                new DenseLayer[] {
                        new DenseLayer(new ReluActivation(dataset[0].first.size() , 10)),
                        new DenseLayer(new SoftMaxActivation(10, 2))
                }, new org.example.neural.CrossEntropy(), xGetter, yGetter);

        model.train(0.001, 300, 150, 5, "", true);

        System.out.println("Cost of training set is: " + model.cost());

        int hit = 0;

        for (Pair<Vector, Vector> vectorVectorPair : testSet) {
            Vector confidence = model.predict(vectorVectorPair.first);

            if (confidence.x(0) >= confidence.x(1) && vectorVectorPair.second.x(0) == 1) {
                hit++;
            } else if (confidence.x(0) < confidence.x(1) && vectorVectorPair.second.x(1) == 1) {
                hit++;
            }
        }

        System.out.println("Accuracy reached: " + (double)hit / testSet.length * 100 + " %");
    }

    static void regression() throws Exception {
        ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Plant\\plant.xlsx");

        Pair<Vector, Vector>[] dataset = new Pair[reader.getRowCount() - 1]; // exclude the first row

        for (int i = 1; i <= dataset.length; i++) {
            Object[] data = reader.getRow(i, 0);
            Object[] x = Arrays.copyOfRange(data, 0, data.length - 1);

            dataset[i - 1] = new Pair<>(new Vector(x), new Vector(((Double) data[data.length - 1]).doubleValue()));
        }

        Pair<Vector, Vector>[] xTrain = Arrays.copyOfRange(dataset, 0, (int) (0.8 * dataset.length));
//        Pair<Vector, Vector>[] xTest = Arrays.copyOfRange(dataset,
//                (int) (0.8 * dataset.length), dataset.length);
        DataGetter<Vector> xGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                return xTrain[i].first;
            }

            @Override
            public int size() {
                return xTrain.length;
            }
        };

        DataGetter<Vector> yGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                return xTrain[i].second;
            }

            @Override
            public int size() {
                return xTrain.length;
            }
        };

        SimpleNeuralNetwork model = new SimpleNeuralNetwork(construct(xTrain[0].first.size()), new MSE(), xGetter, yGetter);
        //0.000001f
        model.train(0.001, 600, 100, 5, "", true);
    }

    static void testDigit(SimpleNeuralNetwork model, int digit) throws IOException {
        System.out.println("-----Testing digit " + digit + "-----");

        File test = new File("D:\\Source code\\Data\\MNIST\\testing\\" + digit);
        File[] allImages = test.listFiles();

        int hit = 0, timer = 0;
        double length = allImages.length;

        for (File f : allImages) {
            Vector hog = new ImageProcessing(f.getAbsolutePath()).hog(4);
            Vector confidence = model.predict(hog);

            int maxIndex = 0;
            double maxLevel = Double.MIN_VALUE;

            for (int i = 0; i < confidence.size(); i++) {
                if (maxLevel < confidence.x(i)) {
                    maxLevel = confidence.x(i);
                    maxIndex = i;
                }
            }

            if (maxIndex == digit) {
                hit++;
            }

            timer++;
            if (timer % 100 == 0) {
                System.out.println("Tested on " + timer + " images. Hit: " + hit);
            }
        }

        System.out.println("Accuracy recognising digit " + digit + ": " + hit / length * 100 + "%\n\n");
    }

    static DenseLayer[] construct(int inputSize) {
        return new DenseLayer[] {
                new DenseLayer(new ReluActivation(inputSize, 10)),
                new DenseLayer(new ReluActivation(10, 10) ),
                new DenseLayer(new LinearActivation(10, 1)),
        };
    }
}