package org.example;

public class LinearRegression {
    private final PredictFunction predictor;
    private final Pair<Vector, Double>[] dataset;
    private final Vector w;
    private double b;
    LinearRegression(PredictFunction predictor, Pair<Vector, Double>[] dataset) {
        this.predictor = predictor;
        this.dataset = dataset;

        w = new Vector(dataset[0].first.size());
        b = 0;
    }


    // R-squared error
    public double cost() {
        int n = dataset.length;
        double total = 0;

        for(Pair<Vector, Double> p : dataset){
            double term = (p.second - predictor.predict(p.first, w, b));
            total += term * term;
        }

        return total / (2 * n);
    }

    public double predict(Vector x) {
        if(x.size() != w.size()){
            return Double.NaN;
        }

        return predictor.predict(x, w, b);
    }

    public void train(double learningRate, int iter) {
        int iteration = 0;

        while(Math.abs(cost()) > 0.0001 && iteration < iter) {
            Vector v = predictor.derivativeByW(w, b, dataset).scaleBy(learningRate);

            b -= learningRate * predictor.derivativeByB(w, b, dataset);
            w.subtract(v);

            iteration++;
        }
    }
}