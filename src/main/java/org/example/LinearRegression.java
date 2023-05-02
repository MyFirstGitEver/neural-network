package org.example;

public class LinearRegression {
    private final PredictFunction predictor;
    private final Pair<Vector, Float>[] dataset;
    private final Vector w;
    private float b;
    LinearRegression(PredictFunction predictor, Pair<Vector, Float>[] dataset) {
        this.predictor = predictor;
        this.dataset = dataset;

        w = new Vector(dataset[0].first.size());
        b = 0;
    }


    // R-squared error
    public float cost() {
        int n = dataset.length;
        float total = 0;

        for(Pair<Vector, Float> p : dataset){
            float term = (p.second - predictor.predict(p.first, w, b));
            total += term * term;
        }

        return total / (2 * n);
    }

    public float predict(Vector x) {
        if(x.size() != w.size()){
            return Float.NaN;
        }

        return predictor.predict(x, w, b);
    }

    public void train(float learningRate, int iter) {
        int iteration = 0;

        while(Math.abs(cost()) > 0.0001 && iteration < iter) {
            Vector v = predictor.derivativeByW(w, b, dataset).scaleBy(learningRate);

            b -= learningRate * predictor.derivativeByB(w, b, dataset);
            w.subtract(v);

            iteration++;
        }
    }
}