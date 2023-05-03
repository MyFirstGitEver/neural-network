package org.example;

import java.util.Arrays;
import java.util.Random;

abstract class PredictFunction {
    abstract double predict(Vector x, Vector w, double b);
    abstract  Vector derivativeByW(Vector w, double b, Pair<Vector, Double>[] dataset);

    public double derivativeByB(Vector w, double b, Pair<Vector, Double>[] dataset) {
        double total = 0;
        int datasetLength = dataset.length;

        for (Pair<Vector, Double> vectordoublePair : dataset) {
            total += (predict(vectordoublePair.first, w, b) - vectordoublePair.second);
        }

        return total / datasetLength;
    }
}

class Matrix {
    private final double[][] entries;

    Matrix(Vector v, boolean columnVector) {
        if(columnVector) {
            entries = new double[v.size()][1];

            for(int i=0;i<v.size();i++) {
                entries[i][0] = v.x(i);
            }
        }
        else {
            entries = new double[1][v.size()];

            for(int i=0;i<v.size();i++) {
                entries[0][i] = v.x(i);
            }
        }
    }

    Matrix(double[]... data) {
        entries = data;
    }

    Matrix(int width, int height) {
        entries = new double[width][height];
    }

    public Vector[] mul(Matrix mat) throws Exception {
        Pair<Integer, Integer> shape = mat.shape();

        if(entries[0].length != shape.first) {
            throw new Exception("Can't multiply these two matrices");
        }

        int common = shape.first;
        Vector[] newMat = new Vector[entries.length];

        for(int i=0;i<entries.length;i++) {
            newMat[i] = new Vector(shape.second);

            for(int j=0;j<shape.second;j++) {
                double total = 0;

                for(int k=0;k<common;k++) {
                    total += entries[i][k] * mat.entries[k][j];
                }

                newMat[i].setX(j, total);
            }
        }

        return newMat;
    }

    public Pair<Integer, Integer> shape() {
        return new Pair<>(entries.length, entries[0].length);
    }

    public boolean identical(Vector[] mat) {
        if(entries.length != mat.length || entries[0].length != mat[0].size()) {
            return false;
        }

        for(int i=0;i<mat.length;i++) {
            for(int j=0;j<mat[0].size();j++) {
                if(mat[i].x(j) != entries[i][j]) {
                    return false;
                }
            }
        }

        return true;
    }

    public void reset() {
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] = 0;
            }
        }
    }

    public void add(Vector[] mat) throws Exception {
        if(entries.length != mat.length || entries[0].length != mat[0].size()) {
            throw new Exception("Can't add these two matrices");
        }
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] += mat[i].x(j);
            }
        }
    }

    public void add(Matrix matrix) throws Exception {
        Pair<Integer, Integer> shape = matrix.shape();

        if(entries.length != shape.first || entries[0].length != shape.second) {
            throw new Exception("Can't add these two matrices");
        }
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] += matrix.entries[i][j];
            }
        }
    }

    public Matrix square() {
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] *= entries[i][j];
            }
        }

        return this;
    }

    public Matrix copy() {
        double[][] newMat = new double[entries.length][entries[0].length];

        for(int i=0;i<newMat.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                newMat[i][j] = entries[i][j];
            }
        }

        return new Matrix(newMat);
    }

    public Vector[] vectorize(boolean byRow) {
        if(byRow) {
            Vector[] vectors = new Vector[entries.length];

            for(int i=0;i<entries.length;i++) {
                vectors[i] = new Vector(entries[i]);
            }

            return vectors;
        }

        Vector[] vectors = new Vector[entries[0].length];

        for(int i=0;i<entries[0].length;i++) {
            vectors[i] = new Vector(entries.length);

            for(int j=0;j<entries.length;j++) {
                vectors[i].setX(j, entries[j][i]);
            }
        }

        return vectors;
    }

    public Matrix scale(double scale) {
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j]  *= scale;
            }
        }

        return this;
    }

    public static Matrix transpose(Vector[] mat) {
        double[][] answer = new double[mat[0].size()][mat.length];

        for(int i=0;i<mat.length;i++){
            for(int j=0;j<mat[0].size();j++) {
                answer[j][i] = mat[i].x(j);
            }
        }

        return new Matrix(answer);
    }
}

class Vector {

    private double[] points;
    Vector(double... points){
        this.points = points;
    }

    Vector(int size){
        points = new double[size];
    }

    Vector(int size, double value){
        points = new double[size];

        Arrays.fill(points, (double) value);
    }

    Vector(Object[] data) {
        points = new double[data.length];

        for(int i=0;i<points.length;i++){
            points[i] = (Double) data[i];
        }
    }

    Vector(String data) {
        String[] list = data.split("\t");

        points = new double[list.length];

        for(int i=0;i<points.length;i++) {
            points[i] = Double.parseDouble(list[i]);
        }
    }

    Vector(Vector[] twoD) throws Exception {
        // flatten this 2d matrix

        if(twoD.length != 1 && twoD[0].size() != 1) {
            throw new Exception("Can't flatten this one!");
        }
        else if(twoD.length == 1) {
            points = new double[twoD[0].size()];

            for(int i=0;i<twoD[0].size();i++) {
                points[i] = twoD[0].x(i);
            }
        }
        else {
            points = new double[twoD.length];

            for(int j=0;j<twoD.length;j++) {
                points[j] = twoD[j].x(0);
            }
        }
    }

    double x(int i){
        return points[i];
    }

    void setX(int pos, double value){
        points[pos] = value;
    }

    int size(){
        return points.length;
    }

    double dot(Vector w) {
        if(points.length != w.size()){
            return Double.NaN;
        }

        int n = points.length;
        double total = 0;
        for(int i=0;i<n;i++){
            total += points[i] * w.x(i);
        }

        return total;
    }

    void subtract(Vector v){
        for(int i=0;i<points.length;i++){
            points[i] -= v.x(i);
        }
    }

    Vector scaleBy(double x){
        for(int i=0;i<points.length;i++){
            points[i] *= x;
        }

        return this;
    }

    Vector copy() {
        double[] newVec = new double[points.length];

        return new Vector(Arrays.copyOfRange(newVec, 0, newVec.length));
    }

    double sum(){
        double total = 0;

        for (double point : points) {
            total += point;
        }

        return total;
    }

    Vector hadamard(Vector v) {
        Vector answer = new Vector(v.size());

        for(int i=0;i<v.size();i++) {
            answer.setX(i, points[i] * v.x(i));
        }

        return answer;
    }

    void reset() {
        Arrays.fill(points, 0);
    }

    void add(Vector v){
        for(int i=0;i<points.length;i++){
            points[i] += v.x(i);
        }
    }

    void randomise() {
        Random random = new Random();

        for(int i=0;i<points.length;i++) {
            points[i] = random.nextDouble() + 0.0001f;
        }
    }

    Vector divide(Vector v, double eps) throws Exception {
        Vector answer = new Vector(v.size());

        if(v.size() != points.length) {
            throw new Exception("Cam't divide");
        }

        for(int i=0;i<answer.size();i++) {
            answer.setX(i, points[i] / (v.x(i) + eps));
        }

        return answer;
    }

    Vector square() {
        for(int i=0;i<points.length;i++) {
            points[i] *= points[i];
        }

        return this;
    }

    Vector sqrt() {
        Vector v = new Vector(points.length);

        for(int i=0;i<v.size();i++) {
            v.setX(i, Math.sqrt(points[i]));
        }

        return v;
    }

    public void concat(Vector v) {
        double[] newVec = new double[size() + v.size()];

        System.arraycopy(points, 0, newVec, 0, points.length);

        for (int i = points.length; i < newVec.length; i++) {
            newVec[i] = v.x(i - points.length);
        }

        points = newVec;
    }

    public void normalise() {
        float length = 0.0f;

        for (int i = 0; i < points.length; i++) {
            length += points[i] * points[i];
        }

        length = (float) Math.sqrt(length);

        if (length == 0) {
            return;
        }

        for (int i = 0; i < points.length; i++) {
            points[i] /= length;
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
;
        for(int i=0;i<points.length;i++) {
            builder.append(points[i] + "\t");
        }

        return builder.toString();
    }
}

class PolynomialPredictor extends PredictFunction{
    @Override
    public double predict(Vector x, Vector w, double b) {
        return x.dot(w) + b;
    }

    @Override
    public Vector derivativeByW(Vector w, double b, Pair<Vector, Double>[] dataset) {
        Vector derivative = new Vector(w.size());

        int datasetLength = dataset.length;
        int features = w.size();

        for(int i=0;i<features;i++){
            for (Pair<Vector, Double> vectordoublePair : dataset) {
                double curr = derivative.x(i);

                curr += vectordoublePair.first.x(i) *
                        (predict(vectordoublePair.first, w, b) - vectordoublePair.second);
                derivative.setX(i, curr);
            }

            derivative.setX(i, derivative.x(i) / datasetLength);
        }

        return derivative;
    }
}
