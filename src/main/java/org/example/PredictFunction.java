package org.example;

import java.util.Arrays;
import java.util.Random;

abstract class PredictFunction {
    abstract float predict(Vector x, Vector w, float b);
    abstract  Vector derivativeByW(Vector w, float b, Pair<Vector, Float>[] dataset);

    public float derivativeByB(Vector w, float b, Pair<Vector, Float>[] dataset) {
        float total = 0;
        int datasetLength = dataset.length;

        for (Pair<Vector, Float> vectorFloatPair : dataset) {
            total += (predict(vectorFloatPair.first, w, b) - vectorFloatPair.second);
        }

        return total / datasetLength;
    }
}

class Matrix {
    private final float[][] entries;

    Matrix(Vector v, boolean columnVector) {
        if(columnVector) {
            entries = new float[v.size()][1];

            for(int i=0;i<v.size();i++) {
                entries[i][0] = v.x(i);
            }
        }
        else {
            entries = new float[1][v.size()];

            for(int i=0;i<v.size();i++) {
                entries[0][i] = v.x(i);
            }
        }
    }

    Matrix(float[]... data) {
        entries = data;
    }

    Matrix(int width, int height) {
        entries = new float[width][height];
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
                float total = 0;

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

    public Matrix scale(float scale) {
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j]  *= scale;
            }
        }

        return this;
    }

    public static Matrix transpose(Vector[] mat) {
        float[][] answer = new float[mat[0].size()][mat.length];

        for(int i=0;i<mat.length;i++){
            for(int j=0;j<mat[0].size();j++) {
                answer[j][i] = mat[i].x(j);
            }
        }

        return new Matrix(answer);
    }
}

class Vector {

    private final float[] points;
    Vector(float... points){
        this.points = points;
    }

    Vector(int size){
        points = new float[size];
    }

    Vector(int size, double value){
        points = new float[size];

        Arrays.fill(points, (float) value);
    }

    Vector(Object[] data) {
        points = new float[data.length];

        for(int i=0;i<points.length;i++){
            points[i] = ((Double)data[i]).floatValue();
        }
    }

    Vector(Vector[] twoD) throws Exception {
        // flatten this 2d matrix

        if(twoD.length != 1 && twoD[0].size() != 1) {
            throw new Exception("Can't flatten this one!");
        }
        else if(twoD.length == 1) {
            points = new float[twoD[0].size()];

            for(int i=0;i<twoD[0].size();i++) {
                points[i] = twoD[0].x(i);
            }
        }
        else {
            points = new float[twoD.length];

            for(int j=0;j<twoD.length;j++) {
                points[j] = twoD[j].x(0);
            }
        }
    }

    float x(int i){
        return points[i];
    }

    void setX(int pos, float value){
        points[pos] = value;
    }

    int size(){
        return points.length;
    }

    float dot(Vector w) {
        if(points.length != w.size()){
            return Float.NaN;
        }

        int n = points.length;
        float total = 0;
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

    Vector scaleBy(float x){
        for(int i=0;i<points.length;i++){
            points[i] *= x;
        }

        return this;
    }

    float sum(){
        float total = 0;

        for (float point : points) {
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
            points[i] = random.nextFloat();
        }
    }

    Vector divide(Vector v, float eps) throws Exception {
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
            v.setX(i, (float) Math.sqrt(points[i]));
        }

        return v;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        builder.append("(");
        for(int i=0;i<points.length;i++) {
            builder.append(points[i] + ", ");
        }

        builder.append(")");

        return builder.toString();
    }
}

class PolynomialPredictor extends PredictFunction{
    @Override
    public float predict(Vector x, Vector w, float b) {
        return x.dot(w) + b;
    }

    @Override
    public Vector derivativeByW(Vector w, float b, Pair<Vector, Float>[] dataset) {
        Vector derivative = new Vector(w.size());

        int datasetLength = dataset.length;
        int features = w.size();

        for(int i=0;i<features;i++){
            for (Pair<Vector, Float> vectorFloatPair : dataset) {
                float curr = derivative.x(i);

                curr += vectorFloatPair.first.x(i) *
                        (predict(vectorFloatPair.first, w, b) - vectorFloatPair.second);
                derivative.setX(i, curr);
            }

            derivative.setX(i, derivative.x(i) / datasetLength);
        }

        return derivative;
    }
}
