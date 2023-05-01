package org.example;

public class Pair<X, Y>{
    public X first;
    public Y second;

    Pair(){

    }

    Pair(X x, Y y){
        this.first = x;
        this.second = y;
    }
}