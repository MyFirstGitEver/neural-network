package org.example;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

public class MatrixTesting {
    public static Stream<Arguments> cases() {
        return Stream.of(
                Arguments.of(new Vector(5, 3, -3, 14), new Vector(5, 1, -3, 9)),
                Arguments.of(new Vector(3, 10, -3, 14), new Vector(5, 1, -3, 15)),
                Arguments.of(new Vector(5, 13, 30, -4), new Vector(5, 1, -3, 9)),
                Arguments.of(new Vector(5, 13, 0, 0), new Vector(16, 1, 3, 9))
        );
    }

    public static Stream<Arguments> cases2() {
        double[][] first1 = {
                {1, 2, 3},
                {3, 4, 5},
                {-3, -6, -7}
        };

        double[][] first2 = {
                {1, 6},
                {-3, 10},
                {3, -3}
        };

        double[][] answer = {
                {4, 17},
                {6, 43},
                {-6, -57}
        };

        double[][] second1 = {
                {-5, 10},
                {10, -8},
                {1, 0},
                {15, -9},
                {10, 19}
        };

        double[][] second2 = {
                {5, 1, 1},
                {6, 2, 15},
        };

        double[][] answer2 = {
                {35, 15, 145},
                {2, -6, -110},
                {5, 1, 1},
                {21, -3, -120},
                {164, 48, 295}
        };

        return Stream.of(
                Arguments.of(new Matrix(first1), new Matrix(first2), new Matrix(answer)),
                Arguments.of(new Matrix(second1), new Matrix(second2), new Matrix(answer2))
        );
    }

    @ParameterizedTest
    @MethodSource("cases")
    public void hadamard(Vector a, Vector b) {
        Vector answer = a.hadamard(b);

        for(int i=0;i<a.size();i++) {
            Assertions.assertEquals(a.x(i) * b.x(i), answer.x(i));
        }
    }

    @ParameterizedTest
    @MethodSource("cases2")
    public void matrixMultiplicationTest(Matrix m1, Matrix m2, Matrix answer) throws Exception {
        Assertions.assertTrue(answer.identical(m1.mul(m2)));
    }
}
