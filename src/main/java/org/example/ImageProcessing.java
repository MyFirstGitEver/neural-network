package org.example;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageProcessing {
    private final Vector[][] pixels;

    private final static double[][] sobelX = {
            {1.0f, 0.0f, -1.0f},
            {2.0f, 0.0f, -2.0f},
            {1.0f, 0.0f, -1.0f},
    };

    private final static double[][] sobelY = {
            {1.0f, 2.0f, 1.0f},
            {0.0f, 0.0f, 0.0f},
            {-1.0f, -2.0f, -1.0f},
    };
    public ImageProcessing(String path) throws IOException {
        BufferedImage image = ImageIO.read(new File(path));

        pixels = new Vector[image.getWidth()][image.getHeight()];

        for(int i=0;i<image.getWidth();i++){
            for(int j=0;j<image.getHeight();j++){
                Color color = new Color(image.getRGB(i, j));

                pixels[i][j] = new Vector(color.getRed(), color.getGreen(), color.getBlue());
            }
        }
    }
    public Vector hog(int cellSize) {
        int m = pixels.length, n = pixels[0].length;

        Vector[][] hists = new Vector[m / cellSize][n / cellSize];

        for(int i=0;i<=m-cellSize;i+=cellSize) {
            for(int j=0;j<=n-cellSize;j+=cellSize) {
                hists[i / cellSize][j / cellSize] = hist(
                        cropPixels(i, j, i + cellSize - 1, j + cellSize - 1),
                        cellSize);
            }
        }

        Vector hogVector = new Vector(0);

        for(int i=0;i<hists.length - 1;i++){
            for(int j=0;j<hists[0].length - 1;j++){
                Vector thirtySixVec = new Vector(0);
                thirtySixVec.concat(hists[i][j]);
                thirtySixVec.concat(hists[i + 1][j]);
                thirtySixVec.concat(hists[i][j + 1]);
                thirtySixVec.concat(hists[i + 1][j + 1]);

                thirtySixVec.normalise();

                hogVector.concat(thirtySixVec);
            }
        }

        return hogVector;
    }

    private Vector[][] cropPixels(int left, int top, int right, int bottom){
        Vector[][] newPixels = new Vector[right - left + 1][bottom - top + 1];

        for(int x=left;x<=right;x++){
            for(int y=top;y<=bottom;y++){
                newPixels[x - left][y - top] = pixels[x][y];
            }
        }

        return newPixels;
    }

    private Vector hist(Vector[][] cell, int celSize){
        Vector[][] magnitude = Transformation.convolve3Total(cell, sobelX, sobelY, true); // 64 x 64
        double[][] gradient = Transformation.gradient(cell, sobelX, sobelY); // 64 x 64

        // 9 nines
        // bin1: 0 -> 160
        // bin_x = i * (20)
        Vector hist = new Vector(9);
        for(int i=0;i < celSize;i++){
            for(int j=0;j < celSize;j++){
                double bin = gradient[i][j] / 20.0f;

                double lower = Math.floor(bin);
                double upper = Math.ceil(bin);

                if(Math.abs(lower - upper) < 0.000001){
                    if(lower == 9){
                        lower = 0;
                    }
                    hist.setX((int) lower, magnitude[i][j].x(0) + hist.x((int) lower)); // add up the magnitude
                }
                else{
                    double distFromLower = gradient[i][j] - lower * 20;
                    double distFromUpper = upper * 20 - gradient[i][j];

                    hist.setX((int) lower, (magnitude[i][j].x(0) * distFromUpper / 20 +
                            hist.x((int) lower)));

                    if(upper < 9){
                        hist.setX((int) upper, (magnitude[i][j].x(0) * distFromLower / 20 +
                                hist.x((int) upper)));
                    }
                    else{
                        hist.setX(0, (magnitude[i][j].x(0) * distFromLower / 20 +
                                hist.x(0)));
                    }
                }
            }
        }

        return hist;
    }
}
