package org.example;

class Transformation {
    public static Vector[][] convolve3Total(Vector[][] pixels, double[][] kernelX, double[][] kernelY, boolean keep){
        int m = pixels.length, n = pixels[0].length;

        Vector[][] newPixels = new Vector[m][n];

        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                double intensityX = convolve(pixels, i, j, kernelX);
                double intensityY = convolve(pixels, i, j, kernelY);

                double totalIntensity = Math.round(Math.sqrt((intensityX * intensityX) + (intensityY * intensityY)));

                if(!keep){
                    totalIntensity = Math.min(totalIntensity, 255);
                }

                newPixels[i][j] = new Vector(
                        totalIntensity,
                        totalIntensity,
                        totalIntensity);
            }
        }

        return newPixels;
    }

    public static double[][] gradient(Vector[][] pixels, double[][] kernelX, double[][] kernelY){
        int m = pixels.length, n = pixels[0].length;

        double[][] newPixels = new double[m][n];

        for(int i=0;i<m - 2;i++){
            for(int j=0;j<n - 2;j++){
                double intensityX = convolve(pixels, i, j, kernelX);
                double intensityY = convolve(pixels, i, j, kernelY);

                if(Math.abs(intensityX) < 0.00001f){
                    newPixels[i][j] = 0.0f;
                }
                else{
                    // map to (0; 180)
                    newPixels[i][j] = Math.round(Math.toDegrees(Math.atan(intensityY / intensityX)) + 90);

                    if(newPixels[i][j] > 160){
                        int t = 3;
                    }
                }
            }
        }

        return newPixels;
    }

    private static double convolve(Vector[][] pixels, int i, int j, double[][] kernel){
        int n = kernel.length;

        double answer = 0;
        for(int x=i;x<i + n;x++) {
            for(int y=j;y<j + n;y++) {
                double term = 0.0f;
                if(x >= 0 && y >= 0 && x < pixels.length && y < pixels[0].length) {
                    term = pixels[x][y].x(0);
                }

                answer += term * kernel[x - i][y - j];
            }
        }

        return answer;
    }
}


