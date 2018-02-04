package cb.ml;

import java.util.ArrayList;

public class NeuralNetwork {
    private static final double LEARNING_RATE = 0.25;
    private int inputLength;
    private int outputLength;
    private int maxSize;
    private int batchSize;
    private ArrayList<double[]> X;
    private ArrayList<double[]> Y;
    private double[][][] syn;

    public NeuralNetwork(int inputLength, int[] hiddenLayerLength, int outputLength, int maxSize, int batchSize) {
        this.inputLength = inputLength;
        this.outputLength = outputLength;
        this.maxSize = maxSize;
        this.batchSize = batchSize;
        X = new ArrayList<>();
        Y = new ArrayList<>();
        syn = new double[hiddenLayerLength.length+1][][];
        syn[0] = new double[inputLength][hiddenLayerLength[0]];
        for (int i = 1; i < hiddenLayerLength.length; i++) {
            syn[i] = new double[hiddenLayerLength[i-1]][hiddenLayerLength[i]];
        }
        syn[hiddenLayerLength.length] = new double[hiddenLayerLength[hiddenLayerLength.length-1]][outputLength];
        randomizeWeights();
    }

    private void randomizeWeights() {
        for (int i = 0; i < syn.length; i++) {
            for (int j = 0; j < syn[i].length; j++) {
                for (int k = 0; k < syn[i][j].length; k++) {
                    syn[i][j][k] = (2*Math.random()-1);
                }
            }
        }
    }

    public void addData(double[] input, double[] output) {
        if (size() >= maxSize) {
            int i = (int)(Math.random()*maxSize/2);
            X.remove(i);
            Y.remove(i);
        }
        X.add(input);
        Y.add(output);
    }

    public double[] predict(double[] input) {
        double[][][] l = new double[syn.length+1][][];
        l[0] = new double[1][inputLength];
        for (int j = 0; j < inputLength; j++) {
            l[0][0][j] = input[j];
        }
        for (int i = 1; i < syn.length + 1; i++) {
            l[i] = nonLinear(dot(l[i-1], syn[i-1]), false);
        }
        return l[syn.length][0];
    }

    public void train(int iterations) {
        double[][] currentX;
        double[][] currentY;
        if (X.size() <= batchSize) {
            currentX = new double[X.size()][inputLength];
            currentY = new double[X.size()][outputLength];
            for (int i = 0; i < X.size(); i++) {
                for (int j = 0; j < X.get(i).length; j++) {
                    currentX[i][j] = X.get(i)[j];
                }
                for (int j = 0; j < Y.get(i).length; j++) {
                    currentY[i][j] = Y.get(i)[j];
                }
            }
        } else {
            currentX = new double[batchSize][inputLength];
            currentY = new double[batchSize][outputLength];
            for (int i = 0; i < batchSize; i++) {
                int index = (int) (Math.random()*X.size());
                for (int j = 0; j < X.get(i).length; j++) {
                    currentX[i][j] = X.get(index)[j];
                    currentY[i][j] = Y.get(index)[j];
                }
            }
        }
        for (int iteration = 0; iteration < iterations; iteration++) {
            double[][][] l = new double[syn.length + 1][][];
            l[0] = new double[currentX.length][inputLength];
            for (int i = 0; i < currentX.length; i++) {
                for (int j = 0; j < inputLength; j++) {
                    l[0][i][j] = currentX[i][j];
                }
            }
            for (int i = 1; i < syn.length + 1; i++) {
                l[i] = nonLinear(dot(l[i-1], syn[i-1]), false);
            }
            double[][][] l_error = new double[syn.length + 1][][];
            double[][][] l_delta = new double[syn.length + 1][][];
            l_error[syn.length] = minus(currentY, l[syn.length]);
            if (iteration % 1000 == 0) {
                double meanError = 0;
                for (int i = 0; i < l_error[syn.length].length; i++) {
                    for (int j = 0; j < l_error[syn.length][0].length; j++) {
                        meanError += Math.abs(l_error[syn.length][i][j]);
                    }
                }
                meanError = meanError / (l_error[syn.length].length * l_error[syn.length][0].length);
                System.out.println("Error: "+ meanError + ", size = "+ size());
            }
            l_delta[syn.length] = multiply(l_error[syn.length], nonLinear(l[syn.length], true));
            for (int i = syn.length-1; i > 0; i--) {
                l_error[i] = dot(l_delta[i+1], transpose(syn[i]));
                l_delta[i] = multiply(l_error[i], nonLinear(l[i], true));
            }
            for (int i = syn.length-1; i >= 0; i--) {
                syn[i] = plus(syn[i], dot(transpose(l[i]), l_delta[i+1]));
            }
        }
    }

    public int size() {
        return X.size();
    }

    private static double[][] transpose(double[][] A) {
        double[][] B = new double[A[0].length][A.length];
        for (int i = 0; i < B.length; i++) {
            for (int j = 0; j < B[0].length; j++) {
                B[i][j] =  A[j][i];
            }
        }
        return B;
    }

    private static double[][] plus(double[][] A, double[][] B) {
        double[][] C = new double[A.length][A[0].length];
        for (int i = 0; i < C.length; i++) {
            for (int j = 0; j < C[0].length; j++) {
                C[i][j] =  A[i][j] + B[i][j];
            }
        }
        return C;
    }

    private static double[][] minus(double[][] A, double[][] B) {
        double[][] C = new double[A.length][A[0].length];
        for (int i = 0; i < C.length; i++) {
            for (int j = 0; j < C[0].length; j++) {
                C[i][j] =  A[i][j] - B[i][j];
            }
        }
        return C;
    }

    private static double[][] multiply(double[][] A, double[][] B) {
        double[][] C = new double[A.length][A[0].length];
        for (int i = 0; i < C.length; i++) {
            for (int j = 0; j < C[0].length; j++) {
                C[i][j] =  A[i][j] * B[i][j];
            }
        }
        return C;
    }

    private static double[][] dot(double[][] A, double[][] B) {
        double[][] C = new double[A.length][B[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < B[0].length; j++) {
                C[i][j] = 0.00000;
                for (int k = 0; k < A[0].length; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    private static double nonLinear(double x, boolean derivate) {
        if (derivate) {
            return LEARNING_RATE*x*(1-x);
        } else {
            return 1 / (1 + Math.exp(-x));
        }
        /*if (derivate) {
            return LEARNING_RATE*(1 - Math.pow(nonLinear(x, false), 2));
        } else {
            return Math.tanh(x);
        }*/
    }

    private static double[][] nonLinear(double[][] x, boolean derivate) {
        double[][] y = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                y[i][j] = nonLinear(x[i][j], derivate);
            }
        }
        return y;
    }

    public static void main(String[] args) {
        //Test XOR
        NeuralNetwork neuralNetwork = new NeuralNetwork(2, new int[]{10}, 1, 100, 100);
        neuralNetwork.addData(new double[]{0,0}, new double[]{0});
        neuralNetwork.addData(new double[]{1,0}, new double[]{1});
        neuralNetwork.addData(new double[]{0,1}, new double[]{1});
        neuralNetwork.addData(new double[]{1,1}, new double[]{0});
        neuralNetwork.train(10000);
        System.out.println(neuralNetwork.predict(new double[]{1,0})[0]);
    }
}
