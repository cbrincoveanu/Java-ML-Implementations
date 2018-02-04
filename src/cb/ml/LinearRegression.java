package cb.ml;

import java.util.ArrayList;

public class LinearRegression {

    private static final int MAX_SIZE = 1000;
    private ArrayList<Double> xValues = new ArrayList<>();
    private ArrayList<Double> yValues = new ArrayList<>();
    private double xMean = 0;
    private double yMean = 0;
    private double b0 = 0;
    private double b1 = 0;
    private double meanSquaredError;
    private double rSquared;
    private double adjustedRSquared;

    public void addPoint(double x, double y) {
        xValues.add(x);
        yValues.add(y);
        if (xValues.size() > MAX_SIZE) {
            xValues.remove(0);
            yValues.remove(0);
        }
        xMean = 0;
        yMean = 0;
        for (int i = 0; i < xValues.size(); i++) {
            xMean += xValues.get(i);
            yMean += yValues.get(i);
        }
        xMean /= xValues.size();
        yMean /= yValues.size();
        if (xValues.size() == 1) {
            b0 = yMean;
            return;
        }
        double numerator = 0;
        double denominator = 0;
        for (int i = 0; i < xValues.size(); i++) {
            double xDiff = (xValues.get(i) - xMean);
            double yDiff = (yValues.get(i) - yMean);
            numerator += xDiff * yDiff;
            denominator += xDiff * xDiff;
        }
        b1 = numerator / denominator;
        b0 = yMean - b1 * xMean;
        meanSquaredError = 0;
        denominator = 0;
        for (int i = 0; i < xValues.size(); i++) {
            double diff = (yValues.get(i) - estimateY(xValues.get(i)));
            meanSquaredError += diff * diff;
            diff = (yValues.get(i) - yMean);
            denominator += diff * diff;
        }
        rSquared = 1 - meanSquaredError / denominator;
        adjustedRSquared = rSquared - (1 - rSquared) / (xValues.size() - 2);
        meanSquaredError = meanSquaredError / xValues.size();
    }

    public double estimateY(double x) {
        return b0 + b1 * x;
    }

    public double getMeanSquaredError() {
        return meanSquaredError;
    }

    public void summary() {
        double p = 1000;
        System.out.println("y = " + Math.round(b0*p)/p + " + " + Math.round(b1*p)/p + " * x, MSE = " + Math.round(meanSquaredError*p)/p + ", R² = " + Math.round(rSquared*p)/p + ", R²_adj = " + Math.round(adjustedRSquared*p)/p);
    }
}
