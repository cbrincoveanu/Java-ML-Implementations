package cb.ml;

import javafx.util.Pair;

import java.util.ArrayList;

public class DecisionTree {
    private ArrayList<Pair<double[],double[]>> entries = null;
    private int featureIndex;
    private double threshold;
    private DecisionTree left;
    private DecisionTree right;

    public DecisionTree(ArrayList<Pair<double[], double[]>> entries, int SPLIT_SIZE, int MAX_DEPTH) {
        if (entries.size() > SPLIT_SIZE && MAX_DEPTH > 0) {
            int bestIndex = 0;
            double bestThreshold = 0;
            double bestInformationGain = 0;
            int dimensions = entries.get(0).getKey().length;
            ArrayList<double[]> current = new ArrayList<>();
            for (Pair<double[], double[]> entry : entries) {
                current.add(entry.getValue());
            }
            double currentEntropy = getEntropy(current);
            for (int i = 0; i < dimensions; i++) {
                double min = Double.POSITIVE_INFINITY;
                double max = Double.NEGATIVE_INFINITY;
                for (Pair<double[], double[]> entry : entries) {
                    min = Math.min(entry.getKey()[i], min);
                    max = Math.max(entry.getKey()[i], max);
                }
                for (int iteration = 0; iteration < 10; iteration++) {
                    double threshold = min + Math.random()*(max-min);
                    ArrayList<double[]> left = new ArrayList<>();
                    ArrayList<double[]> right = new ArrayList<>();
                    for (Pair<double[], double[]> entry : entries) {
                        if (entry.getKey()[i] < threshold) {
                            left.add(entry.getValue());
                        } else {
                            right.add(entry.getValue());
                        }
                    }
                    double informationGain = entries.size()*currentEntropy - left.size()*getEntropy(left) - right.size()*getEntropy(right);
                    if (informationGain > bestInformationGain) {
                        bestInformationGain = informationGain;
                        bestIndex = i;
                        bestThreshold = threshold;
                    }
                }
            }
            if (bestInformationGain > 0) {
                featureIndex = bestIndex;
                threshold = bestThreshold;
                ArrayList<Pair<double[], double[]>> leftEntries = new ArrayList<>();
                ArrayList<Pair<double[], double[]>> rightEntries = new ArrayList<>();
                for (Pair<double[], double[]> entry : entries) {
                    if (entry.getKey()[featureIndex] < threshold) {
                        leftEntries.add(entry);
                    } else {
                        rightEntries.add(entry);
                    }
                }
                left = new DecisionTree(leftEntries, SPLIT_SIZE, MAX_DEPTH-1);
                right = new DecisionTree(rightEntries, SPLIT_SIZE, MAX_DEPTH-1);
            } else {
                this.entries = new ArrayList<>();
                this.entries.addAll(entries);
            }
        } else {
            this.entries = new ArrayList<>();
            this.entries.addAll(entries);
        }
    }

    public ArrayList<double[]> getEntries(double[] features) {
        if (this.entries != null) {
            ArrayList<double[]> ret = new ArrayList<>();
            for (Pair<double[],double[]> entry : entries) {
                ret.add(entry.getValue());
            }
            return ret;
        } else if (features[featureIndex] < threshold) {
            System.out.println("f["+featureIndex+"] = "+ features[featureIndex]+ " < " +threshold);
            return left.getEntries(features);
        } else {
            System.out.println("f["+featureIndex+"] = "+ features[featureIndex]+ " > " +threshold);
            return right.getEntries(features);
        }
    }

    public static double getEntropy(double[] p) {
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            if (p[i] > 0) {
                sum += p[i] * Math.log(p[i]);
            }
        }
        return -sum;
    }

    public static double getEntropy(ArrayList<double[]> y) {
        double[] distribution = new double[y.get(0).length];
        for (double[] v : y) {
            for (int i = 0; i < distribution.length; i++) {
                distribution[i] += v[i];
            }
        }
        double sum = 0;
        for (int i = 0; i < distribution.length; i++) {
            sum += distribution[i];
        }
        for (int i = 0; i < distribution.length; i++) {
            distribution[i] /= sum;
        }
        return getEntropy(distribution);
    }

    public static void main(String[] args) {
        ArrayList<Pair<double[], double[]>> entries = new ArrayList<>();
        for (double i = 0; i < 8; i += 0.1) {
            entries.add(new Pair<>(new double[] {
                    i,
                    Math.random()
            }, new double[] {
                    i > 3 ? 1 : 0,
                    i > 3 ? 0 : 1
            }));
        }
        DecisionTree dt = new DecisionTree(entries, 1, 3);
        ArrayList<double[]> result = dt.getEntries(new double[] {
                8,
                Math.random()
        });
        for (double[] entry : result) {
            for (int i = 0; i < entry.length; i++) {
                System.out.print(entry[i] + " ");
            }
            System.out.println();
        }
    }
}
