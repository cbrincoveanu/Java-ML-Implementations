package cb;

import cb.ml.DecisionTree;
import cb.ml.LinearRegression;
import cb.ml.NeuralNetwork;
import javafx.util.Pair;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main {

    public static void main(String[] args) {
        List<String> list = new ArrayList<>();
        try (Stream<String> stream = Files.lines(Paths.get("data/blood_transfusion.csv"))) {
            list = stream.collect(Collectors.toList());
        } catch (IOException e) {
            e.printStackTrace();
        }
        ArrayList<Pair<double[],double[]>> data = new ArrayList<>();
        ArrayList<Pair<double[],double[]>> test = new ArrayList<>();
        int inputLength = 0;
        for (String item : list) {
            String[] attributes = item.split(",");
            inputLength = attributes.length - 1;
            double[] x = new double[attributes.length - 1];
            for (int i = 0; i < attributes.length - 1; i++) {
                x[i] = new Integer(attributes[i]);
            }
            double[] y;
            if (attributes[attributes.length-1].equals("1")) {
                y = new double[]{1,0};
            } else {
                y = new double[]{0,1};
            }
            data.add(new Pair<>(x,y));
        }
        for (int i = 0; i < 100; i++) {
            test.add(data.remove((int)(Math.random()*data.size())));
        }
        System.out.println("Training decision tree...");
        DecisionTree dt = new DecisionTree(data, 3, 5);
        int accuracy = 0;
        for (int i = 0; i < 100; i++) {
            Pair<double[],double[]> sample = test.get(i);
            ArrayList<double[]> y = dt.getEntries(sample.getKey());
            double[] d = new double[2];
            for (double[] v : y) {
                d[0] += v[0];
                d[1] += v[1];
            }
            if (d[0] < d[1] && sample.getValue()[0] < sample.getValue()[1]) {
                accuracy++;
            } else if (d[0] > d[1] && sample.getValue()[0] > sample.getValue()[1]) {
                accuracy++;
            }
        }
        System.out.println("Decision tree accuracy: "+ accuracy + "%");
        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLength, new int[]{5},2, 1000, 1000);
        accuracy = 0;
        for (Pair<double[],double[]> sample : data) {
            neuralNetwork.addData(sample.getKey(), sample.getValue());
        }
        System.out.println("Training neural network...");
        neuralNetwork.train(5000);
        for (int i = 0; i < 100; i++) {
            Pair<double[],double[]> sample = test.get(i);
            double[] y = neuralNetwork.predict(sample.getKey());
            if (y[0] < y[1] && sample.getValue()[0] < sample.getValue()[1]) {
                accuracy++;
            } else if (y[0] > y[1] && sample.getValue()[0] > sample.getValue()[1]) {
                accuracy++;
            }
        }
        System.out.println("Neural network accuracy: "+ accuracy + "%");
    }
}
