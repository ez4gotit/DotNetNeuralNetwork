using System;
using NeuralNetwork.Classes;
using NeuralNetwork.Lightweight;

class Program
{
    static int[] hiddenSizes = { 5 };

        static void Main(string[] args)
        {
        double[][] trainingInputs = new double[3][];
    double[][] trainingOutputs = new double[3][];
        trainingInputs[0] = new double[] { 0.1, 0.2, 0.3 };
        trainingInputs[1] = new double[] { 0.3, 0.4, 0.5 };
        trainingInputs[2] = new double[] { 0.7, 0.8, 0.9 };
        trainingOutputs[0] = new double[]{ 0.4};
        trainingOutputs[1] = new double[] { 0.6};
        trainingOutputs[2] = new double[] { 1};
        LWNeuralNetwork network = new LWNeuralNetwork(3,hiddenSizes,1,LWNeuralNetwork.sigmoidActivationFunction);
        network.BackPropagationTrain(trainingInputs,trainingOutputs,1000000, 0.01);
        double[] newInput = Array.ConvertAll(Console.ReadLine().Split(' '),Double.Parse);
        Console.WriteLine(network.FeedForward(newInput)[0]);
        }
    }

