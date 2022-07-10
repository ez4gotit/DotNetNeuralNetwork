using System;
using NeuralNetwork.Classes;
namespace NeuralNetwork
{
    class Program
    {
        static int[] size = { 3, 5, 5, 1 };
        static double[] input = {1, 1, 1 };
        static NeuralNetworkTemplate neuralNetwork = new Classes.NeuralNetworkTemplate(size);
        static void Main(string[] args)
        {
            neuralNetwork.XMLPath = @"C:\Users\user\Documents\GitHub\NeuralNetwork\XMLFile1.xml";
            neuralNetwork.Setup();
            
            Console.WriteLine(neuralNetwork.ProcessValues(neuralNetwork.Neurons, input)[0]);
        }
    }
}
