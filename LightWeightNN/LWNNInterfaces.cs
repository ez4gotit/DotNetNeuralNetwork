using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.LightWeightNN
{
    public interface ILWNeuralNetwork
    {
        void BackPropagationTrain(double[][] inputs, double[][] outputs, int numEpochs, double learningRate);
        double[] FeedForward(double[] input);
    }
}
