using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork2
{


    public class NN2

    {

        public NN2(int[] _size)
        {
            size = _size;
            maxLayerSize = 0;
            for (int i = 0; i < size.Length; i++)
            {
                if (maxLayerSize < size[i]) maxLayerSize = size[i];

            }
            hidden = new double[size.Length, maxLayerSize];

        }

        public double[] input { get; set; }
        int[] size;
        int maxLayerSize;



        double[] output { get; set; }
        double[,] hidden;
        double[,] weights;

        /* public double[][] setup()
         {
             return;
         }*/

        public void connectRandomly(ref double[,] _weights, int threshold)
        {
            Random randomConn = new Random();
            Random randomWeight = new Random(1);

            for (int x = 0; x < maxLayerSize; x++) for (int y = 0; y < size.Length; y++)
                {
                    if (randomConn.Next(0, 100) <= threshold) _weights[x, y] = randomWeight.Next(-1, 1) / (double)(2 * randomWeight.Next(1, 100));
                    else weights[x, y] = 0f;

                }





        }

/*        public void Train(double learningRate, int epochs, double errorThreshold)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double errorSum = 0;

                for (int i = 0; i < dataSet.Length; i++)
                {
                    // Feed forward
                    ProcessValues(Neurons, dataSet[i].input);

                    // Calculate output layer error
                    for (int n = 0; n < Output.Length; n++)
                    {
                        double error = dataSet[i].output[n] - Output[n].value;
                        Output[n].error = error;
                        errorSum += Math.Abs(error);
                    }

                    // Calculate hidden layer errors
                    for (int layerIndex = size.Length - 2; layerIndex > 0; layerIndex--)
                    {
                        for (int neuronIndex = 0; neuronIndex < size[layerIndex]; neuronIndex++)
                        {
                            double errorSummation = 0;
                            for (int connectionIndex = 0; connectionIndex < Neurons[layerIndex + 1, 0].connections.Length; connectionIndex++)
                            {
                                Connection connection = Neurons[layerIndex + 1, connectionIndex].connections[neuronIndex];
                                errorSummation += connection.weight * Neurons[layerIndex + 1, connectionIndex].error;
                            }
                            Neurons[layerIndex, neuronIndex].error = errorSummation;
                        }
                    }

                    // Update weights and biases
                    for (int layerIndex = 1; layerIndex < size.Length; layerIndex++)
                    {
                        for (int neuronIndex = 0; neuronIndex < size[layerIndex]; neuronIndex++)
                        {
                            Neuron neuron = Neurons[layerIndex, neuronIndex];

                            // Update biases
                            neuron.bias += learningRate * neuron.error;

                            // Update weights
                            for (int connectionIndex = 0; connectionIndex < neuron.connections.Length; connectionIndex++)
                            {
                                Connection connection = neuron.connections[connectionIndex];
                                Neuron prevNeuron = Neurons[layerIndex - 1, connectionIndex];

                                connection.weight += learningRate * neuron.error * prevNeuron.value;
                            }
                        }
                    }
                }

                // Calculate average error for this epoch
                double averageError = errorSum / Output.Length;

                // Stop training if average error is below threshold
                if (averageError < errorThreshold)
                {
                    return;
                }
            }
        }*/
    }






    public static class Maths
    {
        #region ActivationFunctions

        static double SigmaDifferential(double x)
        {

            return (Math.Pow(Math.E, -x) / Math.Pow((1 + Math.Pow(Math.E, -x)), 2));
        }
        static double activationSigma(double x)
        {
            return (1 / (1 + Math.Pow(Math.E, -x)));
        }


        static double activationHyperbolic(double x)
        {
            return ((2 / (1 + Math.Pow(Math.E, -2 * x))) - 1);
            // return (Math.Pow(Math.E, 2*x)-1) / (Math.Pow(Math.E, 2 * x) + 1);
        }

        static double HyperbolicDifferential(double x)
        {
            return (4 * Math.Pow(Math.E, -2 * x) / Math.Pow(1 + Math.Pow(Math.E, -2 * x), 2));
        }

        #endregion ActivationFunctions


        #region EMS Func

         /*public double EMS(double[] awaitValue, double[] resultValue)
         {
             double error = 0;
             for (int i = 0; i < size[size.Length - 1], i++)
             {
                 error += MathF.Pow((double)(resultValue[i] - awaitValue[i]), 2);
             }
             return error / 2;
         }*/
 
        #endregion

    }

}



