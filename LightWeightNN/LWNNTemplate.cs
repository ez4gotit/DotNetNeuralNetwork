using System;
using NeuralNetwork.LightWeightNN;
namespace NeuralNetwork.Lightweight
{
    [Serializable]public class LWNeuralNetwork : ILWNeuralNetwork
    {
        private int inputSize;
        private int[] hiddenSizes;
        private int outputSize;

        private double[][] inputToHiddenWeights;
        private double[][][] hiddenToHiddenWeights;
        private double[][] hiddenToOutputWeights;

        private double[][] hiddenBiases;
        private double[] outputBiases;

        private Func<double, double> activationFunction;

        public readonly int InputSize;
        public LWNeuralNetwork()
        { }
        public LWNeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, Func<double, double> activationFunction)
        {
            InputSize = inputSize;
            this.inputSize = inputSize;
            this.hiddenSizes = hiddenSizes;
            this.outputSize = outputSize;
            this.activationFunction = activationFunction;

            int numHiddenLayers = hiddenSizes.Length;
            inputToHiddenWeights = new double[inputSize][];
            hiddenToHiddenWeights = new double[numHiddenLayers - 1][][];
            for (int i = 0; i < inputSize; i++)
            {
                inputToHiddenWeights[i] = new double[hiddenSizes[0]];
            }
            for (int i = 0; i < numHiddenLayers - 1; i++)
            {
                hiddenToHiddenWeights[i] = new double[hiddenSizes[i]][];
                for (int j = 0; j < hiddenSizes[i]; j++)
                {
                    hiddenToHiddenWeights[i][j] = new double[hiddenSizes[i + 1]];
                }
            }
            hiddenToOutputWeights = new double[hiddenSizes[numHiddenLayers - 1]][];
            for (int i = 0; i < hiddenSizes[numHiddenLayers - 1]; i++)
            {
                hiddenToOutputWeights[i] = new double[outputSize];
            }

            hiddenBiases = new double[numHiddenLayers][];
            for (int i = 0; i < numHiddenLayers; i++)
            {
                hiddenBiases[i] = new double[hiddenSizes[i]];
            }
            outputBiases = new double[outputSize];
        }

        public void BackPropagationTrain(double[][] inputs, double[][] outputs, int numEpochs, double learningRate)
        {
            int numExamples = inputs.Length;
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                for (int example = 0; example < numExamples; example++)
                {
                    // forward pass
                    double[][] hiddenActivations = new double[hiddenSizes.Length][];
                    hiddenActivations[0] = new double[hiddenSizes[0]];
                    for (int i = 0; i < hiddenSizes[0]; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < inputSize; j++)
                        {
                            sum += inputs[example][j] * inputToHiddenWeights[j][i];
                        }
                        sum += hiddenBiases[0][i];
                        hiddenActivations[0][i] = activationFunction(sum);
                    }

                    for (int i = 1; i < hiddenSizes.Length; i++)
                    {
                        hiddenActivations[i] = new double[hiddenSizes[i]];
                        for (int j = 0; j < hiddenSizes[i]; j++)
                        {
                            double sum = 0;
                            for (int k = 0; k < hiddenSizes[i - 1]; k++)
                            {
                                sum += hiddenActivations[i - 1][k] * hiddenToHiddenWeights[i - 1][k][j];
                            }
                            sum += hiddenBiases[i][j];
                            hiddenActivations[i][j] = activationFunction(sum);
                        }
                    }

                    double[] outputActivations = new double[outputSize];
                    for (int i = 0; i < outputSize; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < hiddenSizes[hiddenSizes.Length - 1]; j++)
                        {
                            sum += hiddenActivations[hiddenSizes.Length - 1][j] * hiddenToOutputWeights[j][i];
                        }
                        sum += outputBiases[i];
                        outputActivations[i] = activationFunction(sum);
                    }

                    // backward pass
                    double[] outputErrors = new double[outputSize];
                    for (int i = 0; i < outputSize; i++)
                    {
                        outputErrors[i] = outputActivations[i] - outputs[example][i];
                    }

                    double[][] hiddenErrors = new double[hiddenSizes.Length][];
                    hiddenErrors[hiddenSizes.Length - 1] = new double[hiddenSizes[hiddenSizes.Length - 1]];
                    for (int i = 0; i < hiddenSizes[hiddenSizes.Length - 1]; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < outputSize; j++)
                        {
                            sum += outputErrors[j] * hiddenToOutputWeights[i][j];
                        }
                        hiddenErrors[hiddenSizes.Length - 1][i] = sum * activationFunctionDerivative(hiddenActivations[hiddenSizes.Length - 1][i], activationFunction);
                    }

                    for (int i = hiddenSizes.Length - 2; i >= 0; i--)
                    {
                        hiddenErrors[i] = new double[hiddenSizes[i]];
                        for (int j = 0; j < hiddenSizes[i]; j++)
                        {
                            double sum = 0;
                            for (int k = 0; k < hiddenSizes[i + 1]; k++)
                            {
                                sum += hiddenErrors[i + 1][k] * hiddenToHiddenWeights[i][j][k];
                            }
                            hiddenErrors[i][j] = sum * activationFunctionDerivative(hiddenActivations[i][j], activationFunction);
                        }
                    }

                    // weight and bias updates
                    for (int i = 0; i < outputSize; i++)
                    {
                        for (int j = 0; j < hiddenSizes[hiddenSizes.Length - 1]; j++)
                        {
                            hiddenToOutputWeights[j][i] -= learningRate * outputErrors[i] * hiddenActivations[hiddenSizes.Length - 1][j];
                        }
                        outputBiases[i] -= learningRate * outputErrors[i];
                    }

                    for (int i = hiddenSizes.Length - 1; i >= 1; i--)
                    {
                        for (int j = 0; j < hiddenSizes[i]; j++)
                        {
                            for (int k = 0; k < hiddenSizes[i - 1]; k++)
                            {
                                hiddenToHiddenWeights[i - 1][k][j] -= learningRate * hiddenErrors[i][j] * hiddenActivations[i - 1][k];
                            }
                            hiddenBiases[i][j] -= learningRate * hiddenErrors[i][j];
                        }
                    }

                    for (int i = 0; i < hiddenSizes[0]; i++)
                    {
                        for (int j = 0; j < inputSize; j++)
                        {
                            inputToHiddenWeights[j][i] -= learningRate * hiddenErrors[0][i] * inputs[example][j];
                        }
                        hiddenBiases[0][i] -= learningRate * hiddenErrors[0][i];
                    }
                }
            }
        }




        public void StochasticBackPropagationTrain(double[][] inputs, double[][] outputs, int numEpochs, double learningRate)
        {
            int numExamples = inputs.Length;
            Random rand = new Random();

            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                for (int example = 0; example < numExamples; example++)
                {
                    // Forward pass
                    double[][] hiddenActivations = new double[hiddenSizes.Length][];
                    hiddenActivations[0] = new double[hiddenSizes[0]];
                    for (int i = 0; i < hiddenSizes[0]; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < inputSize; j++)
                        {
                            sum += inputs[example][j] * inputToHiddenWeights[j][i];
                        }
                        sum += hiddenBiases[0][i];
                        hiddenActivations[0][i] = activationFunction(sum);
                    }

                    for (int i = 1; i < hiddenSizes.Length; i++)
                    {
                        hiddenActivations[i] = new double[hiddenSizes[i]];
                        for (int j = 0; j < hiddenSizes[i]; j++)
                        {
                            double sum = 0;
                            for (int k = 0; k < hiddenSizes[i - 1]; k++)
                            {
                                sum += hiddenActivations[i - 1][k] * hiddenToHiddenWeights[i - 1][k][j];
                            }
                            sum += hiddenBiases[i][j];
                            hiddenActivations[i][j] = activationFunction(sum);
                        }
                    }

                    double[] outputActivations = new double[outputSize];
                    for (int i = 0; i < outputSize; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < hiddenSizes[hiddenSizes.Length - 1]; j++)
                        {
                            sum += hiddenActivations[hiddenSizes.Length - 1][j] * hiddenToOutputWeights[j][i];
                        }
                        sum += outputBiases[i];
                        outputActivations[i] = activationFunction(sum);
                    }

                    // Backward pass
                    double[] outputErrors = new double[outputSize];
                    for (int i = 0; i < outputSize; i++)
                    {
                        outputErrors[i] = outputActivations[i] - outputs[example][i];
                    }

                    double[][] hiddenErrors = new double[hiddenSizes.Length][];
                    hiddenErrors[hiddenSizes.Length - 1] = new double[hiddenSizes[hiddenSizes.Length - 1]];
                    for (int i = 0; i < hiddenSizes[hiddenSizes.Length - 1]; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < outputSize; j++)
                        {
                            sum += outputErrors[j] * hiddenToOutputWeights[i][j];
                        }
                        hiddenErrors[hiddenSizes.Length - 1][i] = sum * activationFunctionDerivative(hiddenActivations[hiddenSizes.Length - 1][i], activationFunction);
                    }

                    for (int i = hiddenSizes.Length - 2; i >= 0; i--)
                    {
                        hiddenErrors[i] = new double[hiddenSizes[i]];
                        for (int j = 0; j < hiddenSizes[i]; j++)
                        {
                            double sum = 0;
                            for (int k = 0; k < hiddenSizes[i + 1]; k++)
                            {
                                sum += hiddenErrors[i + 1][k]*hiddenToHiddenWeights[i][j][k];
                            }
                            hiddenErrors[i][j] = sum * activationFunctionDerivative(hiddenActivations[i][j], activationFunction);
                        }
                    }// Update weights and biases
                    for (int i = 0; i < outputSize; i++)
                    {
                        for (int j = 0; j < hiddenSizes[hiddenSizes.Length - 1]; j++)
                        {
                            hiddenToOutputWeights[j][i] -= learningRate * outputErrors[i] * hiddenActivations[hiddenSizes.Length - 1][j];
                        }
                        outputBiases[i] -= learningRate * outputErrors[i];
                    }

                    for (int i = hiddenSizes.Length - 1; i >= 1; i--)
                    {
                        for (int j = 0; j < hiddenSizes[i]; j++)
                        {
                            for (int k = 0; k < hiddenSizes[i - 1]; k++)
                            {
                                hiddenToHiddenWeights[i - 1][k][j] -= learningRate * hiddenErrors[i][j] * hiddenActivations[i - 1][k];
                            }
                            hiddenBiases[i][j] -= learningRate * hiddenErrors[i][j];
                        }
                    }

                    for (int i = 0; i < hiddenSizes[0]; i++)
                    {
                        for (int j = 0; j < inputSize; j++)
                        {
                            inputToHiddenWeights[j][i] -= learningRate * hiddenErrors[0][i] * inputs[example][j];
                        }
                        hiddenBiases[0][i] -= learningRate * hiddenErrors[0][i];
                    }
                }
            }
        }









        public double[] FeedForward(double[] input)
        {
            double[][] hiddenActivations = new double[hiddenSizes.Length][];
            hiddenActivations[0] = new double[hiddenSizes[0]];
            for (int i = 0; i < hiddenSizes[0]; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputSize; j++)
                {
                    sum += input[j] * inputToHiddenWeights[j][i];
                }
                sum += hiddenBiases[0][i];
                hiddenActivations[0][i] = activationFunction(sum);
            }

            for (int i = 1; i < hiddenSizes.Length; i++)
            {
                hiddenActivations[i] = new double[hiddenSizes[i]];
                for (int j = 0; j < hiddenSizes[i]; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < hiddenSizes[i - 1]; k++)
                    {
                        sum += hiddenActivations[i - 1][k] * hiddenToHiddenWeights[i - 1][k][j];
                    }
                    sum += hiddenBiases[i][j];
                    hiddenActivations[i][j] = activationFunction(sum);
                }
            }

            double[] outputActivations = new double[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < hiddenSizes[hiddenSizes.Length - 1]; j++)
                {
                    sum += hiddenActivations[hiddenSizes.Length - 1][j] * hiddenToOutputWeights[j][i];
                }
                sum += outputBiases[i];
                outputActivations[i] = activationFunction(sum);
            }

            return outputActivations;
        }
        public static double sigmoidActivationFunction(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        private double sigmoidActivationFunctionDerivative(double x)
        {
            double fx = activationFunction(x);
            return fx * (1 - fx);
        }
        private double activationFunctionDerivative(double x, Func<double, double> activationFunction)
        {
            if (activationFunction == sigmoidActivationFunction)
                return sigmoidActivationFunctionDerivative(x);
            else if (activationFunction == tanhActivationFunction) return tanhActivationFunctionDerivative(x);
            else return 0;

        }
        public static double tanhActivationFunction(double x)
        {
            return 2.0 / (1.0 + Math.Exp(-2 * x)) - 1.0;
        }  

        private double tanhActivationFunctionDerivative(double x)
        {
            double fx = tanhActivationFunction(x);
            return (1.0 - fx) * fx;
        }
    }
}
