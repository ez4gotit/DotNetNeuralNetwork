using System;
namespace NeuralNetwork.Lightweight
{
    class LWNNTemplate
    {
        // number of input nodes
        int inputSize;
        // number of hidden nodes
        int hiddenSize;
        // number of output nodes
        int outputSize;

        // weight matrices
        double[,] inputToHiddenWeights;
        double[,] hiddenToOutputWeights;

        // bias vectors
        double[] hiddenBias;
        double[] outputBias;

        // activation function
        Func<double, double> activationFunction;

        public LWNNTemplate(int inputSize, int hiddenSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            inputToHiddenWeights = new double[inputSize, hiddenSize];
            hiddenToOutputWeights = new double[hiddenSize, outputSize];
            hiddenBias = new double[hiddenSize];
            outputBias = new double[outputSize];

            activationFunction = x => Math.Tanh(x);
        }

        public double[] FeedForward(double[] inputs)
        {
            // calculate activations of hidden layer
            double[] hiddenActivations = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputSize; j++)
                {
                    sum += inputs[j] * inputToHiddenWeights[j, i];
                }
                sum += hiddenBias[i];
                hiddenActivations[i] = activationFunction(sum);
            }

            // calculate activations of output layer
            double[] outputActivations = new double[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < hiddenSize; j++)
                {
                    sum += hiddenActivations[j] * hiddenToOutputWeights[j, i];
                }
                sum += outputBias[i];
                outputActivations[i] = activationFunction(sum);
            }

            return outputActivations;
        }

        public void BackPropagationTrain(double[][] inputs, double[][] outputs, int numEpochs, double learningRate)
        {
            int numExamples = inputs.Length;
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                for (int example = 0; example < numExamples; example++)
                {
                    // forward pass
                    double[] hiddenActivations = new double[hiddenSize];
                    for (int i = 0; i < hiddenSize; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < inputSize; j++)
                        {
                            sum += inputs[example][j] * inputToHiddenWeights[j, i];
                        }
                        sum += hiddenBias[i];
                        hiddenActivations[i] = activationFunction(sum);
                    }

                    double[] outputActivations = new double[outputSize];
                    for (int i = 0; i < outputSize; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < hiddenSize; j++)
                        {
                            sum += hiddenActivations[j] * hiddenToOutputWeights[j, i];
                        }
                        sum += outputBias[i];
                        outputActivations[i] = activationFunction(sum);
                    }

                    // backward pass
                    double[] outputErrors = new double[outputSize];
                    for (int i = 0; i < outputSize; i++)
                    {
                        outputErrors[i] = outputs[example][i] - outputActivations[i];
                    }

                    double[] hiddenErrors = new double[hiddenSize];
                    for (int i = 0; i < hiddenSize; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < outputSize; j++)
                        {
                            sum += outputErrors[j] * hiddenToOutputWeights[i, j];
                        }
                        hiddenErrors[i] = sum * activationFunctionDerivative(hiddenActivations[i]);
                    }

                    // update weights and biases
                    for (int i = 0; i < outputSize; i++)
                    {
                        for (int j = 0; j < hiddenSize; j++)
                        {
                            hiddenToOutputWeights[j, i] += learningRate * outputErrors[i] * hiddenActivations[j];
                        }
                        outputBias[i] += learningRate * outputErrors[i];
                    }

                    for (int i = 0; i < hiddenSize; i++)
                    {
                        for (int j = 0; j < inputSize; j++)
                        {
                            inputToHiddenWeights[j, i] += learningRate * hiddenErrors[i] * inputs[example][j];
                        }
                        hiddenBias[i] += learningRate * hiddenErrors[i];
                    }
                }
            }
        }

        private double activationFunctionDerivative(double x)
        {
            // derivative of the hyperbolic tangent function
            return 1 - Math.Tanh(x) * Math.Tanh(x);
        }

    }



    public class LWNeuralNetwork
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

        public LWNeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, Func<double, double> activationFunction)
        {
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
                        hiddenErrors[hiddenSizes.Length - 1][i] = sum * activationFunctionDerivative(hiddenActivations[hiddenSizes.Length - 1][i]);
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
                            hiddenErrors[i][j] = sum * activationFunctionDerivative(hiddenActivations[i][j]);
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
        private double activationFunctionDerivative(double x)
        {
            double fx = activationFunction(x);
            return fx * (1 - fx);
        }
    }
}
