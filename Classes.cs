using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Xml.Serialization;

namespace NeuralNetwork.Classes 
{
    public enum LayerType{Input,Output, Hiden }



   
    public struct Dataset
    {
        public double[] input;
        public double[] output;
    
    }


    struct Vector2
    {
        public double x;
        public double y;


        double length()
        {
            return Math.Pow((Math.Pow(x, 2) + Math.Pow(y, 2)), .5);
        }
    }

    struct Vector3
    {
        public double x;
        public double y;
        public double z;

        double length()
        {
            return Math.Pow((Math.Pow(x, 2) + Math.Pow(y, 2) + Math.Pow(z, 2)), .5);
        }
        void convert(Vector2 vector2)
        {
            x = vector2.x;
            y = vector2.y;
            z = 0;
        }
        void zero()
        {
            x = 0;
            y = 0;
            z = 0;
        }
    }

     

    public struct Connection  //Synapse
    {
        /*public double biasVal;*/
        public int link; // link to array of previous layer
        public double weight;
    }





    public class NeuralNetworkTemplate
    {




        NeuronLayer[] layers;



        public Neuron[] Output { get; private set; }
        public static int[] size;
        public Dataset[] dataSet;

        Random random = new Random();
        public Neuron[,] Neurons;

        public string XMLPath;




        static int maxLayerSize = 0;
        #region NeuronLayer

        public struct NeuronLayer
        {
            public NeuralNetworkTemplate.Neuron[] neuron;
            
            public static NeuronLayer[] toNeuronLayers(NeuralNetworkTemplate.Neuron[,] neurons)
            {
                NeuronLayer[] neuronLayers = new NeuronLayer[size.Length];
                
                for (int i = 0; i < size.Length; i++)
                {
                    if (size[i] > maxLayerSize) maxLayerSize = size[i];
                }

                for (int i = 0; i < neuronLayers.Length; i++)
                {

                    neuronLayers[i].neuron = new Neuron[maxLayerSize];
                }
                for (int i = 0; i < neuronLayers.Length; i++)
                {

                    for (int n = 0; n < maxLayerSize; n++)
                        neuronLayers[i].neuron[n] = neurons[i, n];
                }
                return neuronLayers;
            }

            public static Neuron[,] toNeurons(NeuronLayer[] _layers)
            {
                Neuron[,] _neurons = new Neuron[size.Length, maxLayerSize];

                for(int l =0; l< size.Length; l++)
                {
                    for (int n = 0; n < maxLayerSize; n++)
                    {
                        _neurons[l,n] = _layers[l].neuron[n];
                    }
                }
                return _neurons;
            }

            public static void WriteXML(string path, NeuronLayer[] neuronLayers)
            {
          
                Stream stream = File.OpenWrite(path);
                XmlSerializer serializer = new XmlSerializer(typeof(NeuronLayer[]));
                  serializer.Serialize(stream, neuronLayers);
                stream.Close();
            }
            public static NeuronLayer[] ReadXML(string path)
            {
                NeuronLayer[] _layers;
                Stream stream = File.OpenRead(path);
                XmlSerializer serializer = new XmlSerializer(typeof(NeuronLayer[]));
                _layers = (NeuronLayer[])serializer.Deserialize(stream);
                stream.Close();
                return _layers;
                
            }
        }



        #endregion















        public NeuralNetworkTemplate(int[] _size)
        {
            if (_size.Length >= 3) size = _size;
            else
            {
                Debug.WriteLine("<<< catch : int[] _size : Network might consist of three or more layers");
                Console.WriteLine("<<< catch : int[] _size : Network might consist of three or more layers");
            }
            for(int i = 0; i< size.Length; i++)
            {
                if (size[i] > maxLayerSize) maxLayerSize = size[i];
            }
        }





        public void SaveToXML(string path)
        {

            layers = NeuronLayer.toNeuronLayers(Neurons);
            NeuronLayer.WriteXML(path, layers);
            
        }

        public void LoadFromXML(string path)
        {
            Neurons = NeuronLayer.toNeurons(NeuronLayer.ReadXML(path));
        }
        
        public void Setup()
        {
            Output = new Neuron[size[size.Length - 1]];
            int maxVal = 0;
            for (int i = 0; i < size.Length; i++)
            {
                if (maxVal <= size[i]) maxVal = size[i];

            }
            Neurons = new Neuron[size.Length, maxVal];
            NeuronLayer[] neuronLayers = new NeuronLayer[size.Length];
            for (int i = 0; i < size.Length; i++)
            {
                for (int n = 0; n < size[i]; n++)
                {
                    neuronLayers[i].neuron = new Neuron[maxVal];
                    if (i == 0)
                    {
                        Neurons[i, n] = new Neuron(LayerType.Input);
                        Neurons[i, n].connections = new Connection[1];
                        Neurons[i, n].connections[0].weight = 0;
                    }

                    else if (i >= 1 && i < size.Length - 1)
                    {
                        Neurons[i, n] = new Neuron(LayerType.Hiden);
                        Neurons[i, n].connections = new Connection[size[i - 1]];
                        ConnectRandomly(Neurons[i, n], size[i - 1]);
                        
                    }
                    else
                    {
                        Neurons[i, n] = new Neuron(LayerType.Output);
                        Output[n] = Neurons[i, n];
                        Neurons[i, n].connections = new Connection[size[i - 1]];
                        ConnectRandomly(Neurons[i, n], size[i - 1]);
                        
                    }
                    neuronLayers[i].neuron[n] = Neurons[i, n];
                    
                }
            }
            NeuronLayer.WriteXML(XMLPath, neuronLayers);
            layers = neuronLayers;
        }
        #region ConnectRandomly

        void ConnectRandomly(Neuron neuron, int previousLayerSize, int density = 70)
        {
            for (int i = 0; i < previousLayerSize; i++)
            {

                
                if (neuron.type == LayerType.Input) neuron.connections[i].weight = 0;


                if (random.Next(100) <= density)
                {
                    neuron.connections[i].link = i;
                    neuron.connections[i].weight = (2 * (50 - (double)random.Next(100)) / 100);

                }

            }
        }
        #endregion
 
        #region ProcessValues
        public double[] outputValues;
        public double[] ProcessValues(Neuron[,] _Neurons,double[] inputValues)
        {


            outputValues = new double[size[size.Length - 1]];
            for (int i = 0; i < size[0]; i++)
            {
                _Neurons[0, i].value = inputValues[i];
            }
            for (int i = 1; i < size.Length; i++)
            {
                for (int n = 0; n < size[i]; n++)
                {
                    for (int x = 0; x < _Neurons[i, n].connections.Length; x++)
                    {
                        if (_Neurons[i, n] != null && _Neurons[i - 1, x] != null) _Neurons[i, n].value = activationHyperbolic(_Neurons[i, n].value + _Neurons[i - 1, x].value * _Neurons[i, n].connections[x].weight);

                        if (i == size.Length - 1)
                        {
                            outputValues[n] = _Neurons[i, n].value;
                        }

                    }
                }
            }

            return outputValues;
        }

        #endregion

        public void Propogation(Dataset[] dataSet, double h_coefficient, double accuracy, int repeats = 100000)
        {
            double[] errors;
            int step= 0;
            for (int counter = 0; counter < dataSet.Length; counter++)
            {
                errors = OutputError(dataSet[counter].output, ProcessValues(Neurons, dataSet[counter].input));
                Console.WriteLine($"{step} : {outputValues[0]}");

                BackPropagation(errors, h_coefficient, counter);
                for (int a = 0, i = 0; i < errors.Length; i++)
                { if ((errors[i] < accuracy && a >= errors.Length) || repeats == step) return; 
                    a++; }
                step++;
                Propogation(dataSet,h_coefficient,accuracy,repeats);  

            }

        }
        public void BackPropagation(double[] errors ,double h_coefficient, int counter)
            {
            NeuronLayer[] Layers = new NeuronLayer[size.Length];
            for (int err = 0; err < errors.Length; err++)
            {
                for (int l = size.Length - 1; l > 0; l--)
                {
                    for (int n = 0; n < size[l]; n++)
                    {
                        for (int c = 0; c < Neurons[l, n].connections.Length; c++)
                        {

                            Neuron currentNeuron = Neurons[l, n];
                            if ((object)currentNeuron.connections[c].link != null && dataSet != null)
                            {
                                if (l != 0) currentNeuron.connections[c].weight = currentNeuron.connections[c].weight + h_coefficient * errors[err] * (currentNeuron.value) * (1 - currentNeuron.value) * Math.Log((double)(1 / Neurons[l - 1, c].value) - 1);
                                else currentNeuron.connections[c].weight = currentNeuron.connections[c].weight + h_coefficient * errors[err] * (currentNeuron.value) * (1 - currentNeuron.value) * Math.Log((double)(1 / (dataSet[counter].input[c])) - 1);
                            }
                        }
                    }

                }
            }


            NeuronLayer.WriteXML(XMLPath, NeuronLayer.toNeuronLayers(Neurons));
        }

        #region ActivationFunctions

        double SigmaDifferential(double x)
        {
            
            return (Math.Pow(Math.E, -x)/Math.Pow((1+ Math.Pow(Math.E, -x)),2));
        }
        double activationSigma(double x)
        {
            return (1/(1+Math.Pow(Math.E,-x)));
        }


        double activationHyperbolic(double x)
        {
            return (2/(1+ Math.Pow(Math.E, -2*x))-1);
           // return (Math.Pow(Math.E, 2*x)-1) / (Math.Pow(Math.E, 2 * x) + 1);
        }

        double HyperbolicDifferential(double x)
        {
            return (4*Math.Pow(Math.E,-2*x)/Math.Pow(1+Math.Pow(Math.E, -2*x), 2));
        }

        #endregion ActivationFunctions

        #region Error Function

        double[] OutputError(double[] awaitValue, double[] resultValue)
        {
            double[] error = new double[awaitValue.Length];
            for (int i = 0; i < size[size.Length - 1]; i++)
            {
                error[i] = (float)(resultValue[i] - awaitValue[i]);
            }
            return error;
        }
        #endregion

        #region EMS Func

        public double[] FirstEMS(double[] awaitValue, double[] resultValue)
        {
            double[] error = new double[awaitValue.Length];
            for(int i = 0; i < size[size.Length-1]; i++)
            {
                error[i] = MathF.Pow((float)(resultValue[i] - awaitValue[i]), 2)/2;
            }
            return error;
        }

        #endregion




        public class Neuron
        {
            public double value = 0;

            public Connection[] connections;
            public LayerType type;
            public Neuron(LayerType _type)
            {
                type = _type;

            }
            public Neuron()
                {
                }

        }
    }




    
}
