using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Xml.Serialization;

namespace NeuralNetwork.Classes
{
    public enum LayerType{Input,Output, Hiden }


    struct NeuronLayer
    {
        public NeuralNetworkTemplate.Neuron[] neuron;
        public int[] size;


        public void WriteXML(string path, NeuronLayer[] neuronLayers)
        {
            Stream stream = File.OpenWrite(path);
            XmlSerializer serializer = new XmlSerializer(typeof(NeuronLayer[]));
            serializer.Serialize(stream, neuronLayers);
        }
        public NeuronLayer[] ReadXML(string path)
        {
            Stream stream = File.OpenRead(path);
            XmlSerializer serializer = new XmlSerializer(typeof(NeuronLayer[]));
            return (NeuronLayer[])serializer.Deserialize(stream);
        }
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

        public int link; // link to array of previous layer
        public double weight;
    }





    public class NeuralNetworkTemplate
    {
        
        public Neuron[] Output { get; private set; }
        public int[] size;

        Random random = new Random();
        public Neuron[,] Neurons;

        


        public NeuralNetworkTemplate(int[] _size)
        {
            if (_size.Length >= 3) size = _size;
            else
            {
                Debug.WriteLine("<<< catch : int[] _size : Network might consist of three or more layers");
                Console.WriteLine("<<< catch : int[] _size : Network might consist of three or more layers");
            }
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
            for (int i = 0; i < size.Length; i++)
            {
                for (int n = 0; n < size[i]; n++)
                {
                    if (i == 0) Neurons[i, n] = new Neuron(LayerType.Input);
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
                }
            }

        }


        void ConnectRandomly(Neuron neuron, int previousLayerSize, int density = 70)
        {
            for (int i = 0; i < previousLayerSize; i++)
            {
                if (random.Next(100) <= density)
                {
                    neuron.connections[i].link = i;
                    neuron.connections[i].weight = (2 * (50 - (double)random.Next(100)) / 100);

                }

            }
        }

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


        void BackPropagation()
            {

            }

        #region ActivationFunctions
        double activationSigma(double x)
        {
            return (1/(1+Math.Pow(Math.E,-x)));
        }


        double activationHyperbolic(double x)
        {
            return (Math.Pow(Math.E, 2*x)-1) / (Math.Pow(Math.E, 2 * x) + 1);
        }

        #endregion ActivationFunctions


        public class Neuron
        {
            public double value = 0;

            public Connection[] connections;
            LayerType type;
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
