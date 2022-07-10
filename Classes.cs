using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Xml.Serialization;

namespace NeuralNetwork.Classes 
{
    public enum LayerType{Input,Output, Hiden }



    public struct NeuronLayer
    {
        public NeuralNetworkTemplate.Neuron[] neuron;
        public int[] size;


        public static void WriteXML(string path, NeuronLayer[] neuronLayers)
        {
            Stream stream = File.OpenWrite(path);
            XmlSerializer serializer = new XmlSerializer(typeof(NeuronLayer[]));
            serializer.Serialize(stream, neuronLayers);
        }
        public static NeuronLayer[] ReadXML(string path)
        {
            
            Stream stream = File.OpenRead(path);
            XmlSerializer serializer = new XmlSerializer(typeof(NeuronLayer[]));
            return (NeuronLayer[])serializer.Deserialize(stream);
        }
    }

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
        
        public Neuron[] Output { get; private set; }
        public int[] size;
        public Dataset[] dataSet;

        Random random = new Random();
        public Neuron[,] Neurons;

        public string XMLPath;


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
            NeuronLayer[] neuronLayers = new NeuronLayer[size.Length];
            for (int i = 0; i < size.Length; i++)
            {
                for (int n = 0; n < size[i]; n++)
                {
                    neuronLayers[i].neuron = new Neuron[maxVal];
                    if (i == 0)
                    {
                        Neurons[i, n] = new Neuron(LayerType.Input);
                        
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

        }
        #region ConnectRandomly

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

        public void Propogation(Dataset[] dataSet)
        {
            for (int i = 0; i < dataSet.Length; i++)
            {
                EMS(dataSet[0].output, ProcessValues(Neurons, dataSet[0].input));
            }

        }
        public void BackPropagation()
            {
            




            /*NeuronLayer.WriteXML(XMLPath,);*/
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
            return (2/(1+ Math.Pow(Math.E, -2*x)))-1);
           // return (Math.Pow(Math.E, 2*x)-1) / (Math.Pow(Math.E, 2 * x) + 1);
        }

        double HyperbolicDifferential(double x)
        {
            return (4*Math.Pow(Math.E,-2*x)/Math.Pow(1+Math.Pow(Math.E, -2*x), 2));
        }

        #endregion ActivationFunctions


        #region EMS Func

        public double EMS(double[] awaitValue, double[] resultValue)
        {
            double error = 0;
            for(int i = 0; i < size[size.Length-1], i++)
            {
                error += MathF.Pow((float)(resultValue[i] - awaitValue[i]), 2);
            }
            return error/2;
        }

        #endregion




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
