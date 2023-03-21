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
            hidden = new double[size.Length,maxLayerSize];

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
            
            for(int x = 0; x < maxLayerSize; x++)for(int y = 0;y< size.Length; y++)
                {
                    if (randomConn.Next(0, 100) <= threshold) _weights[x, y] = randomWeight.Next(-1,1) /(double)(2 * randomWeight.Next(1,100));
                    else weights[x, y] = 0f;
                    
                }
            

            


        }

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
            return (2 / (1 + Math.Pow(Math.E, -2 * x))) - 1);
            // return (Math.Pow(Math.E, 2*x)-1) / (Math.Pow(Math.E, 2 * x) + 1);
        }

        static double HyperbolicDifferential(double x)
        {
            return (4 * Math.Pow(Math.E, -2 * x) / Math.Pow(1 + Math.Pow(Math.E, -2 * x), 2));
        }

        #endregion ActivationFunctions


        #region EMS Func

       /* public double EMS(double[] awaitValue, double[] resultValue)
        {
            double error = 0;
            for (int i = 0; i < size[size.Length - 1], i++)
            {
                error += MathF.Pow((double)(resultValue[i] - awaitValue[i]), 2);
            }
            return error / 2;
        }
*/
        #endregion

    }


}
