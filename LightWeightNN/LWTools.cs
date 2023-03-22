using NeuralNetwork.Lightweight;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NeuralNetwork.LightWeight.Tools
{
    public static class LWTools
    {
        public static void SerializeToJson(LWNeuralNetwork network, string filePath)
        {
            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) }
            };
            string json = JsonSerializer.Serialize(network, options);
            File.WriteAllText(filePath, json);
        }

        public static LWNeuralNetwork DeserializeFromJson(string filePath)
        {
            string json = File.ReadAllText(filePath);
            var options = new JsonSerializerOptions
            {
                Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) }
            };
            return JsonSerializer.Deserialize<LWNeuralNetwork>(json, options);
        }

        public static void SerializeNetworkToBinaryFile(LWNeuralNetwork network, string filename)
        {
            using (FileStream fileStream = new FileStream(filename, FileMode.Create))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                formatter.Serialize(fileStream, network);
            }
        }
        public static LWNeuralNetwork DeserializeNetworkFromBinaryFile(string filename)
        {
            using (FileStream fileStream = new FileStream(filename, FileMode.Open))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                return (LWNeuralNetwork)formatter.Deserialize(fileStream);
            }
        }
    }
}

