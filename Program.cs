using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

// FROM : https://medium.com/machinelearningadvantage/easy-k-means-clustering-with-c-and-ml-net-7b154ccd219e
// Ran command: dotnet add package Microsoft.ML

namespace K_Means_CS
{

    /// <summary>
    /// A data transfer class that holds a single square.
    /// </summary>
    public class CubeData
    {
        [LoadColumn(0)] public float r;
        [LoadColumn(1)] public float g;
        [LoadColumn(2)] public float b;
        [LoadColumn(3)] public float who_knows;
    }

    /// <summary>
    /// A prediction class that holds a single cluster prediction.
    /// </summary>
    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")] public uint PredictedClusterId;
        [ColumnName("Score")] public float[] Distances;
    }

    /// <summary>
    /// The application class.
    /// </summary>
    class Program
    {
        /// <summary>
        /// The main application entry point.
        /// </summary>
        /// <param name="args"The command line arguments></param>
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // Read the cube data from a text file
            var trainingData = mlContext.Data.LoadFromTextFile<CubeData>(
                path: "cube_data.txt",
                hasHeader: false,
                separatorChar: ',');

            // Set up a learning pipeline
            // Step 1: ML.NET KMeans can only cluster on one column, so concatenate all values into a single value called "Features"
            var pipeline = mlContext.Transforms.Concatenate(
                "Features",
                "r", 
                "g", 
                "b",
                "who_knows")
                // Step 2: Use the k-means clustering algorithm, set to 6 clusters for each face of cube
                .Append(mlContext.Clustering.Trainers.KMeans(
                    "Features",
                    numberOfClusters: 6));

            // Train the model on the data file
            Console.WriteLine("Start training model....");
            var model = pipeline.Fit(trainingData);
            Console.WriteLine("Model training complete!");

            int[] predictions = new int[53];

            // The trainingData object that data was loaded into is not enumerable, so create an enumerable object to loop through
            // Instructions taken from: https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/inspect-intermediate-data-ml-net
            IEnumerable<CubeData> CubeDataEnumerable =
                mlContext.Data.CreateEnumerable<CubeData>(trainingData, reuseRowObject: true);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<CubeData, ClusterPrediction>(model);
            int i = 0;

            // Iterate over each row
            foreach (CubeData row in CubeDataEnumerable)
            {
                var new_pred = predictionEngine.Predict(
                  new CubeData()
                  {
                    r = row.r,
                    g = row.g,
                    b = row.b,
                    who_knows = row.who_knows,
                  }
                );
                Console.WriteLine($"Cluster: {new_pred.PredictedClusterId}");
                Console.WriteLine($"Distances: {string.Join(" ", new_pred.Distances)}");
                i++;
            }
        }
    }
}
