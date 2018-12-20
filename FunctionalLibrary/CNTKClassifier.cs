using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

using Accord;
using Accord.Math;
using Accord.Statistics.Analysis;
using CNTK;
using KerasSharp;
using KerasSharp.Activations;
using KerasSharp.Backends;
using KerasSharp.Initializers;
using KerasSharp.Losses;
using KerasSharp.Metrics;
using KerasSharp.Models;
using KerasSharp.Optimizers;


using static KerasSharp.Backends.Current;


namespace FunctionalLibrary
{
    public class CNTKClassifier
    {
        public string TrainPath = "test";
        public string ValidatePath = "test";
        public Sequential Model { get; set; } = new Sequential();
        public int?[] Input_Shape;
        public int num_Classes;
        List<double[,]> TrainData = new List<double[,]>();
        List<double[,]> ValidData = new List<double[,]>();
        List<int> TrainLabels = new List<int>();
        List<int> ValidLabels = new List<int>();

        static CNTKClassifier()
        {
            Current.Switch("KerasSharp.Backends.CNTK.GPU");
        }

        public CNTKClassifier() { }

        public CNTKClassifier(string Train, string Validate)
        {
            Current.Switch("KerasSharp.Backends.CNTK.GPU");
            Model.Add(new Dense(12, input_shape: Input_Shape, activation: new ReLU()));
            Model.Add(new Dense(8, activation: new ReLU()));
            Model.Add(new Dense(num_Classes, activation: new Sigmoid()));
            Model.Compile(loss: new MeanSquareError(),
                optimizer: new Adam(),
                metrics: new[] { new Accuracy() });
        }

        public void InitialData()
        {
            int i = 0;
            foreach(string folder in Directory.GetFiles(TrainPath))
            {
                foreach (string file in Directory.GetFiles(folder))
                {
                    ExtractMFCC.ReadWav(file, out double[] L, out double[] R);
                    TrainData.Add(ExtractMFCC.MFCC_20_calculation(L));
                    TrainLabels.Add(i);
                }
                i++;
            }
            i = 0;
            foreach (string folder in Directory.GetFiles(ValidatePath))
            {
                foreach (string file in Directory.GetFiles(folder))
                {
                    ExtractMFCC.ReadWav(file, out double[] L, out double[] R);
                    ValidData.Add(ExtractMFCC.MFCC_20_calculation(L));
                    TrainLabels.Add(i);
                }
                i++;
            }
            foreach(var t in TrainData)
            {
                if(t.Length / 20 != ExtractMFCC.countFramesMax)
                {
                    t.Add(new double[(ExtractMFCC.countFramesMax - (t.Length / 20)), 20]);
                }
            }
            foreach (var t in TrainData)
            {
                if (t.Length / 20 != ExtractMFCC.countFramesMax)
                {
                    t.Add(new double[(ExtractMFCC.countFramesMax - (t.Length / 20)), 20]);
                }
            }
        }

        public void Train()
        {
            Model.fit(TrainData.ToArray(), TrainLabels.ToArray(), epochs: 10, batch_size: 10);
        }

        public void Test()
        {

        }

        public void Predict(string file)
        {
            ExtractMFCC.ReadWav(file, out double[] L, out double[] R);
            var mfcc = ExtractMFCC.MFCC_20_calculation(L);
            float[] pred = Model.predict(mfcc)[0].To<float[]>();

        }
    }
}
