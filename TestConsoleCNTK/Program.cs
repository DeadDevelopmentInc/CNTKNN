using FunctionalLibrary;
using System;
using System.Diagnostics;
using System.IO;

namespace TestConsoleCNTK
{
    class Program
    {
        static void Main(string[] args)
        {
            var r = ExtractMFCC.ReadWav("OSR_us_000_0010_8k.wav", out double[] L, out double[] R);
            if (L != null)
            {
                var result = ExtractMFCC.MFCC_20_calculation(L);
                Console.WriteLine(result.Length);
            }
            
        }
    }
}
