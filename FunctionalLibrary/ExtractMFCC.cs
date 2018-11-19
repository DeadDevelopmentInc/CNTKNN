using MathNet.Numerics.IntegralTransforms;
using System;
using System.IO;
using System.Linq;
using System.Numerics;

namespace FunctionalLibrary
{
    public static class ExtractMFCC
    {
        public const int BlockLength = 2048;
        public static double[] Frame;                                                          //один фрейм
        public static double[,] FrameMass;                                                     //массив всех фреймов по BlockLength отсчетов или 128 (for 16khz) мс        
        public static Complex[,] FrameMassFft;                                                 //массив результатов FFT для всех фреймов
        public static int countFramesMax = 0;
        static readonly int[] _filterPoints = {6,18,31,46,63,82,103,127,154,184,218,
                              257,299,348,402,463,531,608,695,792,901,1023};            //массив опорных точек для фильтрации спекрта фрейма

        static readonly double[,] _h = new double[20, BlockLength / 2];                        //массив из 20-ти фильтров для каждого MFCC

        public static bool ReadWav(string filename, out double[] L, out double[] R)
        {
            L = R = null;
            //float [] left = new float[1];

            //float [] right;
            try
            {
                using (FileStream fs = File.Open(filename, FileMode.Open))
                {
                    BinaryReader reader = new BinaryReader(fs);

                    // chunk 0
                    int chunkID = reader.ReadInt32();
                    int fileSize = reader.ReadInt32();
                    int riffType = reader.ReadInt32();


                    // chunk 1
                    int fmtID = reader.ReadInt32();
                    int fmtSize = reader.ReadInt32(); // bytes for this chunk
                    int fmtCode = reader.ReadInt16();
                    int channels = reader.ReadInt16();
                    int sampleRate = reader.ReadInt32();
                    int byteRate = reader.ReadInt32();
                    int fmtBlockAlign = reader.ReadInt16();
                    int bitDepth = reader.ReadInt16();

                    if (fmtSize == 18)
                    {
                        // Read any extra values
                        int fmtExtraSize = reader.ReadInt16();
                        reader.ReadBytes(fmtExtraSize);
                    }

                    // chunk 2
                    int dataID = reader.ReadInt32();
                    int bytes = reader.ReadInt32();

                    // DATA!
                    byte[] byteArray = reader.ReadBytes(bytes);

                    int bytesForSamp = bitDepth / 8;
                    int samps = bytes / bytesForSamp;


                    double[] asFloat = null;
                    switch (bitDepth)
                    {
                        case 64:
                            double[]
                            asDouble = new double[samps];
                            Buffer.BlockCopy(byteArray, 0, asDouble, 0, bytes);
                            asFloat = Array.ConvertAll(asDouble, e => (double)e);
                            break;
                        case 32:
                            asFloat = new double[samps];
                            Buffer.BlockCopy(byteArray, 0, asFloat, 0, bytes);
                            break;
                        case 16:
                            Int16[]
                            asInt16 = new Int16[samps];
                            Buffer.BlockCopy(byteArray, 0, asInt16, 0, bytes);
                            asFloat = Array.ConvertAll(asInt16, e => e / (double)Int16.MaxValue);
                            break;
                        default:
                            return false;
                    }
                    switch (channels)
                    {
                        case 1:
                            L = asFloat;
                            R = null;
                            return true;
                        case 2:
                            L = new double[samps];
                            R = new double[samps];
                            for (int i = 0, s = 0; i < samps; i++)
                            {
                                L[i] = asFloat[s++];
                                R[i] = asFloat[s++];
                            }
                            return true;
                        default:
                            return false;
                    }
                }
            }
            catch
            {
                return false;
            }
        }

        public static double[,] MFCC_20_calculation(double[] wavPcm)
        {
            int countFrames = (wavPcm.Length * 2 / BlockLength) + 1; //количество отрезков в сигнале
            if(countFrames > countFramesMax)
            {
                countFramesMax = countFrames;
            }
            // RMS_gate(wavPcm);          //применение noise gate
            Normalize(wavPcm); //нормализация
            FrameMass = SplitToFrames(wavPcm); //формирование массива фреймов
            ApplyHammingWindow(FrameMass); //окно Хэмминга для каждого отрезка
            FrameMassFft = CalculateFramesFFT(FrameMass); //FFT для каждого фрейма

            double[,] mfccMass = new double[countFrames, 20]; //массив наборов MFCC для каждого фрейма

            //***********   Расчет гребенчатых фильтров спектра:    *************
            for (int i = 0; i < 20; i++)
            {
                for (int j = 0; j < BlockLength / 2; j++)
                {
                    if (j < _filterPoints[i]) _h[i, j] = 0;
                    if ((_filterPoints[i] <= j) & (j <= _filterPoints[i + 1]))
                        _h[i, j] = ((double)(j - _filterPoints[i]) / (_filterPoints[i + 1] - _filterPoints[i]));
                    if ((_filterPoints[i + 1] <= j) & (j <= _filterPoints[i + 2]))
                        _h[i, j] = ((double)(_filterPoints[i + 2] - j) / (_filterPoints[i + 2] - _filterPoints[i + 1]));
                    if (j > _filterPoints[i + 2]) _h[i, j] = 0;
                }
            }
            for (int nframe = 0; nframe < countFrames; nframe++)
            {
                //**********    Применение фильтров и логарифмирование энергии спектра для каждого фрейма   ***********
                double[] s = new double[20];
                for (int i = 0; i < 20; i++)
                {
                    for (int j = 0; j < (BlockLength / 2); j++)
                        s[i] += Math.Pow(FrameMassFft[nframe, j].Magnitude, 2) * _h[i, j];

                    if (Math.Abs(s[i]) > float.Epsilon)
                        s[i] = Math.Log(s[i], Math.E);
                }

                //**********    DCT и массив MFCC для каждого фрейма на выходе     ***********
                for (int l = 0; l < 20; l++)
                    for (int i = 0; i < 20; i++) mfccMass[nframe, l] += s[i] * Math.Cos(Math.PI * l * (i * 0.5 / 20));
            }

            return mfccMass;
        }

        /// <summary>
        /// Функция для подавления шума по среднекравратичному уровню
        /// </summary>
        /// <param name="wavPcm">Массив значений амплитуд аудиосигнала</param>
        private static void RMS_gate(double[] wavPcm)
        {
            int k = 0;
            double rms = 0;

            for (int j = 0; j < wavPcm.Length; j++)
            {
                if (k < 100)
                {
                    rms += Math.Pow((wavPcm[j]), 2);
                    k++;
                }
                else
                {
                    if (Math.Sqrt(rms / 100) < 0.005)
                        for (int i = j - 100; i <= j; i++) wavPcm[i] = 0;
                    k = 0; rms = 0;
                }
            }
        }

        /// <summary>
        /// Функция нормализации сигнала
        /// </summary>
        /// <param name="wavPcm">Массив значений амплитуд аудиосигнала</param>
        private static void Normalize(double[] wavPcm)
        {
            double[] absWavBuf = new double[wavPcm.Length];
            for (int i = 0; i < wavPcm.Length; i++)
                if (wavPcm[i] < 0) absWavBuf[i] = -wavPcm[i];   //приводим все значения амплитуд к абсолютной величине 
                else absWavBuf[i] = wavPcm[i];                    //для определения максимального пика
            double max = absWavBuf.Max();
            double k = 1f / max;        //получаем коэффициент нормализации            

            for (int i = 0; i < wavPcm.Length; i++) //записываем нормализованные значения в исходный массив амплитуд

                wavPcm[i] = wavPcm[i] * k;
        }

        /// <summary>
        /// Функция для формирования двумерного массива отрезков сигнала длиной по 128мс.
        /// При этом начало каждого следующего отрезка делит предыдущий пополам
        /// </summary>
        /// <param name="wavPcm">Массив значений амплитуд аудиосигнала</param>
        private static double[,] SplitToFrames(double[] wavPcm)
        {
            int countFrames = 0;
            int countSamp = 0;

            var frameMass1 = new double[wavPcm.Length * 2 / BlockLength + 1, BlockLength];
            for (int j = 0; j < wavPcm.Length; j++)
            {
                if (j >= (BlockLength / 2))      //запись фреймов в массив
                {
                    countSamp++;
                    if (countSamp >= BlockLength + 1)
                    {
                        countFrames += 2;
                        countSamp = 1;
                    }
                    frameMass1[countFrames, countSamp - 1] = wavPcm[j - (BlockLength / 2)];
                    frameMass1[countFrames + 1, countSamp - 1] = wavPcm[j];
                }
            }
            return frameMass1;
        }

        /// <summary>
        /// Оконная функция Хэмминга
        /// </summary>
        /// <param name="frames">Двумерный массив отрезвов аудиосигнала</param>
        public static void ApplyHammingWindow(double[,] frames)
        {
            double w = 2.0 * Math.PI / BlockLength;

            for (int nframe = 0; nframe < frames.GetLength(0); nframe++)
                for (int nsample = 0; nsample < BlockLength; nsample++)
                    frames[nframe, nsample] = (0.54 - 0.46 * Math.Cos(w * nsample)) * frames[nframe, nsample];
        }

        /// <summary>
        /// Быстрое преобразование фурье для набора отрезков
        /// </summary>
        /// <param name="frames">Двумерный массив отрезвов аудиосигнала</param>
        /// <param name="wav_PCM">Массив значений амплитуд аудиосигнала</param>
        private static Complex[,] CalculateFramesFFT(double[,] frames)
        {
            var frameMassComplex = new Complex[frames.GetLength(0), BlockLength]; //для хранения результатов FFT каждого фрейма в комплексном виде

            var fftFrame = new Complex[BlockLength];     //спектр одного фрейма

            for (int k = 0; k < frames.GetLength(0); k++)
            {
                for (int i = 0; i < BlockLength; i++)
                    fftFrame[i] = frames[k, i];

                Fourier.Forward(fftFrame, FourierOptions.Matlab);

                for (int i = 0; i < BlockLength; i++)
                    frameMassComplex[k, i] = fftFrame[i];
            }
            return frameMassComplex;
        }
    }
}
