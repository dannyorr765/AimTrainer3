using System;

/// <summary>
/// Feedforward neural network with tanh activation.
/// Pure C#, no Unity dependency — testable standalone.
/// </summary>
public class NeuralNetwork
{
    private int[] layers;
    private float[][] biases;      // biases[l] → destination layer of connection l
    private float[][][] weights;   // weights[l][j][i] → layer l neuron i to layer l+1 neuron j
    private static Random rng = new Random();

    public NeuralNetwork(int[] layers)
    {
        this.layers = (int[])layers.Clone();

        biases = new float[layers.Length - 1][];
        weights = new float[layers.Length - 1][][];

        for (int l = 0; l < layers.Length - 1; l++)
        {
            int inSize = layers[l];
            int outSize = layers[l + 1];
            float scale = (float)Math.Sqrt(2.0 / inSize); // He initialization

            biases[l] = new float[outSize];
            weights[l] = new float[outSize][];

            for (int j = 0; j < outSize; j++)
            {
                weights[l][j] = new float[inSize];
                for (int i = 0; i < inSize; i++)
                    weights[l][j][i] = Gaussian() * scale;
            }
        }
    }

    public float[] Forward(float[] inputs)
    {
        float[] current = inputs;

        for (int l = 0; l < weights.Length; l++)
        {
            float[] next = new float[layers[l + 1]];

            for (int j = 0; j < next.Length; j++)
            {
                float sum = biases[l][j];
                for (int i = 0; i < current.Length; i++)
                    sum += weights[l][j][i] * current[i];
                next[j] = (float)Math.Tanh(sum);
            }

            current = next;
        }

        return current;
    }

    public NeuralNetwork Clone()
    {
        var clone = new NeuralNetwork(layers);
        for (int l = 0; l < weights.Length; l++)
        {
            Array.Copy(biases[l], clone.biases[l], biases[l].Length);
            for (int j = 0; j < weights[l].Length; j++)
                Array.Copy(weights[l][j], clone.weights[l][j], weights[l][j].Length);
        }
        return clone;
    }

    /// <summary>
    /// Gaussian noise applied per-weight with probability `rate`.
    /// </summary>
    public void Mutate(float rate, float magnitude)
    {
        for (int l = 0; l < weights.Length; l++)
        {
            for (int j = 0; j < biases[l].Length; j++)
                if (rng.NextDouble() < rate)
                    biases[l][j] += Gaussian() * magnitude;

            for (int j = 0; j < weights[l].Length; j++)
                for (int i = 0; i < weights[l][j].Length; i++)
                    if (rng.NextDouble() < rate)
                        weights[l][j][i] += Gaussian() * magnitude;
        }
    }

    /// <summary>
    /// Uniform crossover: each weight independently picked from parent A or B.
    /// </summary>
    public static NeuralNetwork Crossover(NeuralNetwork a, NeuralNetwork b)
    {
        var child = a.Clone();
        for (int l = 0; l < child.weights.Length; l++)
        {
            for (int j = 0; j < child.biases[l].Length; j++)
                if (rng.NextDouble() < 0.5)
                    child.biases[l][j] = b.biases[l][j];

            for (int j = 0; j < child.weights[l].Length; j++)
                for (int i = 0; i < child.weights[l][j].Length; i++)
                    if (rng.NextDouble() < 0.5)
                        child.weights[l][j][i] = b.weights[l][j][i];
        }
        return child;
    }

    // --- Serialization ---

    public float[] Flatten()
    {
        int count = 0;
        for (int l = 0; l < weights.Length; l++)
        {
            count += biases[l].Length;
            for (int j = 0; j < weights[l].Length; j++)
                count += weights[l][j].Length;
        }

        float[] flat = new float[count];
        int idx = 0;
        for (int l = 0; l < weights.Length; l++)
            for (int j = 0; j < weights[l].Length; j++)
            {
                for (int i = 0; i < weights[l][j].Length; i++)
                    flat[idx++] = weights[l][j][i];
                flat[idx++] = biases[l][j];
            }
        return flat;
    }

    public void Unflatten(float[] flat)
    {
        int idx = 0;
        for (int l = 0; l < weights.Length; l++)
            for (int j = 0; j < weights[l].Length; j++)
            {
                for (int i = 0; i < weights[l][j].Length; i++)
                    weights[l][j][i] = flat[idx++];
                biases[l][j] = flat[idx++];
            }
    }

    public int[] GetTopology() => (int[])layers.Clone();

    // --- Helpers ---

    private static float Gaussian()
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
    }
}
