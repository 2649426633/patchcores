using System.Buffers.Binary;

namespace IndustrialAnomaly.Runtime;

public sealed class BinaryMatrix
{
    private static readonly byte[] Magic = "F32M"u8.ToArray();

    public int Rows { get; }
    public int Cols { get; }
    public float[] Data { get; }

    public BinaryMatrix(int rows, int cols, float[] data)
    {
        if (rows <= 0 || cols <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows));
        if (data.Length != checked(rows * cols))
            throw new ArgumentException("Matrix data length does not match rows*cols.", nameof(data));

        Rows = rows;
        Cols = cols;
        Data = data;
    }

    public ReadOnlySpan<float> Row(int row)
    {
        if ((uint)row >= (uint)Rows)
            throw new ArgumentOutOfRangeException(nameof(row));
        return Data.AsSpan(row * Cols, Cols);
    }

    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        writer.Write(Magic);
        writer.Write(Rows);
        writer.Write(Cols);
        foreach (var value in Data)
            writer.Write(value);
    }

    public static BinaryMatrix Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        var magic = reader.ReadBytes(4);
        if (!magic.SequenceEqual(Magic))
            throw new InvalidDataException($"Unsupported matrix format: {path}");

        var rows = reader.ReadInt32();
        var cols = reader.ReadInt32();
        if (rows <= 0 || cols <= 0)
            throw new InvalidDataException($"Invalid matrix shape {rows}x{cols}: {path}");

        var values = new float[checked(rows * cols)];
        for (var i = 0; i < values.Length; i++)
            values[i] = reader.ReadSingle();

        if (stream.Position != stream.Length)
            throw new InvalidDataException($"Unexpected trailing bytes in matrix: {path}");

        return new BinaryMatrix(rows, cols, values);
    }
}
