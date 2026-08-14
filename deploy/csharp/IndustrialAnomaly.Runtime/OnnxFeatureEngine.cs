using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace IndustrialAnomaly.Runtime;

public sealed class OnnxFeatureEngine : IDisposable
{
    private readonly InferenceSession _patchCoreSession;
    private readonly InferenceSession _dinoSession;
    private readonly EngineManifest _manifest;

    public EngineManifest Manifest => _manifest;

    public OnnxFeatureEngine(string engineDirectory)
    {
        var root = Path.GetFullPath(engineDirectory);
        _manifest = EngineManifest.Load(Path.Combine(root, "engine_config.json"));

        _patchCoreSession = new InferenceSession(
            Path.Combine(root, _manifest.PatchCore.File)
        );
        _dinoSession = new InferenceSession(
            Path.Combine(root, _manifest.DINOv2.File)
        );
    }

    public BinaryMatrix ExtractPatchCoreEmbeddings(Mat bgrImage)
    {
        return RunPatchCore(bgrImage, null).Embeddings;
    }

    public PatchCoreOnnxResult RunPatchCore(Mat bgrImage, BinaryMatrix? memoryBank)
    {
        if (bgrImage.Empty())
            throw new ArgumentException("Input image is empty.", nameof(bgrImage));

        var cfg = _manifest.PatchCore;
        var size = cfg.InputShape[2];
        var inputData = PrepareImage(bgrImage, size);
        long[] imageShape = [1, 3, size, size];

        var memory = memoryBank ?? new BinaryMatrix(
            1,
            cfg.EmbeddingDim,
            new float[cfg.EmbeddingDim]
        );
        if (memory.Cols != cfg.EmbeddingDim)
        {
            throw new ArgumentException(
                $"PatchCore memory dimension is {memory.Cols}; expected {cfg.EmbeddingDim}.",
                nameof(memoryBank)
            );
        }
        long[] memoryShape = [memory.Rows, memory.Cols];

        using var imageInput = OrtValue.CreateTensorValueFromMemory(inputData, imageShape);
        using var memoryInput = OrtValue.CreateTensorValueFromMemory(memory.Data, memoryShape);
        var inputs = new Dictionary<string, OrtValue>
        {
            [cfg.Input] = imageInput,
            [cfg.MemoryInput] = memoryInput,
        };

        using var runOptions = new RunOptions();
        using var outputs = _patchCoreSession.Run(
            runOptions,
            inputs,
            _patchCoreSession.OutputNames
        );
        if (outputs.Count != 2)
            throw new InvalidDataException(
                $"PatchCore ONNX returned {outputs.Count} outputs; expected 2."
            );

        var embeddingValues = outputs[0].GetTensorDataAsSpan<float>().ToArray();
        var scores = outputs[1].GetTensorDataAsSpan<float>().ToArray();
        var rows = cfg.OutputShape[1];
        var cols = cfg.EmbeddingDim;

        if (embeddingValues.Length != rows * cols)
            throw new InvalidDataException("Unexpected PatchCore embedding output shape.");
        if (scores.Length != rows)
            throw new InvalidDataException("Unexpected PatchCore score output shape.");

        return new PatchCoreOnnxResult(
            new BinaryMatrix(rows, cols, embeddingValues),
            scores
        );
    }

    public (float[] Cls, float[] Center) ExtractDinoEmbeddings(Mat bgrImage)
    {
        if (bgrImage.Empty())
            throw new ArgumentException("Input image is empty.", nameof(bgrImage));

        var size = _manifest.DINOv2.InputShape[2];
        var inputData = PrepareImage(bgrImage, size);
        long[] shape = [1, 3, size, size];

        using var input = OrtValue.CreateTensorValueFromMemory(inputData, shape);
        var inputs = new Dictionary<string, OrtValue>
        {
            [_manifest.DINOv2.Input] = input,
        };
        using var runOptions = new RunOptions();
        using var outputs = _dinoSession.Run(
            runOptions,
            inputs,
            _dinoSession.OutputNames
        );

        if (outputs.Count != 2)
            throw new InvalidDataException(
                $"DINOv2 ONNX returned {outputs.Count} outputs; expected 2."
            );

        var cls = outputs[0].GetTensorDataAsSpan<float>().ToArray();
        var center = outputs[1].GetTensorDataAsSpan<float>().ToArray();
        if (cls.Length != _manifest.DINOv2.EmbeddingDim ||
            center.Length != _manifest.DINOv2.EmbeddingDim)
        {
            throw new InvalidDataException(
                "DINOv2 ONNX embedding dimension does not match engine_config.json."
            );
        }
        return (cls, center);
    }

    private float[] PrepareImage(Mat bgrImage, int size)
    {
        using var resized = new Mat();
        Cv2.Resize(bgrImage, resized, new Size(size, size), 0, 0, InterpolationFlags.Linear);

        using var rgb = new Mat();
        if (resized.Channels() == 1)
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.GRAY2RGB);
        else if (resized.Channels() == 4)
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGRA2RGB);
        else
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

        var mean = _manifest.Normalization.Mean;
        var std = _manifest.Normalization.Std;
        if (mean.Length != 3 || std.Length != 3)
            throw new InvalidDataException("Expected three-channel ImageNet normalization.");

        var plane = size * size;
        var output = new float[3 * plane];
        for (var y = 0; y < size; y++)
        {
            for (var x = 0; x < size; x++)
            {
                var pixel = rgb.At<Vec3b>(y, x);
                var index = y * size + x;
                output[index] = (pixel.Item0 / 255.0f - mean[0]) / std[0];
                output[plane + index] = (pixel.Item1 / 255.0f - mean[1]) / std[1];
                output[2 * plane + index] = (pixel.Item2 / 255.0f - mean[2]) / std[2];
            }
        }
        return output;
    }

    public void Dispose()
    {
        _patchCoreSession.Dispose();
        _dinoSession.Dispose();
    }
}
