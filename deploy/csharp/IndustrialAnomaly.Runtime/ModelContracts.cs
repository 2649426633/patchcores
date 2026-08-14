using System.Text.Json;
using System.Text.Json.Serialization;

namespace IndustrialAnomaly.Runtime;

public sealed record DefectClassDefinition
{
    public required string Name { get; init; }
    public required string ImageDirectory { get; init; }
}

public sealed record ProductBuildDefinition
{
    public required string ProductName { get; init; }
    public required string NormalImageDirectory { get; init; }
    public required IReadOnlyList<DefectClassDefinition> DefectClasses { get; init; }

    public float TileFraction { get; init; } = 0.75f;
    public float TileOverlap { get; init; } = 0.25f;
    public float CoresetRatio { get; init; } = 0.10f;
    public int MaxPatchCoreMemoryRows { get; init; } = 16000;
    public float BboxRelativeThreshold { get; init; } = 0.70f;
    public float RoiMargin { get; init; } = 0.50f;
    public float ClsWeight { get; init; } = 0.50f;
    public float CenterWeight { get; init; } = 0.50f;
}

public sealed record EngineManifest
{
    [JsonPropertyName("format_version")]
    public int FormatVersion { get; init; }

    [JsonPropertyName("patchcore")]
    public required PatchCoreEngineManifest PatchCore { get; init; }

    [JsonPropertyName("dinov2")]
    public required DINOEngineManifest DINOv2 { get; init; }

    [JsonPropertyName("normalization")]
    public required NormalizationManifest Normalization { get; init; }

    public static EngineManifest Load(string path)
    {
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<EngineManifest>(json)
            ?? throw new InvalidDataException($"Invalid engine manifest: {path}");
    }
}

public sealed record PatchCoreEngineManifest
{
    [JsonPropertyName("file")]
    public required string File { get; init; }

    [JsonPropertyName("input")]
    public required string Input { get; init; }

    [JsonPropertyName("memory_input")]
    public required string MemoryInput { get; init; }

    [JsonPropertyName("input_shape")]
    public required int[] InputShape { get; init; }

    [JsonPropertyName("outputs")]
    public required string[] Outputs { get; init; }

    [JsonPropertyName("output_shape")]
    public required int[] OutputShape { get; init; }

    [JsonPropertyName("score_shape")]
    public required int[] ScoreShape { get; init; }

    [JsonPropertyName("score_metric")]
    public required string ScoreMetric { get; init; }

    [JsonPropertyName("patch_grid")]
    public required int[] PatchGrid { get; init; }

    [JsonPropertyName("embedding_dim")]
    public int EmbeddingDim { get; init; }
}

public sealed record DINOEngineManifest
{
    [JsonPropertyName("file")]
    public required string File { get; init; }

    [JsonPropertyName("input")]
    public required string Input { get; init; }

    [JsonPropertyName("input_shape")]
    public required int[] InputShape { get; init; }

    [JsonPropertyName("outputs")]
    public required string[] Outputs { get; init; }

    [JsonPropertyName("embedding_dim")]
    public int EmbeddingDim { get; init; }

    [JsonPropertyName("center_fraction")]
    public float CenterFraction { get; init; }
}

public sealed record NormalizationManifest
{
    [JsonPropertyName("mean")]
    public required float[] Mean { get; init; }

    [JsonPropertyName("std")]
    public required float[] Std { get; init; }

    [JsonPropertyName("channel_order")]
    public string ChannelOrder { get; init; } = "RGB";

    [JsonPropertyName("tensor_layout")]
    public string TensorLayout { get; init; } = "NCHW";
}

public sealed record ProductModelManifest
{
    public int FormatVersion { get; init; } = 1;
    public required string ProductName { get; init; }
    public required string PatchCoreMemoryFile { get; init; }
    public required string DefectClsFile { get; init; }
    public required string DefectCenterFile { get; init; }
    public required IReadOnlyList<string> DefectLabels { get; init; }
    public required IReadOnlyList<string> Classes { get; init; }
    public float TileFraction { get; init; }
    public float TileOverlap { get; init; }
    public float CoresetRatio { get; init; }
    public int PatchCoreMemoryRows { get; init; }
    public string PatchCoreMemoryStrategy { get; init; } = "bounded_reservoir";
    public float BboxRelativeThreshold { get; init; }
    public float RoiMargin { get; init; }
    public float ClsWeight { get; init; }
    public float CenterWeight { get; init; }

    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(
            path,
            JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true })
        );
    }
}

public sealed record PatchCoreOnnxResult(
    BinaryMatrix Embeddings,
    float[] PatchScores
);
