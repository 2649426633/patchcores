using System.Text.Json;

namespace IndustrialAnomaly.Runtime;

public sealed record DefectPrediction(
    string PredictedClass,
    float Top1Similarity,
    string? Top2Class,
    float Top2Similarity,
    float Margin,
    IReadOnlyDictionary<string, float> ClassScores
);

public sealed class ProductModel
{
    public string DirectoryPath { get; }
    public ProductModelManifest Manifest { get; }
    public BinaryMatrix PatchCoreMemory { get; }
    public BinaryMatrix DefectCls { get; }
    public BinaryMatrix DefectCenter { get; }

    private ProductModel(
        string directoryPath,
        ProductModelManifest manifest,
        BinaryMatrix patchCoreMemory,
        BinaryMatrix defectCls,
        BinaryMatrix defectCenter
    )
    {
        DirectoryPath = directoryPath;
        Manifest = manifest;
        PatchCoreMemory = patchCoreMemory;
        DefectCls = defectCls;
        DefectCenter = defectCenter;
    }

    public static ProductModel Load(string productDirectory)
    {
        var root = Path.GetFullPath(productDirectory);
        var manifestPath = Path.Combine(root, "product_model.json");
        if (!File.Exists(manifestPath))
            throw new FileNotFoundException("product_model.json not found.", manifestPath);

        var manifest = JsonSerializer.Deserialize<ProductModelManifest>(
            File.ReadAllText(manifestPath)
        ) ?? throw new InvalidDataException($"Invalid product model: {manifestPath}");

        var memory = BinaryMatrix.Load(Path.Combine(root, manifest.PatchCoreMemoryFile));
        var cls = BinaryMatrix.Load(Path.Combine(root, manifest.DefectClsFile));
        var center = BinaryMatrix.Load(Path.Combine(root, manifest.DefectCenterFile));

        if (cls.Rows != center.Rows || cls.Cols != center.Cols)
            throw new InvalidDataException("CLS and Center defect banks have different shapes.");
        if (manifest.DefectLabels.Count != cls.Rows)
            throw new InvalidDataException("Defect label count does not match defect bank rows.");

        return new ProductModel(root, manifest, memory, cls, center);
    }

    public DefectPrediction PredictDefect(float[] clsQuery, float[] centerQuery)
    {
        if (clsQuery.Length != DefectCls.Cols || centerQuery.Length != DefectCenter.Cols)
            throw new ArgumentException("DINOv2 query dimension does not match defect bank.");

        NormalizeInPlace(clsQuery);
        NormalizeInPlace(centerQuery);

        var clsScores = BestPerClass(DefectCls, clsQuery);
        var centerScores = BestPerClass(DefectCenter, centerQuery);
        var fused = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);

        foreach (var className in Manifest.Classes)
        {
            if (!clsScores.TryGetValue(className, out var clsScore) ||
                !centerScores.TryGetValue(className, out var centerScore))
            {
                throw new InvalidDataException($"Defect class has no exemplar: {className}");
            }
            fused[className] =
                Manifest.ClsWeight * clsScore + Manifest.CenterWeight * centerScore;
        }

        var ranked = fused.OrderByDescending(x => x.Value).ToArray();
        var top1 = ranked[0];
        var top2 = ranked.Length > 1 ? ranked[1] : default;
        var top2Score = ranked.Length > 1 ? top2.Value : float.NegativeInfinity;

        return new DefectPrediction(
            top1.Key,
            top1.Value,
            ranked.Length > 1 ? top2.Key : null,
            top2Score,
            ranked.Length > 1 ? top1.Value - top2Score : float.PositiveInfinity,
            fused
        );
    }

    private Dictionary<string, float> BestPerClass(BinaryMatrix bank, float[] query)
    {
        var result = Manifest.Classes.ToDictionary(
            x => x,
            _ => float.NegativeInfinity,
            StringComparer.OrdinalIgnoreCase
        );

        for (var row = 0; row < bank.Rows; row++)
        {
            var score = Dot(bank.Row(row), query);
            var label = Manifest.DefectLabels[row];
            if (!result.TryGetValue(label, out var current) || score > current)
                result[label] = score;
        }
        return result;
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector dimensions differ.");
        double sum = 0;
        for (var i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return (float)sum;
    }

    private static void NormalizeInPlace(float[] vector)
    {
        double normSq = 0;
        foreach (var value in vector)
            normSq += value * value;
        var norm = Math.Sqrt(normSq);
        if (norm <= 1e-12)
            return;
        var scale = (float)(1.0 / norm);
        for (var i = 0; i < vector.Length; i++)
            vector[i] *= scale;
    }
}
