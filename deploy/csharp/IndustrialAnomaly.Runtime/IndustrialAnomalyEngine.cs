using OpenCvSharp;

namespace IndustrialAnomaly.Runtime;

public sealed record InspectionPrediction(
    float PatchCoreAnomalyScore,
    string AnomalyDecision,
    Rect? Bbox,
    string? PredictedDefect,
    float? Top1Similarity,
    float? Margin,
    string FinalResult
);

public sealed class IndustrialAnomalyEngine
{
    private readonly OnnxFeatureEngine _featureEngine;
    private readonly ProductModel _productModel;
    private readonly PatchCoreTiledInspector _inspector;

    public IndustrialAnomalyEngine(OnnxFeatureEngine featureEngine, ProductModel productModel)
    {
        _featureEngine = featureEngine;
        _productModel = productModel;
        _inspector = new PatchCoreTiledInspector(featureEngine);

        if (_productModel.PatchCoreMemory.Cols != _featureEngine.Manifest.PatchCore.EmbeddingDim)
            throw new InvalidDataException("Product PatchCore memory dimension is incompatible with ONNX engine.");
        if (_productModel.DefectCls.Cols != _featureEngine.Manifest.DINOv2.EmbeddingDim)
            throw new InvalidDataException("Product DINOv2 bank dimension is incompatible with ONNX engine.");
    }

    public InspectionPrediction Inspect(Mat image, float? anomalyThreshold = null)
    {
        var cfg = _productModel.Manifest;
        var result = _inspector.Inspect(
            image,
            _productModel.PatchCoreMemory,
            cfg.TileFraction,
            cfg.TileOverlap,
            cfg.BboxRelativeThreshold
        );

        if (result.Regions.Count == 0)
        {
            var decision = anomalyThreshold.HasValue && result.AnomalyScore < anomalyThreshold.Value
                ? "PASS"
                : anomalyThreshold.HasValue ? "NG" : "UNCALIBRATED";
            var final = decision == "PASS" ? "PASS" : "NO_LOCALIZED_REGION";
            return new InspectionPrediction(
                result.AnomalyScore,
                decision,
                null,
                null,
                null,
                null,
                final
            );
        }

        var primary = result.Regions[0];
        using var roi = _inspector.CropSquareWithMargin(image, primary.Bbox, cfg.RoiMargin);
        var dino = _featureEngine.ExtractDinoEmbeddings(roi);
        var defect = _productModel.PredictDefect(dino.Cls, dino.Center);

        string anomalyDecision;
        string finalResult;
        if (!anomalyThreshold.HasValue)
        {
            anomalyDecision = "UNCALIBRATED";
            finalResult = $"KNOWN_DEFECT_CANDIDATE: {defect.PredictedClass}";
        }
        else if (result.AnomalyScore < anomalyThreshold.Value)
        {
            anomalyDecision = "PASS";
            finalResult = "PASS";
        }
        else
        {
            anomalyDecision = "NG";
            finalResult = $"NG: {defect.PredictedClass}";
        }

        return new InspectionPrediction(
            result.AnomalyScore,
            anomalyDecision,
            primary.Bbox,
            defect.PredictedClass,
            defect.Top1Similarity,
            defect.Margin,
            finalResult
        );
    }

    public InspectionPrediction InspectFile(
        string imagePath,
        string? markedOutputPath = null,
        float? anomalyThreshold = null
    )
    {
        using var image = Cv2.ImRead(imagePath, ImreadModes.Color);
        if (image.Empty())
            throw new FileNotFoundException($"Cannot read image: {imagePath}", imagePath);

        var result = Inspect(image, anomalyThreshold);
        if (!string.IsNullOrWhiteSpace(markedOutputPath))
            SaveMarkedImage(image, result, markedOutputPath!);
        return result;
    }

    public static void SaveMarkedImage(Mat source, InspectionPrediction result, string outputPath)
    {
        using var canvas = source.Clone();
        var label = result.FinalResult;

        if (result.Bbox is Rect bbox)
        {
            var thickness = Math.Max(2, (int)Math.Round(Math.Min(canvas.Width, canvas.Height) / 350.0));
            Cv2.Rectangle(canvas, bbox, Scalar.Red, thickness);
            var text = result.Top1Similarity.HasValue
                ? $"{label} | PatchCore={result.PatchCoreAnomalyScore:F3} | sim={result.Top1Similarity:F3}"
                : $"{label} | PatchCore={result.PatchCoreAnomalyScore:F3}";
            var origin = new Point(bbox.X, Math.Max(30, bbox.Y - 10));
            Cv2.PutText(
                canvas,
                text,
                origin,
                HersheyFonts.HersheySimplex,
                Math.Max(0.6, Math.Min(canvas.Width, canvas.Height) / 1500.0),
                Scalar.White,
                Math.Max(1, thickness - 1),
                LineTypes.AntiAlias
            );
        }
        else
        {
            Cv2.PutText(
                canvas,
                label,
                new Point(20, 50),
                HersheyFonts.HersheySimplex,
                1.0,
                Scalar.White,
                2,
                LineTypes.AntiAlias
            );
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
        Cv2.ImWrite(outputPath, canvas);
    }
}
