using OpenCvSharp;

namespace IndustrialAnomaly.Runtime;

public sealed class ProductModelBuilder
{
    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    };

    private readonly OnnxFeatureEngine _engine;
    private readonly PatchCoreTiledInspector _inspector;

    public ProductModelBuilder(OnnxFeatureEngine engine)
    {
        _engine = engine;
        _inspector = new PatchCoreTiledInspector(engine);
    }

    public ProductModelManifest Build(
        ProductBuildDefinition definition,
        string outputRoot,
        Action<string>? progress = null,
        int randomSeed = 20260814
    )
    {
        ValidateDefinition(definition);

        var productDir = Path.Combine(Path.GetFullPath(outputRoot), SafeName(definition.ProductName));
        Directory.CreateDirectory(productDir);
        progress?.Invoke($"Product model: {productDir}");

        var memory = BuildNormalMemory(definition, progress, randomSeed);
        var memoryFile = "patchcore_memory.bin";
        memory.Save(Path.Combine(productDir, memoryFile));
        progress?.Invoke($"PatchCore memory: {memory.Rows} x {memory.Cols}");

        var defect = BuildDefectBanks(definition, memory, productDir, progress);
        var clsFile = "defect_cls.bin";
        var centerFile = "defect_center.bin";
        defect.Cls.Save(Path.Combine(productDir, clsFile));
        defect.Center.Save(Path.Combine(productDir, centerFile));

        var manifest = new ProductModelManifest
        {
            ProductName = definition.ProductName,
            PatchCoreMemoryFile = memoryFile,
            DefectClsFile = clsFile,
            DefectCenterFile = centerFile,
            DefectLabels = defect.Labels,
            Classes = definition.DefectClasses.Select(x => x.Name).Distinct().ToArray(),
            TileFraction = definition.TileFraction,
            TileOverlap = definition.TileOverlap,
            CoresetRatio = definition.CoresetRatio,
            PatchCoreMemoryRows = memory.Rows,
            PatchCoreMemoryStrategy = "bounded_reservoir_v1",
            BboxRelativeThreshold = definition.BboxRelativeThreshold,
            RoiMargin = definition.RoiMargin,
            ClsWeight = definition.ClsWeight,
            CenterWeight = definition.CenterWeight,
        };
        manifest.Save(Path.Combine(productDir, "product_model.json"));

        progress?.Invoke($"Defect exemplars: {defect.Labels.Count}");
        progress?.Invoke("Product model build finished.");
        return manifest;
    }

    private BinaryMatrix BuildNormalMemory(
        ProductBuildDefinition definition,
        Action<string>? progress,
        int randomSeed
    )
    {
        var files = EnumerateImages(definition.NormalImageDirectory);
        if (files.Count == 0)
            throw new InvalidOperationException("Normal image folder contains no supported images.");

        var patchRowsPerTile = _engine.Manifest.PatchCore.OutputShape[1];
        long totalRows = 0;
        foreach (var file in files)
        {
            using var image = Cv2.ImRead(file, ImreadModes.Color);
            if (image.Empty())
                throw new InvalidDataException($"Cannot read normal image: {file}");
            var tileCount = Tiling.ComputeWindows(
                image.Width,
                image.Height,
                definition.TileFraction,
                definition.TileOverlap
            ).Count;
            totalRows += (long)tileCount * patchRowsPerTile;
        }

        var ratioTarget = Math.Max(1L, (long)Math.Round(totalRows * definition.CoresetRatio));
        var capacity = checked((int)Math.Min(definition.MaxPatchCoreMemoryRows, ratioTarget));
        progress?.Invoke(
            $"Normal images={files.Count}, source patch rows={totalRows}, memory target={capacity}"
        );

        var dim = _engine.Manifest.PatchCore.EmbeddingDim;
        var reservoir = new float[checked(capacity * dim)];
        long seen = 0;
        var kept = 0;
        var random = new Random(randomSeed);

        for (var imageIndex = 0; imageIndex < files.Count; imageIndex++)
        {
            var file = files[imageIndex];
            using var image = Cv2.ImRead(file, ImreadModes.Color);
            var windows = Tiling.ComputeWindows(
                image.Width,
                image.Height,
                definition.TileFraction,
                definition.TileOverlap
            );

            foreach (var window in windows)
            {
                using var tile = new Mat(image, window.Rect).Clone();
                var embeddings = _engine.ExtractPatchCoreEmbeddings(tile);
                for (var row = 0; row < embeddings.Rows; row++)
                {
                    var destination = -1;
                    if (kept < capacity)
                    {
                        destination = kept++;
                    }
                    else
                    {
                        var candidate = random.NextInt64(seen + 1);
                        if (candidate < capacity)
                            destination = (int)candidate;
                    }

                    if (destination >= 0)
                    {
                        embeddings.Row(row).CopyTo(
                            reservoir.AsSpan(destination * dim, dim)
                        );
                    }
                    seen++;
                }
            }

            progress?.Invoke(
                $"Normal [{imageIndex + 1}/{files.Count}] {Path.GetFileName(file)}"
            );
        }

        if (kept == 0)
            throw new InvalidOperationException("No PatchCore embeddings were generated.");

        if (kept == capacity)
            return new BinaryMatrix(kept, dim, reservoir);

        var compact = new float[checked(kept * dim)];
        reservoir.AsSpan(0, compact.Length).CopyTo(compact);
        return new BinaryMatrix(kept, dim, compact);
    }

    private DefectBuildResult BuildDefectBanks(
        ProductBuildDefinition definition,
        BinaryMatrix memory,
        string productDir,
        Action<string>? progress
    )
    {
        var dim = _engine.Manifest.DINOv2.EmbeddingDim;
        var clsRows = new List<float[]>();
        var centerRows = new List<float[]>();
        var labels = new List<string>();
        var supportRoot = Path.Combine(productDir, "support_rois");

        foreach (var defectClass in definition.DefectClasses)
        {
            var files = EnumerateImages(defectClass.ImageDirectory);
            if (files.Count == 0)
                throw new InvalidOperationException(
                    $"Defect class '{defectClass.Name}' has no supported images."
                );

            var classOutput = Path.Combine(supportRoot, SafeName(defectClass.Name));
            Directory.CreateDirectory(classOutput);

            for (var index = 0; index < files.Count; index++)
            {
                var file = files[index];
                using var image = Cv2.ImRead(file, ImreadModes.Color);
                if (image.Empty())
                    throw new InvalidDataException($"Cannot read defect image: {file}");

                var inspection = _inspector.Inspect(
                    image,
                    memory,
                    definition.TileFraction,
                    definition.TileOverlap,
                    definition.BboxRelativeThreshold
                );
                if (inspection.Regions.Count == 0)
                {
                    progress?.Invoke(
                        $"WARN {defectClass.Name}/{Path.GetFileName(file)}: no PatchCore region; skipped"
                    );
                    continue;
                }

                var primary = inspection.Regions[0];
                using var roi = _inspector.CropSquareWithMargin(
                    image,
                    primary.Bbox,
                    definition.RoiMargin
                );
                var embeddings = _engine.ExtractDinoEmbeddings(roi);
                if (embeddings.Cls.Length != dim || embeddings.Center.Length != dim)
                    throw new InvalidDataException("Unexpected DINOv2 embedding dimension.");

                clsRows.Add(embeddings.Cls);
                centerRows.Add(embeddings.Center);
                labels.Add(defectClass.Name);

                var roiPath = Path.Combine(
                    classOutput,
                    $"{Path.GetFileNameWithoutExtension(file)}_roi.png"
                );
                Cv2.ImWrite(roiPath, roi);

                progress?.Invoke(
                    $"Defect {defectClass.Name} [{index + 1}/{files.Count}] " +
                    $"score={inspection.AnomalyScore:F4} bbox={primary.Bbox}"
                );
            }
        }

        if (labels.Count == 0)
            throw new InvalidOperationException("No defect exemplar was successfully created.");

        return new DefectBuildResult(
            StackRows(clsRows, dim),
            StackRows(centerRows, dim),
            labels
        );
    }

    private static BinaryMatrix StackRows(List<float[]> rows, int dim)
    {
        var data = new float[checked(rows.Count * dim)];
        for (var i = 0; i < rows.Count; i++)
        {
            if (rows[i].Length != dim)
                throw new InvalidDataException("Embedding row dimension mismatch.");
            rows[i].CopyTo(data, i * dim);
        }
        return new BinaryMatrix(rows.Count, dim, data);
    }

    private static List<string> EnumerateImages(string directory)
    {
        var full = Path.GetFullPath(directory);
        if (!Directory.Exists(full))
            throw new DirectoryNotFoundException(full);

        return Directory.EnumerateFiles(full)
            .Where(x => ImageExtensions.Contains(Path.GetExtension(x)))
            .OrderBy(x => Path.GetFileName(x), StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static void ValidateDefinition(ProductBuildDefinition definition)
    {
        if (string.IsNullOrWhiteSpace(definition.ProductName))
            throw new ArgumentException("ProductName is required.");
        if (definition.DefectClasses.Count == 0)
            throw new ArgumentException("At least one defect class is required.");
        if (definition.CoresetRatio is <= 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(definition.CoresetRatio));
        if (definition.MaxPatchCoreMemoryRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(definition.MaxPatchCoreMemoryRows));

        var duplicate = definition.DefectClasses
            .GroupBy(x => x.Name, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault(g => g.Count() > 1);
        if (duplicate is not null)
            throw new ArgumentException($"Duplicate defect class name: {duplicate.Key}");
    }

    private static string SafeName(string value)
    {
        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        var chars = value.Trim().Select(c => invalid.Contains(c) ? '_' : c).ToArray();
        var result = new string(chars);
        if (string.IsNullOrWhiteSpace(result))
            throw new ArgumentException("Name cannot be converted to a valid directory name.");
        return result;
    }

    private sealed record DefectBuildResult(
        BinaryMatrix Cls,
        BinaryMatrix Center,
        IReadOnlyList<string> Labels
    );
}
