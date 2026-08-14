using IndustrialAnomaly.Runtime;

static int Usage()
{
    Console.WriteLine("IndustrialAnomaly.Console");
    Console.WriteLine();
    Console.WriteLine("Build product model:");
    Console.WriteLine("  build <engineDir> <outputRoot> <productName> <normalDir> <class=folder> [class=folder ...]");
    Console.WriteLine();
    Console.WriteLine("Inspect image:");
    Console.WriteLine("  inspect <engineDir> <productDir> <imagePath> <markedOutput> [anomalyThreshold]");
    Console.WriteLine();
    Console.WriteLine("Inspect folder:");
    Console.WriteLine("  inspect-folder <engineDir> <productDir> <imageDir> <outputDir> [anomalyThreshold]");
    return 2;
}

static string Csv(string? value)
{
    value ??= string.Empty;
    if (value.Contains(',') || value.Contains('"') || value.Contains('\n') || value.Contains('\r'))
        return $"\"{value.Replace("\"", "\"\"")}\"";
    return value;
}

static float? ParseThreshold(string[] commandArgs, int index)
{
    if (commandArgs.Length <= index)
        return null;
    return float.Parse(
        commandArgs[index],
        System.Globalization.CultureInfo.InvariantCulture
    );
}

if (args.Length == 0)
    return Usage();

try
{
    var command = args[0].Trim().ToLowerInvariant();
    if (command == "build")
    {
        if (args.Length < 6)
            return Usage();

        var engineDir = args[1];
        var outputRoot = args[2];
        var productName = args[3];
        var normalDir = args[4];
        var classes = new List<DefectClassDefinition>();

        foreach (var item in args.Skip(5))
        {
            var split = item.IndexOf('=');
            if (split <= 0 || split >= item.Length - 1)
                throw new ArgumentException($"Invalid class mapping: {item}. Use class=folder.");
            classes.Add(new DefectClassDefinition
            {
                Name = item[..split],
                ImageDirectory = item[(split + 1)..],
            });
        }

        using var featureEngine = new OnnxFeatureEngine(engineDir);
        var builder = new ProductModelBuilder(featureEngine);
        var definition = new ProductBuildDefinition
        {
            ProductName = productName,
            NormalImageDirectory = normalDir,
            DefectClasses = classes,
        };

        var manifest = builder.Build(definition, outputRoot, Console.WriteLine);
        Console.WriteLine();
        Console.WriteLine($"DONE product={manifest.ProductName}");
        Console.WriteLine($"memory rows={manifest.PatchCoreMemoryRows}");
        Console.WriteLine($"classes={string.Join(", ", manifest.Classes)}");
        return 0;
    }

    if (command == "inspect")
    {
        if (args.Length is < 5 or > 6)
            return Usage();

        var engineDir = args[1];
        var productDir = args[2];
        var imagePath = args[3];
        var markedOutput = args[4];
        var threshold = ParseThreshold(args, 5);

        using var featureEngine = new OnnxFeatureEngine(engineDir);
        var product = ProductModel.Load(productDir);
        var engine = new IndustrialAnomalyEngine(featureEngine, product);
        var result = engine.InspectFile(imagePath, markedOutput, threshold);

        Console.WriteLine($"PatchCore={result.PatchCoreAnomalyScore:F6}");
        Console.WriteLine($"decision={result.AnomalyDecision}");
        Console.WriteLine($"bbox={result.Bbox}");
        Console.WriteLine($"class={result.PredictedDefect ?? "-"}");
        Console.WriteLine(
            result.Top1Similarity.HasValue
                ? $"similarity={result.Top1Similarity.Value:F6}"
                : "similarity=-"
        );
        Console.WriteLine(
            result.Margin.HasValue
                ? $"margin={result.Margin.Value:F6}"
                : "margin=-"
        );
        Console.WriteLine($"final={result.FinalResult}");
        Console.WriteLine($"marked={Path.GetFullPath(markedOutput)}");
        return 0;
    }

    if (command == "inspect-folder")
    {
        if (args.Length is < 5 or > 6)
            return Usage();

        var engineDir = args[1];
        var productDir = args[2];
        var imageDir = Path.GetFullPath(args[3]);
        var outputDir = Path.GetFullPath(args[4]);
        var threshold = ParseThreshold(args, 5);

        if (!Directory.Exists(imageDir))
            throw new DirectoryNotFoundException(imageDir);

        var supported = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
        };
        var files = Directory.EnumerateFiles(imageDir)
            .Where(path => supported.Contains(Path.GetExtension(path)))
            .OrderBy(path => Path.GetFileName(path), StringComparer.OrdinalIgnoreCase)
            .ToList();
        if (files.Count == 0)
            throw new InvalidOperationException($"No supported images found in: {imageDir}");

        Directory.CreateDirectory(outputDir);
        var csvPath = Path.Combine(outputDir, "results.csv");

        using var featureEngine = new OnnxFeatureEngine(engineDir);
        var product = ProductModel.Load(productDir);
        var engine = new IndustrialAnomalyEngine(featureEngine, product);
        using var writer = new StreamWriter(csvPath, false, new System.Text.UTF8Encoding(true));
        writer.WriteLine("image,patchcore,decision,x,y,width,height,class,similarity,margin,final,marked");

        for (var index = 0; index < files.Count; index++)
        {
            var file = files[index];
            var stem = Path.GetFileNameWithoutExtension(file);
            var markedPath = Path.Combine(outputDir, $"{stem}_marked.jpg");
            var result = engine.InspectFile(file, markedPath, threshold);

            var bbox = result.Bbox;
            var similarity = result.Top1Similarity?.ToString(
                "F6", System.Globalization.CultureInfo.InvariantCulture
            ) ?? string.Empty;
            var margin = result.Margin?.ToString(
                "F6", System.Globalization.CultureInfo.InvariantCulture
            ) ?? string.Empty;

            writer.WriteLine(string.Join(",", new[]
            {
                Csv(Path.GetFileName(file)),
                result.PatchCoreAnomalyScore.ToString("F6", System.Globalization.CultureInfo.InvariantCulture),
                Csv(result.AnomalyDecision),
                bbox?.X.ToString() ?? string.Empty,
                bbox?.Y.ToString() ?? string.Empty,
                bbox?.Width.ToString() ?? string.Empty,
                bbox?.Height.ToString() ?? string.Empty,
                Csv(result.PredictedDefect),
                similarity,
                margin,
                Csv(result.FinalResult),
                Csv(markedPath),
            }));
            writer.Flush();

            Console.WriteLine(
                $"[{index + 1}/{files.Count}] {Path.GetFileName(file)} " +
                $"PatchCore={result.PatchCoreAnomalyScore:F4} " +
                $"class={result.PredictedDefect ?? "-"} " +
                $"sim={(result.Top1Similarity.HasValue ? result.Top1Similarity.Value.ToString("F4") : "-")} " +
                $"margin={(result.Margin.HasValue ? result.Margin.Value.ToString("F4") : "-")}"
            );
        }

        Console.WriteLine();
        Console.WriteLine($"DONE images={files.Count}");
        Console.WriteLine($"output={outputDir}");
        Console.WriteLine($"csv={csvPath}");
        return 0;
    }

    return Usage();
}
catch (Exception ex)
{
    Console.Error.WriteLine(ex);
    return 1;
}
