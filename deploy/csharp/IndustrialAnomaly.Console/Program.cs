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
    return 2;
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
        float? threshold = null;
        if (args.Length == 6)
            threshold = float.Parse(args[5], System.Globalization.CultureInfo.InvariantCulture);

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

    return Usage();
}
catch (Exception ex)
{
    Console.Error.WriteLine(ex);
    return 1;
}
