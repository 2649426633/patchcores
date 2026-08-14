using OpenCvSharp;

namespace IndustrialAnomaly.Runtime;

public readonly record struct TileWindow(int X1, int Y1, int X2, int Y2)
{
    public int Width => X2 - X1;
    public int Height => Y2 - Y1;
    public Rect Rect => new(X1, Y1, Width, Height);
}

public sealed record AnomalyRegion(
    Rect Bbox,
    float RankScore,
    float TileScore,
    float Evidence,
    float Peak,
    int Area,
    int TileIndex,
    TileWindow Tile,
    bool TouchesTileBorder,
    int MergedDetections = 1
);

public sealed record TiledInspectionResult(
    float AnomalyScore,
    IReadOnlyList<AnomalyRegion> Regions,
    int TileCount
);

public static class Tiling
{
    public static IReadOnlyList<TileWindow> ComputeWindows(
        int width,
        int height,
        float tileFraction = 0.75f,
        float overlap = 0.25f
    )
    {
        if (tileFraction is < 0.2f or > 1.0f)
            throw new ArgumentOutOfRangeException(nameof(tileFraction));
        if (overlap is < 0.0f or >= 0.9f)
            throw new ArgumentOutOfRangeException(nameof(overlap));

        var tileSize = Math.Max(32, (int)Math.Round(Math.Min(width, height) * tileFraction));
        tileSize = Math.Min(tileSize, Math.Min(width, height));
        var xs = AxisStarts(width, tileSize, overlap);
        var ys = AxisStarts(height, tileSize, overlap);

        var windows = new List<TileWindow>(xs.Count * ys.Count);
        foreach (var y in ys)
        foreach (var x in xs)
            windows.Add(new TileWindow(x, y, x + tileSize, y + tileSize));
        return windows;
    }

    private static List<int> AxisStarts(int length, int tileSize, float overlap)
    {
        if (tileSize >= length)
            return [0];

        var stride = Math.Max(1.0, tileSize * (1.0 - overlap));
        var count = Math.Max(2, (int)Math.Ceiling((length - tileSize) / stride) + 1);
        var maxStart = length - tileSize;
        var starts = new SortedSet<int>();
        for (var i = 0; i < count; i++)
        {
            var value = count == 1 ? 0.0 : maxStart * i / (double)(count - 1);
            starts.Add((int)Math.Round(value));
        }
        starts.Add(maxStart);
        return starts.ToList();
    }
}

public sealed class PatchCoreTiledInspector
{
    private readonly OnnxFeatureEngine _engine;

    public PatchCoreTiledInspector(OnnxFeatureEngine engine)
    {
        _engine = engine;
    }

    public TiledInspectionResult Inspect(
        Mat image,
        BinaryMatrix memoryBank,
        float tileFraction = 0.75f,
        float overlap = 0.25f,
        float relativeThreshold = 0.70f,
        int minArea = 8,
        int maxRegionsPerTile = 4,
        int maxRegions = 8,
        float minGlobalRatio = 0.55f,
        float mergeIou = 0.15f
    )
    {
        if (image.Empty())
            throw new ArgumentException("Input image is empty.", nameof(image));

        var windows = Tiling.ComputeWindows(image.Width, image.Height, tileFraction, overlap);
        var candidates = new List<AnomalyRegion>();
        var bestTileScore = float.NegativeInfinity;

        for (var tileIndex = 0; tileIndex < windows.Count; tileIndex++)
        {
            var window = windows[tileIndex];
            using var tile = new Mat(image, window.Rect).Clone();
            var patchResult = _engine.RunPatchCore(tile, memoryBank);
            var tileScore = patchResult.PatchScores.Max();
            bestTileScore = Math.Max(bestTileScore, tileScore);

            using var anomalyMap = BuildAnomalyMap(patchResult.PatchScores);
            var regions = ExtractRegions(
                anomalyMap,
                relativeThreshold,
                minArea,
                maxRegionsPerTile
            );

            var sx = window.Width / (float)anomalyMap.Width;
            var sy = window.Height / (float)anomalyMap.Height;
            foreach (var local in regions)
            {
                var bbox = new Rect(
                    (int)Math.Round(window.X1 + local.Bbox.X * sx),
                    (int)Math.Round(window.Y1 + local.Bbox.Y * sy),
                    Math.Max(1, (int)Math.Round(local.Bbox.Width * sx)),
                    Math.Max(1, (int)Math.Round(local.Bbox.Height * sy))
                );
                bbox = ClipRect(bbox, image.Width, image.Height);
                var rank = tileScore * (0.50f + 0.50f * local.Evidence);
                candidates.Add(new AnomalyRegion(
                    bbox,
                    rank,
                    tileScore,
                    local.Evidence,
                    local.Peak,
                    local.Area,
                    tileIndex,
                    window,
                    local.TouchesBorder
                ));
            }
        }

        var merged = MergeOverlapping(candidates, mergeIou);
        if (merged.Count > 0)
        {
            var top = merged[0].RankScore;
            merged = merged
                .Where(r => r.RankScore >= top * minGlobalRatio)
                .Take(maxRegions)
                .ToList();
        }

        return new TiledInspectionResult(bestTileScore, merged, windows.Count);
    }

    public Mat CropSquareWithMargin(Mat image, Rect bbox, float margin = 0.50f)
    {
        var bw = Math.Max(1, bbox.Width);
        var bh = Math.Max(1, bbox.Height);
        var cx = bbox.X + bbox.Width / 2.0;
        var cy = bbox.Y + bbox.Height / 2.0;
        var side = Math.Max(8, (int)Math.Ceiling(Math.Max(bw, bh) * (1.0 + 2.0 * margin)));

        var left = (int)Math.Floor(cx - side / 2.0);
        var top = (int)Math.Floor(cy - side / 2.0);
        var right = left + side;
        var bottom = top + side;

        var srcLeft = Math.Max(0, left);
        var srcTop = Math.Max(0, top);
        var srcRight = Math.Min(image.Width, right);
        var srcBottom = Math.Min(image.Height, bottom);

        var fill = MedianBorderColor(image);
        var output = new Mat(side, side, MatType.CV_8UC3, fill);
        if (srcRight > srcLeft && srcBottom > srcTop)
        {
            using var source = new Mat(
                image,
                new Rect(srcLeft, srcTop, srcRight - srcLeft, srcBottom - srcTop)
            );
            using var destination = new Mat(
                output,
                new Rect(srcLeft - left, srcTop - top, source.Width, source.Height)
            );
            source.CopyTo(destination);
        }
        return output;
    }

    private Mat BuildAnomalyMap(float[] patchScores)
    {
        var grid = _engine.Manifest.PatchCore.PatchGrid;
        if (patchScores.Length != grid[0] * grid[1])
            throw new InvalidDataException("Patch score count does not match configured grid.");

        var patchMap = new Mat(grid[0], grid[1], MatType.CV_32FC1);
        for (var y = 0; y < grid[0]; y++)
        for (var x = 0; x < grid[1]; x++)
            patchMap.Set(y, x, patchScores[y * grid[1] + x]);

        var size = _engine.Manifest.PatchCore.InputShape[2];
        var resized = new Mat();
        Cv2.Resize(patchMap, resized, new Size(size, size), 0, 0, InterpolationFlags.Linear);
        patchMap.Dispose();

        var smoothed = new Mat();
        Cv2.GaussianBlur(resized, smoothed, new Size(0, 0), 4.0, 4.0);
        resized.Dispose();
        return smoothed;
    }

    private static List<LocalRegion> ExtractRegions(
        Mat anomalyMap,
        float relativeThreshold,
        int minArea,
        int maxRegions
    )
    {
        using var normalized = Normalize(anomalyMap);
        using var smooth = new Mat();
        Cv2.GaussianBlur(normalized, smooth, new Size(0, 0), 0.8, 0.8);

        using var binaryFloat = new Mat();
        Cv2.Threshold(smooth, binaryFloat, relativeThreshold, 1.0, ThresholdTypes.Binary);
        using var binary = new Mat();
        binaryFloat.ConvertTo(binary, MatType.CV_8UC1, 255.0);

        using var kernel = Mat.Ones(3, 3, MatType.CV_8UC1);
        using var closed = new Mat();
        Cv2.MorphologyEx(binary, closed, MorphTypes.Close, kernel);

        using var labels = new Mat();
        using var stats = new Mat();
        using var centroids = new Mat();
        var count = Cv2.ConnectedComponentsWithStats(
            closed,
            labels,
            stats,
            centroids,
            PixelConnectivity.Connectivity8,
            MatType.CV_32S
        );

        var result = new List<LocalRegion>();
        for (var label = 1; label < count; label++)
        {
            var area = stats.At<int>(label, (int)ConnectedComponentsTypes.Area);
            if (area < minArea)
                continue;

            var x = stats.At<int>(label, (int)ConnectedComponentsTypes.Left);
            var y = stats.At<int>(label, (int)ConnectedComponentsTypes.Top);
            var w = stats.At<int>(label, (int)ConnectedComponentsTypes.Width);
            var h = stats.At<int>(label, (int)ConnectedComponentsTypes.Height);

            var values = new List<float>(area);
            for (var yy = y; yy < y + h; yy++)
            for (var xx = x; xx < x + w; xx++)
            {
                if (labels.At<int>(yy, xx) == label)
                    values.Add(smooth.At<float>(yy, xx));
            }
            if (values.Count == 0)
                continue;

            values.Sort();
            var peak = values[^1];
            var q90Index = Math.Clamp((int)Math.Floor(0.90 * (values.Count - 1)), 0, values.Count - 1);
            var q90 = values[q90Index];
            var mean = values.Average();
            var evidence = 0.55f * peak + 0.30f * q90 + 0.15f * mean;
            var touches = x <= 1 || y <= 1 || x + w >= smooth.Width - 1 || y + h >= smooth.Height - 1;
            if (touches)
                evidence *= 0.92f;

            result.Add(new LocalRegion(new Rect(x, y, w, h), area, peak, q90, mean, evidence, touches));
        }

        return result
            .OrderByDescending(r => r.Evidence)
            .ThenByDescending(r => r.Peak)
            .ThenByDescending(r => r.Q90)
            .ThenByDescending(r => r.Area)
            .Take(Math.Max(0, maxRegions))
            .ToList();
    }

    private static Mat Normalize(Mat source)
    {
        Cv2.MinMaxLoc(source, out var min, out var max);
        var output = new Mat();
        if (max - min < 1e-12)
        {
            output = Mat.Zeros(source.Rows, source.Cols, MatType.CV_32FC1);
            return output;
        }
        source.ConvertTo(output, MatType.CV_32FC1, 1.0 / (max - min), -min / (max - min));
        return output;
    }

    private static List<AnomalyRegion> MergeOverlapping(
        IEnumerable<AnomalyRegion> source,
        float iouThreshold
    )
    {
        var merged = new List<AnomalyRegion>();
        foreach (var region in source.OrderByDescending(r => r.RankScore))
        {
            var index = merged.FindIndex(x => IoU(region.Bbox, x.Bbox) >= iouThreshold);
            if (index < 0)
            {
                merged.Add(region);
                continue;
            }

            var existing = merged[index];
            merged[index] = existing with
            {
                Bbox = Union(existing.Bbox, region.Bbox),
                RankScore = Math.Max(existing.RankScore, region.RankScore),
                TileScore = Math.Max(existing.TileScore, region.TileScore),
                Evidence = Math.Max(existing.Evidence, region.Evidence),
                Peak = Math.Max(existing.Peak, region.Peak),
                MergedDetections = existing.MergedDetections + 1,
            };
        }
        return merged.OrderByDescending(r => r.RankScore).ToList();
    }

    private static float IoU(Rect a, Rect b)
    {
        var x1 = Math.Max(a.Left, b.Left);
        var y1 = Math.Max(a.Top, b.Top);
        var x2 = Math.Min(a.Right, b.Right);
        var y2 = Math.Min(a.Bottom, b.Bottom);
        var iw = Math.Max(0, x2 - x1);
        var ih = Math.Max(0, y2 - y1);
        var intersection = iw * ih;
        if (intersection <= 0)
            return 0f;
        var union = Math.Max(1, a.Width * a.Height + b.Width * b.Height - intersection);
        return intersection / (float)union;
    }

    private static Rect Union(Rect a, Rect b)
    {
        var x1 = Math.Min(a.Left, b.Left);
        var y1 = Math.Min(a.Top, b.Top);
        var x2 = Math.Max(a.Right, b.Right);
        var y2 = Math.Max(a.Bottom, b.Bottom);
        return new Rect(x1, y1, x2 - x1, y2 - y1);
    }

    private static Rect ClipRect(Rect r, int width, int height)
    {
        var x1 = Math.Clamp(r.Left, 0, Math.Max(0, width - 1));
        var y1 = Math.Clamp(r.Top, 0, Math.Max(0, height - 1));
        var x2 = Math.Clamp(r.Right, x1 + 1, width);
        var y2 = Math.Clamp(r.Bottom, y1 + 1, height);
        return new Rect(x1, y1, x2 - x1, y2 - y1);
    }

    private static Scalar MedianBorderColor(Mat image)
    {
        var b = new List<byte>();
        var g = new List<byte>();
        var r = new List<byte>();

        void AddPixel(Vec3b p)
        {
            b.Add(p.Item0);
            g.Add(p.Item1);
            r.Add(p.Item2);
        }

        for (var x = 0; x < image.Width; x++)
        {
            AddPixel(image.At<Vec3b>(0, x));
            AddPixel(image.At<Vec3b>(image.Height - 1, x));
        }
        for (var y = 0; y < image.Height; y++)
        {
            AddPixel(image.At<Vec3b>(y, 0));
            AddPixel(image.At<Vec3b>(y, image.Width - 1));
        }

        static byte Median(List<byte> values)
        {
            values.Sort();
            return values[values.Count / 2];
        }

        return new Scalar(Median(b), Median(g), Median(r));
    }

    private sealed record LocalRegion(
        Rect Bbox,
        int Area,
        float Peak,
        float Q90,
        float Mean,
        float Evidence,
        bool TouchesBorder
    );
}
