import splitfolders

splitfolders.ratio(
    "../processed-images/rede",
    output="../processed-images/output",
    seed=42,
    ratio=(0.7, 0.2, 0.1)
)