import kagglehub

path = kagglehub.dataset_download(
    "dilwong/flightprices",
    output_dir="data/raw"
)

print("Dataset downloaded to:", path)