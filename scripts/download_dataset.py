from src.data_pipeline import download_public_sample


if __name__ == "__main__":
    path = download_public_sample(refresh=True)
    print(f"Dataset downloaded to: {path}")
