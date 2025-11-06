# Download and extract the MovieLens 25M dataset into raw_data directory
# Run this script with the repository root as the working directory

import requests
import os
import zipfile

def download_file(url, output_path, chunk_size=1024):
    print("Downloading...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise exception for HTTP errors

    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # Filter out keep-alive chunks
                file.write(chunk)
    print("Download complete.")

def extract_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted files to {extract_dir}")
    # Remove the zip file after extraction
    os.remove(zip_path)
    print(f"Removed zip file: {zip_path}")

def download_text_file(url, output_path):
    print("Downloading README file...")
    response = requests.get(url)
    response.raise_for_status()  # Raise exception for HTTP errors

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(response.text)
    print(f"Downloaded text file to {output_path}")
    
def main():
    # URL to the MovieLens 25M dataset zip file provided by GroupLens
    movie_dataset_url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    # genome_dataset_url = "https://files.grouplens.org/datasets/tag-genome/tag-genome.zip"
    movie_readme_url = "https://files.grouplens.org/datasets/movielens/ml-25m-README.html"
    # genome_readme_url = "https://files.grouplens.org/datasets/tag-genome/README.html"
    extract_dir = "./raw_data"
    readme_file = os.path.join(extract_dir, "ml-25m-README.html")

    # Download the dataset
    print("Starting movie dataset download...")
    download_file(movie_dataset_url, "ml-25m.zip")

    # print("Starting genome dataset download...")
    # download_file(genome_dataset_url, "tag-genome.zip")

    # Create extract directory if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Extract the downloaded zip file
    print("Extracting files...")
    extract_zip("ml-25m.zip", extract_dir)
    # extract_zip("tag-genome.zip", extract_dir)

    # Download the README file and store it in the extract directory
    print("Downloading README file...")
    download_text_file(movie_readme_url, readme_file)
    # download_text_file(genome_readme_url, readme_file)

if __name__ == "__main__":
    main()