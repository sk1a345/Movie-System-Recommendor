import zipfile
import os
import pandas as pd

# Folder containing the zip files
zip_folder = r'C:\Users\HP\OneDrive\Machine_Learning_projects\Movie-System-Recommender'

# Zip files
zip_files = [
    'tmdb_5000_credits.csv.zip',
    'tmdb_5000_movies.csv.zip'
]

# Folder to extract files to
extract_folder = os.path.join(zip_folder, 'data')
os.makedirs(extract_folder, exist_ok=True)

# Extract each zip file
for zip_file in zip_files:
    zip_path = os.path.join(zip_folder, zip_file)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print(f"Extracted {zip_file}")

# Paths to extracted CSVs
credits_csv = os.path.join(extract_folder, 'tmdb_5000_credits.csv.zip')
movies_csv = os.path.join(extract_folder, 'tmdb_5000_movies.csv.zip')

# Load CSV files
credits_df = pd.read_csv(credits_csv)
movies_df = pd.read_csv(movies_csv)

# Show first few rows to verify
print("Credits DataFrame:")
print(credits_df.head())

print("\nMovies DataFrame:")
print(movies_df.head())
