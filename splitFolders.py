import splitfolders

# Path to the raw images folder (your current location)
input_folder = "./archive/raw-img"

# Output folder where split datasets will be saved
output_folder = "./data"

# Split with 70% for training, 15% for validation, and 15% for testing
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .15, .15))

