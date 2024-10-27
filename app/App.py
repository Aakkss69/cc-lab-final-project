import os
import sys
import random
import boto3
from smart_open import open
import numpy as np
import torch
import streamlit as st
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add the parent directory to the system path to import config
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now you can import config
from config import config
from model import Classifier  # Import your classifier model

# Load S3 configuration
bucket_name = config.S3_DATA_BUCKET
test_prefix = config.S3_DATA_PREFIX + "test/"

# Initialize S3 client
s3 = boto3.client('s3')

# Load the model
#model = Classifier.load_from_checkpoint(
#    "/home/ec2-user/final_proj/streamlit_image_classification/logs/lightning_logs/version_0/checkpoints/streamlit-image-classification-epoch=00-val_loss=0.00.ckpt"
#)

bucket_name = "cc-lab-final-proj-bucket"
s3_key = "models/animal_classficiation_model.ckpt"
local_checkpoint_path = "/tmp/animal_classification_model.ckpt"

# Download the checkpoint from S3 to a temporary local path
s3 = boto3.client('s3')
s3.download_file(bucket_name, s3_key, local_checkpoint_path)

# Load the model from the local checkpoint file
model = Classifier.load_from_checkpoint(local_checkpoint_path)
model.eval()


# Define labels
#labels = [
#    "cane",      # dog
#    "cavallo",   # horse
#    "elefante",  # elephant
#    "farfalla",  # butterfly
#    "gallina",   # chicken
#    "gatto",     # cat
#    "mucca",     # cow
#    "pecora",    # sheep
#    "ragno",     # spider
#    "scoiattolo" # squirrel
#]

labels = [
    "dog",      # dog
    "horse",   # horse
    "elephant",  # elephant
    "butterfly",  # butterfly
    "chicken",   # chicken
    "cat",     # cat
    "cow",     # cow
    "sheep",    # sheep
    "spider",     # spider
    "squirrel" # squirrel
]

# Function to list and sample 10 images from each class in 'data/test/'
def get_sample_images(bucket_name, prefix, sample_size=10):
    sample_images = {}
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    
    for class_folder in response.get('CommonPrefixes', []):
        class_name = class_folder['Prefix'].split('/')[-2]
        class_response = s3.list_objects_v2(Bucket=bucket_name, Prefix=class_folder['Prefix'])
        image_keys = [item['Key'] for item in class_response.get('Contents', []) if item['Key'].lower().endswith(('.jpeg', '.jpg', '.png'))]
        sampled_images = random.sample(image_keys, min(sample_size, len(image_keys)))
        for idx, image_key in enumerate(sampled_images):
            sample_images[f"{class_name}_{idx+1}"] = image_key
    return sample_images

# Fetch 10 sample images from each subdirectory in 'data/test/'
sample_images = get_sample_images(bucket_name, test_prefix)

# Function to load an image from S3
def load_image_from_s3(bucket_name, image_key):
    with open(f"s3://{bucket_name}/{image_key}", 'rb') as file:
        image = Image.open(file).convert("RGB")
    return image

# Preprocess function
def preprocess(image):
    image = np.array(image)
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    image = transform(image=image)["image"]
    return image

# Define the function to make predictions on an image
def predict(image):
    try:
        image = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            topk_prob, topk_label = torch.topk(probabilities, 3)
            predictions = [(topk_prob[i].item(), topk_label[i].item()) for i in range(topk_prob.size(0))]
            return predictions
    except Exception as e:
        print(f"Error predicting image: {e}")
        return []

# Define the Streamlit app
def app():
    st.title("Animal-10 Image Classification")

    # Add a file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Select box for choosing from sample images in S3
    sample = st.selectbox("Or choose from sample images:", list(sample_images.keys()))

    # If an image is uploaded, make a prediction on it
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        predictions = predict(image)

    # If a sample image is chosen, load it from S3 and make a prediction
    elif sample:
        image_key = sample_images[sample]
        image = load_image_from_s3(bucket_name, image_key)
        st.image(image, caption=sample.capitalize() + " Image.", use_column_width=True)
        predictions = predict(image)

    # Show the top 3 predictions with their probabilities
    if predictions:
        st.write("Top 3 predictions:")
        for i, (prob, label) in enumerate(predictions):
            st.write(f"{i+1}. {labels[label]} ({prob*100:.2f}%)")
            st.progress(prob)
    else:
        st.write("No predictions.")

# Run the app
if __name__ == "__main__":
    app()

