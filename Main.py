from datetime import datetime
import albumentations as A
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from clearml import Task
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from simple_parsing import ArgumentParser
from torch.utils.data import Subset

from config import config
from config.args import Args
from config.config import logger
from config.config import BASE_DIR, PROJECT_DIR, CONFIG_DIR, DATA_DIR, LOGS_DIR
from new_dataloader import CustomDataset  # Import your updated dataloader
from model import Classifier


def printing_paths():
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"PROJECT_DIR: {PROJECT_DIR}")
    print(f"CONFIG_DIR: {CONFIG_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"LOGS_DIR: {LOGS_DIR}")


# Preprocessing function
def get_transform(dataset):
    resize = A.Resize(224, 224)
    normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_tensor = ToTensorV2()

    if dataset == "train":
        return A.Compose([resize, normalize, to_tensor])
    elif dataset == "val":
        return A.Compose([resize, normalize, to_tensor])


# Dataset function
def prepare_dataset(bucket_name, train_prefix, val_prefix):
    try:
        trainset = CustomDataset(bucket_name=bucket_name, prefix=train_prefix, transforms=get_transform("train"))
        valset = CustomDataset(bucket_name=bucket_name, prefix=val_prefix, transforms=get_transform("val"))
        return trainset, valset
    except Exception as e:
        logger.error(f"Got an exception: {e}")


# Sanity check subset function
def create_subset(trainset, valset):
    train_subset = Subset(trainset, range(100))
    val_subset = Subset(valset, range(5))
    return train_subset, val_subset


# Dataloaders function
def create_dataloaders(args, train_subset, val_subset):
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_subset, batch_size=16, num_workers=4, shuffle=True
    )
    val_dataloader = DataLoader(
        val_subset, batch_size=16, num_workers=4, shuffle=False
    )
    return train_dataloader, val_dataloader


def set_device():
    if torch.cuda.is_available():
        device = "gpu"
    else:
        device = "cpu"
    return device


def main():
    printing_paths()
    device = set_device()
    logger.info(f"Currently using {device} device...")

    # Read args
    logger.info("Reading arguments...")
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="options")
    args_namespace = parser.parse_args()
    args = args_namespace.options

    # Prepare dataset using S3 bucket and prefixes
    logger.info("Preparing datasets...")
    bucket_name = config.S3_DATA_BUCKET
    train_prefix = config.S3_DATA_PREFIX + "train/"
    val_prefix = config.S3_DATA_PREFIX + "val/"
    trainset, valset = prepare_dataset(bucket_name=bucket_name, train_prefix=train_prefix, val_prefix=val_prefix)
    logger.info(
        f"""Total training images: {len(trainset)}
Total validation images: {len(valset)}"""
    )

    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(args, trainset, valset)

    # Initialize clearml task
    logger.info("Initializing clearml task...")
    task = Task.init(
        project_name="streamlit/image-classification",
        task_name=f"streamlit-image-classification-{datetime.now()}",
    )
    task.connect(args)

    # Save top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="loss/val_loss",
        mode="min",
        filename="streamlit-image-classification-{epoch:02d}-{val_loss:.2f}",
    )

    # Progress bar
    progress_bar = RichProgressBar()

    # Configure trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, progress_bar],
        default_root_dir=config.LOGS_DIR,
        accelerator=device,
        devices=1,
        log_every_n_steps=1,
    )

    # Define classifier
    classifier = Classifier()

    # Fit the model
    logger.info("Starting training...")
    trainer.fit(classifier, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()

