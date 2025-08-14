from sklearn.model_selection import train_test_split
import numpy as np
from dataset import SkinLesionDataset

def split_dataset(image_paths, labels, test_size=0.15, val_size=0.1, random_seed=42):
    stratify_labels = labels
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=(test_size + val_size), stratify=stratify_labels, random_state=random_seed
    )
    val_split = val_size / (test_size + val_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=val_split, stratify=temp_labels, random_state=random_seed
    )
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

def split_dataset_df(val_test_paths, val_test_labels, test_size=0.4, random_seed=42):
    stratify_labels = val_test_labels
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_test_paths, val_test_labels, test_size=0.4, stratify=stratify_labels, random_state=random_seed
    )

    return val_paths, val_labels, test_paths, test_labels

def merge_skin_lesion_datasets(datasets):
    all_image_paths = []
    all_labels = []
    
    for dataset in datasets:
        all_image_paths.extend(dataset.image_paths)  # Combine lists of image paths
        all_labels.append(dataset.labels)           # Append arrays of labels
    
    # Concatenate all label arrays into a single array
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Return a new SkinLesionDataset with combined data
    return SkinLesionDataset(image_paths=all_image_paths, labels=all_labels, transform=datasets[0].transform)