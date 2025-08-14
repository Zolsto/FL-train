import csv
from sklearn import preprocessing
import pandas as pd
import numpy as np
from collections import Counter

class MyDataLoader():
    def __init__(self, image_contactPolarized_path: str, image_contactNonPolarized_path: str, image_nonContactPolarized_path: str):
        super().__init__()
        self.image_contactPolarized_path = image_contactPolarized_path
        self.image_contactNonPolarized_path = image_contactNonPolarized_path
        self.image_nonContactPolarized_path = image_nonContactPolarized_path

    def extract_image_paths_and_labels(self, contactPolarized: bool, contactNonPolarized: bool, nonContactPolarized: bool):
        
        jpg_files_contactPolarized, jpg_files_contactNonPolarized, jpg_files_nonContactPolarized = [], [], []
        
        labels_contactPolarized, labels_contactNonPolarized, labels_nonContactPolarized = [], [], []

        lesion_ids_contactPolarized, lesion_ids_contactNonPolarized, lesion_ids_nonContactPolarized = [], [], []
        
        for i in range(6):
            with open(self.image_contactPolarized_path+f"/{i}/metadata.csv", newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    jpg_files_contactPolarized.append(self.image_contactPolarized_path + f"/{i}/{row['isic_id']}.jpg")
                    labels_contactPolarized.append(row["diagnosis"])
                    lesion_ids_contactPolarized.append(row["lesion_id"])

            with open(self.image_contactNonPolarized_path+f"/{i}/metadata.csv", newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    jpg_files_contactNonPolarized.append(self.image_contactNonPolarized_path + f"/{i}/{row['isic_id']}.jpg")
                    labels_contactNonPolarized.append(row["diagnosis"])
                    lesion_ids_contactNonPolarized.append(row["lesion_id"])
        
            with open(self.image_nonContactPolarized_path+f"/{i}/metadata.csv", newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    jpg_files_nonContactPolarized.append(self.image_nonContactPolarized_path + f"/{i}/{row['isic_id']}.jpg")
                    labels_nonContactPolarized.append(row["diagnosis"])
                    lesion_ids_nonContactPolarized.append(row["lesion_id"])

        count_contactPolarized = Counter(labels_contactPolarized)
        count_contactNonPolarized = Counter(labels_contactNonPolarized)
        count_nonContactPolarized = Counter(labels_nonContactPolarized)

        if contactPolarized:
            for value,count in count_contactPolarized.items():
                print(f"Contact Polarized: {value} - {count}")

        if contactNonPolarized:
            for value,count in count_contactNonPolarized.items():
                print(f"Contact Non Polarized: {value} - {count}")

        if nonContactPolarized:
            for value,count in count_nonContactPolarized.items():
                print(f"Non Contact Polarized: {value} - {count}")

        lesion_ids = lesion_ids_contactPolarized + lesion_ids_contactNonPolarized + lesion_ids_nonContactPolarized
        duplicated_indices = self.find_duplicates(lesion_ids)

        image_paths = jpg_files_contactPolarized + jpg_files_contactNonPolarized + jpg_files_nonContactPolarized
        y = labels_contactPolarized + labels_contactNonPolarized + labels_nonContactPolarized
        unique_paths, unique_labels = self.remove_duplicates(image_paths, y, duplicated_indices, len(jpg_files_contactPolarized), len(jpg_files_contactNonPolarized))
    
        jpg_files_contactPolarized = unique_paths["Contact Polarized"]
        jpg_files_contactNonPolarized = unique_paths["Contact Non Polarized"]
        jpg_files_nonContactPolarized = unique_paths["Non Contact Polarized"]

        labels_contactPolarized = unique_labels["Contact Polarized"]
        labels_contactNonPolarized = unique_labels["Contact Non Polarized"]
        labels_nonContactPolarized = unique_labels["Non Contact Polarized"]

        if contactPolarized:
            image_paths = jpg_files_contactPolarized
            y = labels_contactPolarized
            output_file = "output/Dataset/label_mapping_" + "contactPolarized" + ".txt"
        if contactNonPolarized:
            image_paths = jpg_files_contactNonPolarized
            y = labels_contactNonPolarized
            output_file = "output/Dataset/label_mapping_" + "contactNonPolarized" + ".txt"
        if nonContactPolarized:
            image_paths = jpg_files_nonContactPolarized
            y = labels_nonContactPolarized
            output_file = "output/Dataset/label_mapping_" + "nonContactPolarized" + ".txt"

        le = preprocessing.LabelEncoder()
        le.fit(y)
        labels = le.transform(y)

        # Create a mapping between strings and their corresponding integers
        label_mapping = {label: idx for idx, label in enumerate(le.classes_)}

        # Save the mapping to a text file
        with open(output_file, "w") as file:
            file.write("String-to-Integer Label Mapping:\n")
            for string_label, int_label in label_mapping.items():
                file.write(f"{string_label}: {int_label}\n")

        return image_paths, labels
    
    def extract_device_labels(self, visualization):

        # Load the CSV file
        contact_part = []
        contactNon_part = []
        nonContact_part = []
        for i in range(6):
            df_temp = pd.read_csv(self.image_contactPolarized_path+f"/{i}/+metadata.csv")
            contact_part.append(df_temp)
            df_temp = pd.read_csv(self.image_contactNonPolarized_path+f"/{i}/+metadata.csv")
            contactNon_part.append(df_temp)
            df_temp = pd.read_csv(self.image_nonContactPolarized_path+f"/{i}/+metadata.csv")
            nonContact_part.append(df_temp)
            
        df_contactPolarized = pd.concat(contact_part, ignore_index=True)
        df_contactNonPolarized = pd.concat(contactNon_part, ignore_index=True)
        df_nonContactPolarized = pd.concat(nonContact_part, ignore_index=True)

        len_contactPolarized = len(df_contactPolarized)
        len_contactNonPolarized = len(df_contactNonPolarized)
        len_nonContactPolarized = len(df_nonContactPolarized)

        if visualization == True:
            device_labels = ([0] * 500) + ([1] * 500) + ([2] * 500)
        else:
            device_labels = ([0] * len_contactPolarized) + ([1] * len_contactNonPolarized) + ([2] * len_nonContactPolarized)

        return device_labels
    
    def find_duplicates(self, lesion_ids):
        # Get unique elements and their counts
        unique_elements, counts = np.unique(lesion_ids, return_counts=True)
        
        # Find duplicated elements (those with count > 1)
        duplicated_elements = unique_elements[counts > 1]
        
        # Find the indices of the duplicated elements
        np_lesion_ids = np.array(lesion_ids)
        duplicated_indices = {elem: np.where(np_lesion_ids == elem)[0].tolist() for elem in duplicated_elements}
        
        return duplicated_indices
    
    def remove_duplicates(self, input_list_paths, input_list_labels, duplicated_positions, n_contactPolarized,
                          n_contactNonPolarized):
        
        unique_lesion_paths = input_list_paths.copy()  
        unique_lesion_labels = input_list_labels.copy()

        # Iterate through duplicated positions
        for item, indices in duplicated_positions.items():
            for index in sorted(indices, reverse=True):  
                if (item != '') and (1 != index):
                    unique_lesion_paths[index] = None
                    unique_lesion_labels[index] = None

        unique_paths_contactPolarized = unique_lesion_paths[:n_contactPolarized]
        unique_paths_contactNonPolarized = unique_lesion_paths[n_contactPolarized:n_contactPolarized + n_contactNonPolarized]
        unique_paths_nonContactPolarized = unique_lesion_paths[n_contactPolarized + n_contactNonPolarized:]

        unique_labels_contactPolarized = unique_lesion_labels[:n_contactPolarized]
        unique_labels_contactNonPolarized = unique_lesion_labels[n_contactPolarized:n_contactPolarized + n_contactNonPolarized]
        unique_labels_nonContactPolarized = unique_lesion_labels[n_contactPolarized + n_contactNonPolarized:]

        unique_paths_contactPolarized = [elem for elem in unique_paths_contactPolarized if elem is not None]
        unique_paths_contactNonPolarized = [elem for elem in unique_paths_contactNonPolarized if elem is not None]
        unique_paths_nonContactPolarized = [elem for elem in unique_paths_nonContactPolarized if elem is not None]

        unique_labels_contactPolarized = [elem for elem in unique_labels_contactPolarized if elem is not None]
        unique_labels_contactNonPolarized = [elem for elem in unique_labels_contactNonPolarized if elem is not None]
        unique_labels_nonContactPolarized = [elem for elem in unique_labels_nonContactPolarized if elem is not None]

        total_unique = unique_labels_contactPolarized + unique_labels_contactNonPolarized + unique_labels_nonContactPolarized

        unique_paths = {"Contact Polarized": unique_paths_contactPolarized, "Contact Non Polarized": unique_paths_contactNonPolarized, "Non Contact Polarized": unique_paths_nonContactPolarized}
        unique_labels = {"Contact Polarized": unique_labels_contactPolarized, "Contact Non Polarized": unique_labels_contactNonPolarized, "Non Contact Polarized": unique_labels_nonContactPolarized}

        count_ak = np.sum(np.array(total_unique) == "actinic keratosis")
        count_bcc = np.sum(np.array(total_unique) == "basal cell carcinoma")
        count_mel = np.sum(np.array(total_unique) == "melanoma")
        count_nev = np.sum(np.array(total_unique) == "nevus")
        count_sk = np.sum(np.array(total_unique) == "seborrheic keratosis")
        count_scc = np.sum(np.array(total_unique) == "squamous cell carcinoma")

        return unique_paths, unique_labels
