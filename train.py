# import hydra
import torch
import torch.nn.functional as F
import numpy as np
import warnings
# from omegaconf import DictConfig
from torch.utils.data import DataLoader

from flwr.server.strategy import FedAvg

from dataModule import SkinLesionDataModule
from data_loader import MyDataLoader
from modules.federated.efficientnet import EfficientNetModel
from modules.federated.partitioners import CombinedDirichletPartitioner, DirichletPartitioner
from modules.federated.visualization import plot_label_distributions
import modules.federated.strategies as strategies
from modules.trainers.fed_trainer import FedTrainer
from utils import merge_skin_lesion_datasets

from torch.utils.tensorboard import SummaryWriter
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import os


def main():

    print("GUIDE TO DEBUG")
    print("\n- To print shape of model's output and labels: federated/task.py line 232")

    print()

    folder_path_contactPolarized = "../../Sensor based dataset/Filtered-by-pathology-unique/Contact Polarized"
    folder_path_contactNonPolarized = "../../Sensor based dataset/Filtered-by-pathology-unique/Contact Non Polarized"
    folder_path_nonContactPolarized = "../../Sensor based dataset/Filtered-by-pathology-unique/Non Contact Polarized"

    # CSV_contactPolarized = "../../Sensor based dataset/Filtered-by-pathology-unique/Contact Polarized"
    # CSV_contactNonPolarized = "../../Sensor based dataset/Filtered-by-pathology-unique/Contact Non Polarized"
    # CSV_nonContactPolarized = "../../Sensor based dataset/Filtered-by-pathology-unique/Non Contact Polarized"

    loader = MyDataLoader(folder_path_contactPolarized, folder_path_contactNonPolarized, folder_path_nonContactPolarized)

    # Initialize model and data
    unique_paths_contactPolarized, unique_labels_contactPolarized = loader.extract_image_paths_and_labels(contactPolarized=True, contactNonPolarized=False, nonContactPolarized=False)
    unique_paths_contactNonPolarized, unique_labels_contactNonPolarized = loader.extract_image_paths_and_labels(contactPolarized=False, contactNonPolarized=True, nonContactPolarized=False)
    unique_paths_nonContactPolarized, unique_labels_nonContactPolarized = loader.extract_image_paths_and_labels(contactPolarized=False, contactNonPolarized=False, nonContactPolarized=True)

    data_module_contactPolarized = SkinLesionDataModule(unique_paths_contactPolarized, unique_labels_contactPolarized, batch_size=32, augm=True, selec_augm=False)
    data_module_contactNonPolarized = SkinLesionDataModule(unique_paths_contactNonPolarized, unique_labels_contactNonPolarized, batch_size=32, augm=True, selec_augm=False)
    data_module_nonContactPolarized = SkinLesionDataModule(unique_paths_nonContactPolarized, unique_labels_nonContactPolarized, batch_size=32, augm=True, selec_augm=False)

    data_module_contactPolarized.setup()
    data_module_contactNonPolarized.setup()
    data_module_nonContactPolarized.setup()

    train_dataset_contactPolarized = data_module_contactPolarized.train_dataloader().dataset
    train_dataset_contactNonPolarized = data_module_contactNonPolarized.train_dataloader().dataset
    train_dataset_nonContactPolarized = data_module_nonContactPolarized.train_dataloader().dataset

    print(len(train_dataset_contactPolarized.image_paths))
    print(len(train_dataset_contactNonPolarized.image_paths))
    print(len(train_dataset_nonContactPolarized.image_paths))

    val_dataset_contactPolarized = data_module_contactPolarized.val_dataloader().dataset
    val_dataset_contactNonPolarized = data_module_contactNonPolarized.val_dataloader().dataset
    val_dataset_nonContactPolarized = data_module_nonContactPolarized.val_dataloader().dataset

    test_dataset_contactPolarized = data_module_contactPolarized.test_dataloader().dataset
    test_dataset_contactNonPolarized = data_module_contactNonPolarized.test_dataloader().dataset
    test_dataset_nonContactPolarized = data_module_nonContactPolarized.test_dataloader().dataset

    shift_train_contactNonPolarized = len(train_dataset_contactPolarized.image_paths)
    shift_train_nonContactPolarized = len(train_dataset_contactPolarized.image_paths) + len(train_dataset_contactNonPolarized.image_paths)

    shift_val_contactNonPolarized = len(val_dataset_contactPolarized.image_paths)
    shift_val_nonContactPolarized = len(val_dataset_contactPolarized.image_paths) + len(val_dataset_contactNonPolarized.image_paths)

    group_split = [57, 75, 80]

    partitioner_train_contactPolarized = DirichletPartitioner(num_partitions=57, alpha=30, seed=42, partition_by=1, min_partition_size=80, self_balancing=False, shuffle=True, shift_indices=0, shift_partition_id=0)
    partitioner_train_contactNonPolarized = DirichletPartitioner(num_partitions=18, alpha=20, seed=42, partition_by=1, min_partition_size=80, self_balancing=False, shuffle=True, shift_indices=shift_train_contactNonPolarized, shift_partition_id=57)
    partitioner_train_nonContactPolarized = DirichletPartitioner(num_partitions=5, alpha=20, seed=42, partition_by=1, min_partition_size=80, self_balancing=False, shuffle=True, shift_indices=shift_train_nonContactPolarized, shift_partition_id=75)

    partitioner_val_contactPolarized = DirichletPartitioner(num_partitions=57, alpha=30, seed=42, partition_by=1, min_partition_size=15, self_balancing=False, shuffle=True, shift_indices=0, shift_partition_id=0)
    partitioner_val_contactNonPolarized = DirichletPartitioner(num_partitions=18, alpha=30, seed=42, partition_by=1, min_partition_size=15, self_balancing=False, shuffle=True, shift_indices=shift_val_contactNonPolarized, shift_partition_id=57)
    partitioner_val_nonContactPolarized = DirichletPartitioner(num_partitions=5, alpha=30, seed=42, partition_by=1, min_partition_size=15, self_balancing=False, shuffle=True, shift_indices=shift_val_nonContactPolarized, shift_partition_id=75)

    partitioner_train_contactPolarized.dataset = train_dataset_contactPolarized
    partitioner_train_contactNonPolarized.dataset = train_dataset_contactNonPolarized
    partitioner_train_nonContactPolarized.dataset = train_dataset_nonContactPolarized

    partitioner_val_contactPolarized.dataset = val_dataset_contactPolarized
    partitioner_val_contactNonPolarized.dataset = val_dataset_contactNonPolarized
    partitioner_val_nonContactPolarized.dataset = val_dataset_nonContactPolarized

    partitioner_train_contactPolarized.create_partitions()
    partitioner_train_contactNonPolarized.create_partitions()
    partitioner_train_nonContactPolarized.create_partitions()

    partitioner_val_contactPolarized.create_partitions()
    partitioner_val_contactNonPolarized.create_partitions()
    partitioner_val_nonContactPolarized.create_partitions()

    list_train_datasets = [train_dataset_contactPolarized, train_dataset_contactNonPolarized, train_dataset_nonContactPolarized]
    train_dataset = merge_skin_lesion_datasets(list_train_datasets)

    list_val_datasets = [val_dataset_contactPolarized, val_dataset_contactNonPolarized, val_dataset_nonContactPolarized]
    val_dataset = merge_skin_lesion_datasets(list_val_datasets)

    # Create 2 versions of the test dataset
    list_test_datasets = [test_dataset_contactPolarized, test_dataset_contactNonPolarized, test_dataset_nonContactPolarized]
    test_dataset = merge_skin_lesion_datasets(list_test_datasets)
    test_dataloader = DataLoader(test_dataset)
    test_dict = {
        "group0" : DataLoader(test_dataset_contactPolarized),
        "group1" : DataLoader(test_dataset_contactNonPolarized),
        "group2" : DataLoader(test_dataset_nonContactPolarized)
    }


    train_combined_partitioner = CombinedDirichletPartitioner([partitioner_train_contactPolarized, partitioner_train_contactNonPolarized, partitioner_train_nonContactPolarized], train_dataset)
    val_combined_partitioner = CombinedDirichletPartitioner([partitioner_val_contactPolarized, partitioner_val_contactNonPolarized, partitioner_val_nonContactPolarized], val_dataset)

    partitioners_dict = {"train": train_combined_partitioner, "val": val_combined_partitioner}

    #fig_contactPolarized_train, _, _ = plot_label_distributions(train_combined_partitioner, label_name=1, max_num_partitions=80, plot_type="bar", legend=True, figsize=(20, 14))
    #fig_contactPolarized_train.savefig("Data Visualization/all_partitions_train.png")

    #fig_contactPolarized_val, _val, _val = plot_label_distributions(val_combined_partitioner, label_name=1, max_num_partitions=14, plot_type="bar", legend=True, figsize=(18, 12))
    #fig_contactPolarized_val.savefig("Data Visualization/all_partitions_val.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device selected: ", device)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA current device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    model = EfficientNetModel(num_classes=6, fine_tune_layers=True, premodel="presint")
    # Load pre-trained weights
    pre_trained_weights = [layer.cpu().numpy() for layer in model.state_dict().values()]
    #print(list(model.state_dict().keys()))
    start_parameters = ndarrays_to_parameters(pre_trained_weights)
    split_index = -1
    for i, name in enumerate(model.state_dict().keys()):
        if split_index > 0:
            break

        if "classifier" in name:
            split_index = i
    
    print(split_index)
    model.to(device)
    #print(model.named_parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    #raise Exception("Stopped")

    logdir = "output/FedAvg/my-nopre"
    os.makedirs(logdir, exist_ok=True)
    server_writer = SummaryWriter(log_dir=logdir)

    fed_trainer = FedTrainer(
        c_model=model,
        n_classes=6,
        criterion=loss_fn,
        optimizer=torch.optim.AdamW,
        device=device,
        save_path=logdir,
    )

    fed_trainer.set_data(
        dataloader_test=test_dataloader,
        batch_size=32,
        partitioner=partitioners_dict,
        seed=42,
    )

    evaluate_f = strategies.utils.get_evaluate_fn(
        model=model,
        device=device,
        criterion=loss_fn,
        dataloader=test_dataloader,
        writer=server_writer,
        loader_dict=test_dict
    )

    # FedAvg strategy
    fed_trainer.set_client_and_server(
        strategy=strategies.MyFedAvg(
            save_path=logdir,
            fraction_fit=0.00001,
            fraction_evaluate=1, 
            min_fit_clients=5, 
            #min_evaluate_clients=10,
            min_available_clients=10, 
            on_fit_config_fn=strategies.get_on_fit_config({"local_epochs": 5}),
            #on_evaluate_config_fn=strategies.get_on_evaluate_config({"batch_size": 32}),
            fit_metrics_aggregation_fn=strategies.get_fit_metrics_aggregation_fn(),
            #evaluate_metrics_aggregation_fn=strategies.get_fit_metrics_aggregation_fn()
            evaluate_fn=evaluate_f,
            initial_parameters=None
        ),
        num_rounds=50,
        log_every=1,
    )

    # Personalized strategy
#    fed_trainer.set_client_and_server(
#        strategy=strategies.AllClusterAvg(
#            # Parameters of ClusterAvg
#            group_split=group_split,
#            writer=server_writer,
#            param_split=split_index,
#            weighted_loss=True,
#            group_at_end=True,
#            save_path=logdir,
#            separate_eval=False,
#            # Parameters like FedAvg
#            fraction_fit=0.00001,   #0.00001
#            fraction_evaluate=1,
#            min_fit_clients=5,
#            # min_evaluate_clients=10,
#            min_available_clients=10,
#            on_fit_config_fn=strategies.get_on_fit_config({"local_epochs": 5}),
#            # on_evaluate_config_fn=strategies.get_on_evaluate_config({"batch_size": 32}),
#            fit_metrics_aggregation_fn=strategies.get_fit_metrics_aggregation_fn(),
#            # evaluate_metrics_aggregation_fn=strategies.get_fit_metrics_aggregation_fn()
#            evaluate_fn=evaluate_f,
#            initial_parameters=None,
#        ),
#        num_rounds=50,
#        log_every=1,
#    )

    fed_trainer(num_clients=80)
    server_writer.close()
    # Evaluate the model on test set
    #test_metrics = fed_trainer.evaluate_on_test_set()
    #print(f"Test Loss: {test_metrics['test_loss']:.4f}")
    #print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
