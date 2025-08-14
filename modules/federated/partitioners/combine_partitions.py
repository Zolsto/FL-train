from modules.federated.partitioners import DirichletPartitioner
from typing import List

class CombinedDirichletPartitioner(DirichletPartitioner):
    def __init__(self, partitioners: List[DirichletPartitioner], dataset):
        # Initialize with dummy parameters; they will not be used directly
        super().__init__(num_partitions=80, partition_by=1, alpha=1, min_partition_size=1, shift_indices=0, shift_partition_id=0, shuffle=False)
        
        self._partition_id_to_indices = {}
        self.dataset = dataset
        
        # Combine the datasets and partition mappings
        self._combine_partitioners(partitioners)

        self._partition_id_to_indices_determined = True
    
    def _combine_partitioners(self, partitioners: List[DirichletPartitioner]) -> None:
        """
        Combine the datasets and partitions from multiple DirichletPartitioner objects.
        """
        temp_partition_id_to_indices = {}
        for partitioner in partitioners:
            # Ensure all partitions have been created
            if not partitioner._partition_id_to_indices_determined:
                raise ValueError("Partitions must be created before combining.")
            temp_partition_id_to_indices.update(partitioner._partition_id_to_indices)
            
        self._partition_id_to_indices.update(temp_partition_id_to_indices)