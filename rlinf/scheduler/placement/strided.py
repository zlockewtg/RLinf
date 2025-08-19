# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

from .placement import Placement, PlacementStrategy


class StridedPlacementStrategy(PlacementStrategy):
    """Placement strategy that allows one process to have multiple GPUs in a strided manner, e.g., GPU 0, 2, 4 (stride 2) or GPU 0, 3, 6 (stride 3)."""

    def __init__(
        self,
        master_node: int = 0,
        num_nodes: int = 0,
        stride: int = 1,
        num_gpus_per_process: int = 1,
    ):
        """Initialize the StridedPlacementStrategy.

        Args:
            master_node (int): The ID of the master node where the first process will be placed
            num_nodes (int): Total number of nodes in the cluster.
            stride (int): The stride value for the placement. Defaults to 1.
            num_gpus_per_process (int): Number of GPUs per process. Defaults to 1.

        """
        super().__init__(master_node, num_nodes)
        self._placement_strategy = "STRIDED"
        self._num_gpus_per_process = num_gpus_per_process
        self._stride = stride

    def get_placement(
        self,
        num_nodes_in_cluster: int,
        num_gpus_per_node: int,
        isolate_gpu: bool = True,
    ) -> List[Placement]:
        """Generate a list of placements based on the strided strategy.

        Args:
            num_nodes_in_cluster (int): Total number of nodes in the cluster.
            num_gpus_per_node (int): Number of GPUs per node.
            isolate_gpu (bool): Whether to isolate the GPUs for each process by setting `CUDA_VISIBLE_DEVICES`. Defaults to True.

        Returns:
            List[Placement]: A list of Placement objects representing the placements of processes on GPUs.

        """
        placements = []
        self._logger.info(
            f"Using strided placement with {self._num_gpus_per_process} GPUs per process, starting from node {self._master_node}."
        )
        assert isolate_gpu, (
            "Strided placement strategy requires isolate_gpu to be True."
        )
        assert self._num_nodes + self._master_node <= num_nodes_in_cluster, (
            f"Master node {self._master_node} and number of nodes {self._num_nodes} exceed the number of nodes in the cluster {num_nodes_in_cluster}."
        )

        # Essentially, this placement strategy first splits GPUs into several groups, each of which contains `stride` number of processes and each process has `num_gpus_per_process` GPUs.
        # Then, it fills all GPUs with these groups in a close-packed manner.
        # For example, if we have 8 GPUs per node, 1 node, stride 2 and 2 GPUs per process, the placement will be:
        # Group 0: Process 0 (GPUs 0, 2), Process 1 (GPUs 1, 3)
        # Group 1: Process 2 (GPUs 4, 6), Process 3 (GPUs 5, 7)
        num_gpus_per_group = self._stride * self._num_gpus_per_process
        num_groups = (self._num_nodes * num_gpus_per_node) // num_gpus_per_group

        assert self._stride <= num_gpus_per_node, (
            f"Stride {self._stride} must be less than or equal to number of GPUs per node {num_gpus_per_node}."
        )
        assert (self._stride - 1) + (
            self._num_gpus_per_process - 1
        ) * self._stride < num_gpus_per_node, (
            f"Stride {self._stride} and number of GPUs per process {self._num_gpus_per_process} cannot fit one process in a node with {num_gpus_per_node} GPUs per node."
        )
        assert num_groups * num_gpus_per_group == self._num_nodes * num_gpus_per_node, (
            f"The stride {self._stride} and num_gpus_per_process {self._num_gpus_per_process} cannot fit in the number of nodes {self._num_nodes} * number of GPUs per node {num_gpus_per_node}."
        )

        for group_rank in range(num_groups):
            for process_rank_in_group in range(self._stride):
                rank = group_rank * self._stride + process_rank_in_group
                global_start_gpu_id = (
                    group_rank * num_gpus_per_group + process_rank_in_group
                )
                node_rank = global_start_gpu_id // num_gpus_per_node
                node_id = self._master_node + node_rank
                local_gpu_id = global_start_gpu_id % num_gpus_per_node
                local_rank = local_gpu_id

                num_groups_per_node = num_gpus_per_node // num_gpus_per_group
                local_world_size = num_groups_per_node * self._stride

                cuda_devices = [
                    local_gpu_id + i * self._stride
                    for i in range(self._num_gpus_per_process)
                ]

                assert all(gpu_id < num_gpus_per_node for gpu_id in cuda_devices), (
                    f"CUDA devices {cuda_devices} exceed the number of GPUs per node {num_gpus_per_node}."
                )

                placements.append(
                    Placement(
                        rank=rank,
                        node_id=node_id,
                        node_rank=node_rank,
                        local_gpu_id=local_gpu_id,
                        local_rank=local_rank,
                        local_world_size=local_world_size,
                        cuda_visible_devices=[str(gpu_id) for gpu_id in cuda_devices],
                        isolate_gpu=True,
                    )
                )

        self._logger.info(f"Generated {len(placements)} placements: {placements}.")

        return placements
