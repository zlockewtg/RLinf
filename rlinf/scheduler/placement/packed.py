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


class PackedPlacementStrategy(PlacementStrategy):
    """Placement strategy that allows processes to be placed on GPUs in a close-packed manner. One process can have one or multiple GPUs."""

    def __init__(
        self,
        master_node: int = 0,
        num_nodes: int = 0,
        master_gpu: int = 0,
        num_processes: int = 0,
        num_gpus_per_process: int = 1,
    ):
        """Initialize the PackedPlacementStrategy.

        Args:
            master_node (int): The ID of the master node where the first process will be placed.
            num_nodes (int): Total number of nodes in the cluster.
            master_gpu (int): The GPU ID on the master node where the first process will be placed.
            num_processes (int): Total number of processes to place. Should not be specified if `num_nodes` is set, as it will be calculated based on `num_nodes` and `num_gpus_per_process`.
            num_gpus_per_process (int): Number of GPUs per process. Defaults to 1.
            isolate_gpu (bool): Whether to isolate the GPUs for each process by setting `CUDA_VISIBLE_DEVICES`. Defaults to True. When set to True, each process will only be able to see the GPUs they are assigned with. For example, if the `num_gpus_per_process` is 1, each process will only see 1 GPU, and `torch.cuda.current_device()` will return 0 on all processes. Some programs are incompatible with this behavior and must see all GPUs. So setting isolate_gpu=False will allow such programs to see all GPUs, and we can only make sure the process is scheduled on the right node instead of the right GPUs. Use at your own risk!

        """
        super().__init__(master_node, num_nodes)
        if num_processes != 0 and num_nodes != 0:
            raise ValueError(
                "No need to specify both num_processes and num_nodes. Because when num_nodes is set, num_processes will be calculated based on `num_nodes` and `num_gpus_per_process`. Setting both leads to ambiguity."
            )
        if num_processes == 0 and num_nodes == 0:
            raise ValueError(
                "At least one of num_processes or num_nodes must be specified."
            )
        self._placement_strategy = "PACKED"
        self._master_gpu = master_gpu
        self._num_processes = num_processes
        self._num_gpus_per_process = num_gpus_per_process

    @staticmethod
    def from_gpu_range(
        gpu_range: List[int], num_processes: int, num_gpus_per_node: int
    ):
        """Create a PackedPlacementStrategy from a GPU range.

        Args:
            gpu_range (List[int]): The range of GPU IDs to use for the placement.
            num_processes (int): The number of processes to place.
            num_gpus_per_node (int): The number of GPUs per node.

        Returns:
            PackedPlacementStrategy: The created PackedPlacementStrategy instance.
        """
        start_node = gpu_range[0] // num_gpus_per_node
        num_gpus = len(gpu_range)
        num_gpus_per_process = num_gpus // num_processes
        assert len(gpu_range) % num_processes == 0, (
            f"Number of GPUs {len(gpu_range)} is not divisible by number of processes {num_processes}"
        )
        return PackedPlacementStrategy(
            master_node=start_node,
            master_gpu=gpu_range[0],
            num_processes=num_processes,
            num_gpus_per_process=num_gpus_per_process,
        )

    def get_placement(
        self,
        num_nodes_in_cluster: int,
        num_gpus_per_node: int,
        isolate_gpu: bool = True,
    ) -> List[Placement]:
        """Generate a list of placements based on the packed strategy.

        Args:
            num_nodes_in_cluster (int): Total number of nodes in the cluster.
            num_gpus_per_node (int): Number of GPUs per node.
            isolate_gpu (bool): Whether to isolate the GPUs for each process by setting `CUDA_VISIBLE_DEVICES`. Defaults to True.

        Returns:
            List[Placement]: A list of Placement objects representing the placements of processes on GPUs.

        """
        rank = 0
        placements = []
        assert self._master_gpu < num_gpus_per_node, (
            f"Master GPU ID {self._master_gpu} must be within 0 to {num_gpus_per_node - 1}."
        )

        if self._num_nodes != 0 and self._num_processes == 0:
            self._num_processes = (
                self._num_nodes * num_gpus_per_node - self._master_gpu
            ) // self._num_gpus_per_process

        self._logger.info(
            f"Using packed placement with {self._num_gpus_per_process} GPUs per process, {self._num_processes} processes, starting from node {self._master_node}, GPU ID {self._master_gpu}."
        )

        global_gpu_id = self._master_gpu
        node_rank = 0
        node_id = self._master_node
        local_gpu_id = self._master_gpu
        local_rank = 0
        local_world_size = 1

        while True:
            assert local_gpu_id + self._num_gpus_per_process <= num_gpus_per_node, (
                f"Cannot find placement with {self._num_gpus_per_process} GPUs per process starting from GPU {local_gpu_id} on node {node_id} with {num_gpus_per_node} GPUs per node. Probably because `num_gpus_per_node` ({num_gpus_per_node}) cannot be divided by `num_gpus_per_process` ({self._num_gpus_per_process})."
            )

            if isolate_gpu:
                cuda_visible_devices = [
                    str(gpu_id)
                    for gpu_id in range(
                        local_gpu_id, local_gpu_id + self._num_gpus_per_process
                    )
                ]
            else:
                cuda_visible_devices = [
                    str(gpu_id) for gpu_id in range(num_gpus_per_node)
                ]

            placements.append(
                Placement(
                    rank=rank,
                    node_id=node_id,
                    node_rank=node_rank,
                    local_gpu_id=local_gpu_id,
                    local_rank=local_rank,
                    local_world_size=0,
                    cuda_visible_devices=cuda_visible_devices,
                    isolate_gpu=isolate_gpu,
                )
            )

            # The next placement
            rank += 1
            global_gpu_id += self._num_gpus_per_process
            if global_gpu_id % num_gpus_per_node == 0:  # Place to the next node
                node_rank += 1
                node_id += 1
                local_gpu_id = 0
                local_rank = 0
                next_node = True
            else:
                local_gpu_id = global_gpu_id % num_gpus_per_node
                local_rank += 1
                next_node = False

            if next_node or rank >= self._num_processes:
                # If we are at the end of a node, set local_world_size for all previous placements whose local_world_size == 0
                # Reversal traverse the placements to set local_world_size
                for i in range(len(placements) - 1, -1, -1):
                    if placements[i].local_world_size == 0:
                        placements[i].local_world_size = local_world_size
                    else:
                        break
                local_world_size = 1  # Reset for the next node
            else:
                local_world_size += 1

            if rank >= self._num_processes:
                break

            if self._num_nodes != 0:
                assert node_rank < self._num_nodes, (
                    f"Node rank {node_rank} exceeds the number of nodes {self._num_nodes}. Please check your configuration."
                )

            assert node_id < num_nodes_in_cluster, (
                f"Not enough ({num_nodes_in_cluster}) nodes in the cluster to generate the placement."
            )

        self._logger.info(f"Generated {len(placements)} placements: {placements}.")

        return placements
