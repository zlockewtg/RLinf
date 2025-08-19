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

import logging
from dataclasses import dataclass
from typing import List, overload

from ..cluster import Cluster


@dataclass
class Placement:
    """Class representing the placement of a worker on a specific GPU."""

    rank: int
    """Global rank of the worker in the cluster."""

    node_id: int
    """Node ID where the worker is placed."""

    node_rank: int
    """Rank of the node in the cluster."""

    local_gpu_id: int
    """Local GPU ID on the node."""

    local_rank: int
    """Local rank of the worker on the node."""

    local_world_size: int
    """Local world size (number of workers) on the node."""

    cuda_visible_devices: List[str]
    """List of CUDA visible devices for the worker."""

    isolate_gpu: bool
    """Flag to indicate if the local rank should be set to zero. This is useful for workers that require multiple GPUs."""


class PlacementStrategy:
    """Base class for placement strategies.

    To setup a placement strategy, you can either use the `PlacementStrategy.get_placement_strategy` method to parse config from the OmegaConf yaml, or directly instantiate a subclass of `PlacementStrategy`.

    A config yaml looks something like this:

    .. code-block:: yaml

        placement:
            strategy: packed  # or strided
            master_node: 0  # The node where the first worker will be placed.
            num_nodes: 1  # The total number of nodes in the cluster.
            master_gpu: 0  # The GPU where the first worker will be placed.

    You can then pass the placement config to the `PlacementStrategy.get_placement_strategy` method to get the appropriate placement strategy instance.
    The config attributes can be found in the documentations of the `__init__` parameters of the corresponding placement strategy class, e.g., `PackedPlacementStrategy` or `StridedPlacementStrategy`.

    Currently, we support two placement strategy subclasses, representing two strategies:

    1. PackedPlacementStrategy: Places workers in a packed manner, filling up each node with workers before moving to the next node.
    2. StridedPlacementStrategy: Places workers in a strided manner, allowing for strided placement across GPUs.

    The following example shows how to place the worker on specified nodes and GPUs with the placement strategy.

    Example::

        >>> from rlinf.scheduler import (
        ...     Cluster,
        ...     Worker,
        ...     PackedPlacementStrategy,
        ...     StridedPlacementStrategy,
        ... )
        >>>
        >>> class MyWorker(Worker):
        ...     def __init__(self, msg: str = "Hello, World!"):
        ...         super().__init__()
        ...         self._msg = msg
        ...
        ...     def hello(self):
        ...         return self._rank
        ...
        ...     def available_gpus(self):
        ...         import torch
        ...         available_gpus = torch.cuda.device_count()
        ...         gpu_ids = [
        ...             torch.cuda.get_device_properties(i) for i in range(available_gpus)
        ...         ]
        ...         return available_gpus
        >>>
        >>> cluster = Cluster(num_nodes=1, num_gpus_per_node=8)
        >>>
        >>> # Launch 8 processes
        >>> my_worker_group = MyWorker.create_group(msg="Hello").launch(cluster=cluster)
        >>> # This will execute the hello method on all the 8 processes in the group.
        >>> futures = my_worker_group.hello()
        >>> # Wait for all workers to complete and get the results.
        >>> futures.wait()
        [0, 1, 2, 3, 4, 5, 6, 7]
        >>>
        >>>
        >>> # This will execute the hello method only on ranks 0 and 1.
        >>> my_worker_group.execute_on([0, 1]).hello().wait()
        [0, 1]
        >>>
        >>>
        >>> # Create a placement strategy. This controls how workers are placed on the cluster.
        >>> # `PackedPlacementStrategy` will fill up nodes with workers before moving to the next node.
        >>> placement = PackedPlacementStrategy(master_node=0, num_nodes=1)
        >>> my_worker = MyWorker.create_group().launch(
        ...     cluster=cluster, name="packed_group", placement_strategy=placement
        ... )
        >>> my_worker.available_gpus().wait() # This will run 8 processes on the first node, each using 1 GPU.
        [1, 1, 1, 1, 1, 1, 1, 1]
        >>>
        >>>
        >>> # `master_gpu` allows for control over worker placement at the GPU granularity.
        >>> # `num_processes` allows for control over how many processes to run.
        >>> placement_fine_grained = PackedPlacementStrategy(
        ...     master_node=0, master_gpu=2, num_processes=2
        ... )
        >>> my_worker_fine = MyWorker.create_group().launch(
        ...     cluster=cluster,
        ...     name="packed_group_fine",
        ...     placement_strategy=placement_fine_grained,
        ... )
        >>> my_worker_fine.available_gpus().wait()  # This will only run on two GPUs (2 and 3) of the first node.
        [1, 1]
        >>>
        >>>
        >>> # `num_gpus_per_process` allows for one process to hold multiple GPUs.
        >>> # For example, if you want a process to hold 4 GPUs, you can set the `num_gpus_per_process` to 4.
        >>> placement_chunked = PackedPlacementStrategy(
        ...     master_node=0, num_nodes=1, num_gpus_per_process=4
        ... )
        >>> my_worker_chunked = MyWorker.create_group().launch(
        ...     cluster=cluster,
        ...     name="my_worker_chunked",
        ...     placement_strategy=placement_chunked,
        ... )
        >>> my_worker_chunked.available_gpus().wait()  # This will run 2 processes, each using 4 GPUs (0-3 and 4-7) of the first node.
        [4, 4]
        >>>
        >>>
        >>> # `StridedPlacementStrategy` allows for strided placement of workers across GPUs.
        >>> # For example, if you want to place workers on every second GPU, you can set the stride to 2.
        >>> placement_strided = StridedPlacementStrategy(
        ...     master_node=0, num_nodes=1, stride=2, num_gpus_per_process=2
        ... )
        >>> my_worker_strided = MyWorker.create_group().launch(
        ...     cluster=cluster,
        ...     name="my_worker_strided",
        ...     placement_strategy=placement_strided,
        ... )
        >>> my_worker_strided.available_gpus().wait()  # This will run 4 processes, each using 2 GPUs (0,2 1,3 4,6 5,7) of the first node.
        [2, 2, 2, 2]

    """

    def __init__(self, master_node: int, num_nodes: int):
        """Initialize the PlacementStrategy.

        Args:
            master_node (int): The ID of the master node.
            num_nodes (int): The total number of nodes in the cluster.

        """
        self._placement_strategy = None
        self._master_node = master_node
        self._num_nodes = num_nodes
        self._logger = logging.getLogger(name=self.__class__.__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(levelname)s %(asctime)s %(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    @overload
    def get_placement(
        self, num_gpus_per_node: int, isolate_gpu: bool = True
    ) -> List[Placement]:
        return None

    @classmethod
    def get_placement_strategy(
        cls, placement_cfg, cluster: Cluster
    ) -> "PlacementStrategy":
        """Get the placement strategy based on the configuration.

        Args:
            placement_cfg: Configuration object containing placement strategy details.
            cluster (Cluster): The Cluster object containing information about the cluster.

        Returns:
            PlacementStrategy: An instance of the appropriate placement strategy class.

        """
        strategy_name = placement_cfg.get("strategy", "default").lower()
        master_node = placement_cfg.get("master_node", 0)
        num_nodes = cluster.num_nodes
        from .packed import PackedPlacementStrategy
        from .strided import StridedPlacementStrategy

        if strategy_name == "packed":
            num_gpus_per_process = placement_cfg.get("num_gpus_per_process", 1)
            master_gpu = placement_cfg.get("master_gpu", 0)
            num_processes = placement_cfg.get("num_processes", 0)
            if num_processes != 0:
                num_nodes = 0  # If num_processes is set, num_nodes should not be used.
            return PackedPlacementStrategy(
                master_node=master_node,
                num_nodes=num_nodes,
                master_gpu=master_gpu,
                num_processes=num_processes,
                num_gpus_per_process=num_gpus_per_process,
            )
        elif strategy_name == "strided":
            stride = placement_cfg.get("stride", 1)
            num_gpus_per_process = placement_cfg.get("num_gpus_per_process", 1)
            return StridedPlacementStrategy(
                master_node=master_node,
                num_nodes=num_nodes,
                stride=stride,
                num_gpus_per_process=num_gpus_per_process,
            )
        elif strategy_name == "default":
            return None  # The WorkerGroup will recognize None and use the default placement strategy which is packed on all GPUs.
        else:
            raise ValueError(f"Unknown placement strategy: {strategy_name}")
