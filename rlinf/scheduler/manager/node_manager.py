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

from ..cluster import NodeInfo
from .manager import Manager


class NodeManager(Manager):
    """Global manager of node metadata information."""

    MANAGER_NAME = "NodeManager"

    def __init__(
        self,
        nodes: list[NodeInfo],
        num_gpus_per_node: int,
        master_ip: str,
        master_port: int,
    ):
        """Initialize the NodeManager.

        Args:
            nodes (list[NodeInfo]): List of NodeInfo objects representing the nodes in the cluster
            num_gpus_per_node (int): Number of GPUs available per node.
            master_ip (str): IP address of the master node.
            master_port (int): Port number of the master node.

        """
        self._nodes = nodes
        self._master_ip = master_ip
        self._master_port = master_port
        self._num_gpus_per_node = num_gpus_per_node

    def get_nodes(self):
        """Get the list of nodes in the cluster."""
        return self._nodes

    def get_master_ip(self):
        """Get the IP address of the master node."""
        return self._master_ip

    def get_master_port(self):
        """Get the port number of the master node."""
        return self._master_port

    def get_num_gpus_per_node(self):
        """Get the number of GPUs available per node."""
        return self._num_gpus_per_node
