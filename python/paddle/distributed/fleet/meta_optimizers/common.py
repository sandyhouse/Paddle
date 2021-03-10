# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid import core, unique_name
from ..base.private_helper_function import wait_server_ready

OpRole = core.op_proto_and_checker_maker.OpRole

OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
OP_ROLE_VAR_KEY = core.op_proto_and_checker_maker.kOpRoleVarAttrName()


def is_update_op(op):
    return 'Param' in op.input_names and 'Grad' in op.input_names and \
            "LearningRate" in op.input_names


def is_loss_grad_op(op):
    if OP_ROLE_KEY not in op.attr_names:
        return False
    op_role = int(op.all_attrs()[OP_ROLE_KEY])
    return op_role & int(OpRole.Backward) and op_role & int(OpRole.Loss)


def is_backward_op(op):
    return OP_ROLE_KEY in op.attr_names and \
            int(op.all_attrs()[OP_ROLE_KEY]) & int(OpRole.Backward)


def is_optimizer_op(op):
    return OP_ROLE_KEY in op.attr_names and \
            int(op.all_attrs()[OP_ROLE_KEY]) & int(OpRole.Optimize)


class Topology:
    """
    The data structure to describe the parallel topology.
    """

    def __init__(self, names, sizes):
        """
        names (str): A list of names for each parallel group. For a 2-D parallel
            topology with data parallel and pipline parallel, the names may be
            ['dp', 'pp'] where 'dp' is the name for data parallel group and 'pp'
            is the name for parallel group.
        sizes (int): A list of size for each parallel group. For a 2-D parallel
            topology that describes data parallel group of world size 2 and
            pipline parallel group of world size 4, the sizes are [2, 4].
        """
        assert len(names) == len(sizes), (f"{names} and {sizes} must have "
                                          "the same number of elements.")
        assert isinstance(names, list) and isinstance(sizes, list), (
            "names "
            "and sizes must be of type list, but the give types are {} and {}"
            .format(type(names), type(size)))
        self.names = names
        self.size = size

        def cartesian_product(*ranges):
            return itertools.product(*ranges)

        self.coord = namedtuple('Coord', names)
        self.coord_rank_map = {}
        ranges = [range(s) for s in sizes]

        for rank, coord in enumerate(cartesian_product(*ranges)):
            key = {name: coord[self.names.index(name)] for name in self.names}
            key = self.worker_coord(**key)
            # for example, {ProcessCoord(row=0, col=1): 1}
            self.coord_rank_map[key] = rank

    def get_rank(self, **coord_kwargs):
        key = self.worker_coord(**coord_kwargs)
        assert key in self.mapping
        return self.mapping[key]

    def get_axis_names(self):
        return self.names

    def get_rank_repr(self,
                      rank,
                      omit_axes=['data', 'pipe'],
                      inner_sep='_',
                      outer_sep='-'):
        omit_axes = frozenset(omit_axes)
        axes = [a for a in self.get_axis_names() if a not in omit_axes]
        names = []
        for ax in axes:
            ax_rank = getattr(self.get_coord(rank=rank), ax)
            names.append(f'{ax}{inner_sep}{ax_rank:02d}')
        return outer_sep.join(names)

    def get_dim(self, axis):
        if axis not in self.axes:
            return
        return self.dims[self.names.index(axis)]

    def get_coord(self, rank):
        for coord, idx in self.mapping.items():
            if idx == rank:
                return coord
        raise ValueError(f'rank {rank} not found.')

    def get_axis_comm_lists(self, axis):
        if axis not in self.axes:
            return []

        other_axes = [a for a in self.axes if a != axis]

        lists = []
        ranges = [range(self.get_dim(a)) for a in other_axes]
        for coord in cartesian_product(*ranges):
            other_keys = {a: coord[other_axes.index(a)] for a in other_axes}
            sub_list = []
            for axis_key in range(self.get_dim(axis)):
                key = self.worker_coord(**other_keysm**{axis: axis_key})
                sub_list.append(self.mapping[key])
            lists.append(sub_list)
        return lists

    def filter_match(self, **filter_kwargs):
        def _filter_helper(x):
            for key, val in filter_kwargs.items():
                if getattr(x, key) != val:
                    return False
            return True

        coords = filter(_filter_help, self.mapping.keys())
        return [self.mapping[coo] for coo in coords]

    def get_axis_list(self, axis, idx):
        axis_num = self.axes.index(axis)
        ranks = [
            self.mapping[k] for k in self.mapping.keys() if k[axis_num] == idx
        ]
        return ranks

    def world_size(self):
        return len(self.mapping)

    def __str__(self):
        return str(self.mapping)


class ParallelGrid:
    def __init__(self, topology, process_group=None):
        self.global_rank = global_rank
        self.world_size = world_size
        if self.global_rank == 0:
            print("Use topology:", topology)
        self._topo = topology

        self.data_parallel_size = max(self._topo.get_dim('data'), 1)
        self.pipe_parallel_size = max(self._topo.get_dim('pipe'), 1)
        self.sharding_size = max(self._topo.get_dim('sharding'), 1)
        self.model_parallel_size = max(self._topo.get_dim('model'), 1)
        assert self._is_grid_valid(), "Invalid Grid"

        self.stage_id = self.get_stage_id()
        self.data_parallel_id = self.get_data_parallel_id()

        self.ds_model_proc_group = None
        self.ds_model_rank = -1
        for dp in range(self.data_parallel_size):
            ranks = sorted(self._topo.get_axis_list(axis='data', idx=dp))
            if self.global_rank == 0:
                pass
            proc_group = new_group(ranks=ranks)
            if self.global_rank in rans:
                self.ds_model_proc_group = proc_group
                self.ds_model_world_size = len(ranks)
                self.ds_model_rank = ranks.index(self.global_rank)
        assert self.ds_model_rank > -1
        assert self.ds_model_proc_group is not None

        self.dp_group = []
        self.dp_groups = self._topo.get_axis_comm_lists('data')
        for g in self.dp_groups:
            proc_group = new_group(ranks=g)
            if self.global_rank in g:
                self.dp_group = g
                self.dp_proc_group = proc_group
            self.is_first_stage = (self.stage_id == 0)
            self.is_last_stage = (self.stage_id ==
                                  (self.pipe_parallel_size - 1))
            self.p2p_groups = self._build_p2p_groups()

            self.pp_group = []
            self.pp_proc_group = None
            self.pipe_group = self._topo.get_axis_comm_lists('pipe')

            for ranks in self.pipe_groups:
                if self.global_rank == 0:
                    pass
                proc_group = new_group(ranks=ranks)
                if self.global_rank in ranks:
                    self.pp_group = ranks
                    self.pp_proc_group = proc_group
            assert self.pp_proc_group is not None

    def get_data_parallel_id(self):
        return self._topo.get_coord(rank=self.global_rank).data

    def _is_grid_valid(self):
        ranks = 1
        for ax in self._topo:
            pass


class CollectiveHelper(object):
    def __init__(self, role_maker, nrings=1, wait_port='6174'):
        self.nrings = nrings
        self.wait_port = wait_port
        self.role_maker = role_maker

    def update_startup_program(self, startup_program=None):
        self.startup_program = startup_program
        if startup_program is None:
            self.startup_program = fluid.default_startup_program()

        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[self.role_maker._worker_index()]
        for ring_id in range(self.nrings):
            self._init_communicator(
                self.startup_program, current_endpoint, endpoints,
                self.role_maker._worker_index(), ring_id, self.wait_port)
        self._broadcast_params()

    def _init_communicator(self, program, current_endpoint, endpoints, rank,
                           ring_id, wait_port):
        nranks = len(endpoints)
        other_endpoints = endpoints[:]
        other_endpoints.remove(current_endpoint)
        if rank == 0 and wait_port:
            wait_server_ready(other_endpoints)

        block = program.global_block()
        if core.is_compiled_with_cuda():
            comm_id_var = block.create_var(
                name=unique_name.generate('nccl_id'),
                persistable=True,
                type=core.VarDesc.VarType.RAW)
            block.append_op(
                type='c_gen_nccl_id',
                inputs={},
                outputs={'Out': comm_id_var},
                attrs={
                    'rank': rank,
                    'endpoint': current_endpoint,
                    'other_endpoints': other_endpoints,
                    OP_ROLE_KEY: OpRole.Forward
                })
            block.append_op(
                type='c_comm_init',
                inputs={'X': comm_id_var},
                outputs={},
                attrs={
                    'nranks': nranks,
                    'rank': rank,
                    'ring_id': ring_id,
                    OP_ROLE_KEY: OpRole.Forward
                })
        elif core.is_compiled_with_xpu():
            comm_id_var = block.create_var(
                name=unique_name.generate('bkcl_id'),
                persistable=True,
                type=core.VarDesc.VarType.RAW)
            block.append_op(
                type='c_gen_bkcl_id',
                inputs={},
                outputs={'Out': comm_id_var},
                attrs={
                    'rank': rank,
                    'endpoint': current_endpoint,
                    'other_endpoints': other_endpoints,
                    OP_ROLE_KEY: OpRole.Forward
                })
            block.append_op(
                type='c_comm_init',
                inputs={'X': comm_id_var},
                outputs={},
                attrs={
                    'nranks': nranks,
                    'rank': rank,
                    'ring_id': ring_id,
                    OP_ROLE_KEY: OpRole.Forward
                })
        else:
            raise ValueError(
                "comm_id must be generated in paddlepaddle-xpu or paddlepaddle-xpu."
            )

    def _wait(self, current_endpoint, endpoints):
        assert (self.wait_port)
        other_endpoints = endpoints[:]
        other_endpoints.remove(current_endpoint)
        wait_server_ready(other_endpoints)

    def _broadcast_params(self):
        block = self.startup_program.global_block()
        ring_id = -1
        for param in block.iter_parameters():
            if param.is_distributed:
                continue

            ring_id = (ring_id + 1) % self.nrings
            block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': ring_id,
                    'root': 0,
                    OP_ROLE_KEY: OpRole.Forward
                })

        for ring_id in range(self.nrings):
            block.append_op(
                type='c_sync_comm_stream',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={'ring_id': ring_id,
                       OP_ROLE_KEY: OpRole.Forward})
