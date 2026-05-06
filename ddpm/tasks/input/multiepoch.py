import torch
from torch import Tensor as _T

from abc import ABC, abstractmethod
from typing import Dict, Set, List, Tuple

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles


default_dtype = torch.float32


class MultiEpochSensoryGenerator(ABC):
    """
    Can generate sensory inputs as vectoral data or as images

    Outputs are tensor with shape [... self.prep_sensory_shape[i]] or [... self.diffusion_sensory_shapes[i]]

    self.underlying_sensory_shape is used to design the input block

    XXX images only really works with time varying case!
    """

    prep_sensory_shape: List[List[int]]
    diffusion_sensory_shapes: List[List[int]]
    underlying_sensory_shape: List[int]
    required_task_variable_keys: Set[str]

    task_metadata = {}

    @abstractmethod
    def generate_prep_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        raise NotImplementedError

    @abstractmethod
    def generate_diffusion_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        raise NotImplementedError


class DelayedIndexCuingSensoryGeneratorWithMemory(MultiEpochSensoryGenerator):
    """
    Prep epoch 1: provide report dimensions as cartestians
    Prep epoch 2: nothing
    Prep epoch 3: provide just the index
    Prep epoch 4: nothing
    Diffusion epoch 1: nothing (but pass as zeros of same shape as prep epoch 1)
    """

    def __init__(self, num_items: int) -> None:
        self.required_task_variable_keys = {"report_features_cart", "cued_item_idx"}
        self.num_items = num_items
        self.underlying_sensory_shape = [num_items * 2]
        self.prep_sensory_shape = [[num_items * 2], [num_items * 2], [1], [num_items * 2]]
        self.diffusion_sensory_shapes = [[num_items * 2]]

    def generate_prep_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["report_features_cart"].shape[0]
        assert tuple(variable_dict["report_features_cart"].shape) == (
            batch_size,
            self.num_items,
            2,
        )
        flattened_coords = variable_dict["report_features_cart"].reshape(
            batch_size, -1
        )  # on each batch row: x1, y1, x2, y2, ... --> [batch, num_items * 2]
        index = variable_dict["cued_item_idx"].unsqueeze(-1)
        empty = torch.zeros(batch_size, self.num_items * 2)
        rets = [flattened_coords, empty, index, empty]
        return [ret.to(default_dtype) for ret in rets]

    def generate_diffusion_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["report_features_cart"].shape[0]
        return [torch.zeros(batch_size, self.num_items * 2, dtype = default_dtype)]



class IndexFirstCuingSensoryGeneratorWithMemory(DelayedIndexCuingSensoryGeneratorWithMemory):
    """
    Cue‑first variant:
    Prep epoch 1: provide just the index (shape [1])
    Prep epoch 2: empty (zeros matching report coords)
    Prep epoch 3: provide report_features_cart flattened ([num_items * 2])
    Prep epoch 4: empty (zeros)
    Diffusion epoch 1: zeros of same shape as the underlying sensory shape (report coords)
    """
    def __init__(self, num_items: int) -> None:
        self.required_task_variable_keys = {"report_features_cart", "cued_item_idx"}
        self.num_items = num_items
        # underlying sensory shape corresponds to the report coords vector
        self.underlying_sensory_shape = [num_items * 2]
        # prep epochs shapes: [index], [empty(report-sized)], [report coords], [empty(report-sized)]
        self.prep_sensory_shape = [[1], [num_items * 2], [num_items * 2], [num_items * 2]]
        self.diffusion_sensory_shapes = [[num_items * 2]]

    def generate_prep_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["report_features_cart"].shape[0]
        assert tuple(variable_dict["report_features_cart"].shape) == (
            batch_size,
            self.num_items,
            2,
        )
        flattened_coords = variable_dict["report_features_cart"].reshape(batch_size, -1).to(default_dtype)
        index = variable_dict["cued_item_idx"].unsqueeze(-1).to(default_dtype)  # [B, 1]
        empty = torch.zeros(batch_size, self.num_items * 2, dtype=default_dtype)
        # cue first: index, empty, report coords, empty
        return [index, empty, flattened_coords, empty]


class JointIndexCuingSensoryGeneratorWithMemory(MultiEpochSensoryGenerator):
    """
    Joint index-cuing generator that normalises the cue-first and cue-delayed
    variants to the same four-epoch input shape.

    Epoch layout:
    - mode 0: index, empty, report coords, empty
    - mode 1: report coords, empty, index, empty

    The index is encoded as a one-hot vector in the first num_items slots of a
    num_items * 2 dimensional input, so both task orders can share the same
    underlying sensory shape and the existing identity input block.
    """

    def __init__(self, num_items: int) -> None:
        self.required_task_variable_keys = {
            "report_features_cart",
            "cued_item_idx",
            "task_mode",
        }
        self.num_items = num_items
        self.underlying_sensory_shape = [num_items * 2]
        self.prep_sensory_shape = [[num_items * 2]] * 4
        self.diffusion_sensory_shapes = [[num_items * 2]]

    def _index_to_vector(self, cued_item_idx: _T) -> _T:
        batch_size = cued_item_idx.shape[0]
        index_vector = torch.zeros(batch_size, self.num_items * 2, dtype=default_dtype)
        index_vector[torch.arange(batch_size), cued_item_idx.long()] = 1.0
        return index_vector

    def _mode_mask(self, variable_dict: Dict[str, _T]) -> _T:
        task_mode = variable_dict["task_mode"]
        if task_mode.ndim == 0:
            task_mode = task_mode[None]
        assert task_mode.shape[0] == variable_dict["cued_item_idx"].shape[0]
        return task_mode.long() == 0

    def generate_prep_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["report_features_cart"].shape[0]
        assert tuple(variable_dict["report_features_cart"].shape) == (
            batch_size,
            self.num_items,
            2,
        )
        flattened_coords = variable_dict["report_features_cart"].reshape(batch_size, -1).to(default_dtype)
        index_vector = self._index_to_vector(variable_dict["cued_item_idx"])
        empty = torch.zeros(batch_size, self.num_items * 2, dtype=default_dtype)
        index_first_mask = self._mode_mask(variable_dict)

        first_epoch = torch.where(index_first_mask[:, None], index_vector, flattened_coords)
        third_epoch = torch.where(index_first_mask[:, None], flattened_coords, index_vector)
        return [first_epoch, empty, third_epoch, empty]

    def generate_diffusion_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["report_features_cart"].shape[0]
        return [torch.zeros(batch_size, self.num_items * 2, dtype=default_dtype)]


class DelayedProbeCuingSensoryGeneratorWithMemory(MultiEpochSensoryGenerator):
    """
    Prep epoch 1: provide report and probe dimensions as cartestians, zero at end of input vector
    Prep epoch 2: nothing (but pass as zeros of same shape as prep epoch 1)
    Prep epoch 3: provide just the probe cartesians at end of input vector
    Prep epoch 4: nothing (but pass as zeros of same shape as prep epoch 1)
    Diffusion epoch 1: nothing (but pass as zeros of same shape as prep epoch 1)
    """

    def __init__(self, num_items: int, cue_with_report: bool) -> None:
        self.required_task_variable_keys = {"report_features_cart", "probe_features_cart", "cued_item_idx"}
        self.num_items = num_items
        self.underlying_sensory_shape = [num_items * 4 + 2]
        self.prep_sensory_shape = [self.underlying_sensory_shape] * 4
        self.diffusion_sensory_shapes = [self.underlying_sensory_shape]
        self.cue_with_report = cue_with_report

    def generate_prep_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["report_features_cart"].shape[0]
        assert tuple(variable_dict["report_features_cart"].shape) == tuple(variable_dict["probe_features_cart"].shape) == (
            batch_size,
            self.num_items,
            2,
        )
        flattened_report_coords = variable_dict["report_features_cart"].reshape(
            batch_size, -1
        )  # on each batch row: x1, y1, x2, y2, ... --> [batch, num_items * 2]
        flattened_probe_coords = variable_dict["probe_features_cart"].reshape(
            batch_size, -1
        )  # on each batch row: x1, y1, x2, y2, ... --> [batch, num_items * 2]
        empty_probe = torch.zeros(batch_size, 2)
        flattened_coords_with_empty = torch.concat([flattened_report_coords, flattened_probe_coords, empty_probe], -1)
        index = variable_dict["cued_item_idx"]
        empty_features = torch.zeros(batch_size, self.num_items * 4)
        if 'override_cue_features_cart' in variable_dict:
            chosen_probe_coords = variable_dict['override_cue_features_cart']
            print('override_cue_features_cart being used by DelayedProbeCuingSensoryGeneratorWithMemory.generate_prep_sensory_inputs! This should not be used for training ever!')
        elif self.cue_with_report:
            chosen_probe_coords = variable_dict["report_features_cart"][range(batch_size), index]
        else:
            chosen_probe_coords = variable_dict["probe_features_cart"][range(batch_size), index]
        chosen_probe_coords = torch.concat([empty_features, chosen_probe_coords], -1)
        empty = torch.zeros(batch_size, self.num_items * 4 + 2)
        return [flattened_coords_with_empty, empty, chosen_probe_coords, empty]

    def generate_diffusion_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["report_features_cart"].shape[0]
        return [torch.zeros(batch_size, self.num_items * 4 + 2)]



class DelayedProbeCuingSensoryGeneratorWithMemoryPalimpsest(DelayedProbeCuingSensoryGeneratorWithMemory):

    palimpsest_feature_names = ['probe', 'report']

    def __init__(self, num_items: int, probe_num_tc: int, report_num_tc: int, probe_tc_width: int, report_tc_width: int, vectorise_input: bool = True) -> None:
        super().__init__(num_items, False)
        self.required_task_variable_keys = {"report_features", "probe_features", "cued_item_idx"}

        self.probe_num_tc = probe_num_tc
        self.report_num_tc = report_num_tc

        self.vectorise_input = vectorise_input

        if vectorise_input:
            self.underlying_sensory_shape = [self.probe_num_tc * self.report_num_tc]
        else:
            self.underlying_sensory_shape = [self.probe_num_tc, self.report_num_tc]

        self.prep_sensory_shape = [self.underlying_sensory_shape] * 4
        self.diffusion_sensory_shapes = [self.underlying_sensory_shape]

        self.probe_centers = torch.linspace(-torch.pi, +torch.pi, probe_num_tc + 1)[:-1]
        self.report_centers = torch.linspace(-torch.pi, +torch.pi, report_num_tc + 1)[:-1]
        self.probe_tuning_scales = torch.ones_like(self.probe_centers) * probe_tc_width
        self.report_tuning_scales = torch.ones_like(self.report_centers) * report_tc_width
    
    @property
    def num_tcs(self) -> Tuple[int]:
        return tuple(getattr(self, f'{pfn}_num_tc') for pfn in self.palimpsest_feature_names)

    @staticmethod
    def generate_responses(features: _T, centers: _T, scales: _T, peak: float) ->_T:
        """
        features [B, N]
        centers [D]
        scales [D]

        output of shape [B, N, D]

        r_i(a) ~= peak * exp(cos(a - a_i) * scale) / exp(scale)
        """
        centers = centers[None,None]
        scales = scales[None,None]
        features = features[:,:,None]
        scaled_diffs = rectify_angles(centers - features).cos() * scales
        rescaled_diffs = scaled_diffs.exp() / scales.exp()
        rescaled_diffs = rescaled_diffs * peak
        return rescaled_diffs

    def generate_joint_resps(self, repr0: _T, repr1: _T) -> _T:
        """
        XXX: shapes!
        """
        joint_repr = (repr0.unsqueeze(-2) * repr1.unsqueeze(-1))  # [B, N, probe size, report size]
        if self.vectorise_input:
            joint_resp = joint_repr.reshape(joint_repr.shape[0], self.num_items, -1)  # [B, total size]
        else:
            joint_resp = joint_repr
        joint_resp = joint_resp.sum(1)
        return joint_resp

    def generate_prep_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:

        probe_repr = self.generate_responses(variable_dict["probe_features"], self.probe_centers, self.probe_tuning_scales, 1.0)        # [B, N, probe size]
        report_repr = self.generate_responses(variable_dict["report_features"], self.report_centers, self.report_tuning_scales, 1.0)
        flat_report_repr = torch.ones_like(report_repr).unsqueeze(-1)

        joint_resp = self.generate_joint_resps(probe_repr, report_repr)

        if 'override_cue_features' in variable_dict:
            print('override_cue_features being used by DelayedProbeCuingSensoryGeneratorWithMemoryPalimpsest.generate_prep_sensory_inputs! This should not be used for training ever!')
            overriden_probe_repr = self.generate_responses(variable_dict["override_cue_features"], self.probe_centers, self.probe_tuning_scales, 1.0)        # [B, N, probe size]
            overriden_probe_repr = overriden_probe_repr.unsqueeze(-2)
            cue_repr = (overriden_probe_repr * flat_report_repr)  # [B, N, probe size, report size]

        else:
            cue_repr = (probe_repr.unsqueeze(-2) * flat_report_repr)  # [B, N, probe size, report size]

        if self.vectorise_input:
            cue_resp = cue_repr.reshape(cue_repr.shape[0], self.num_items, -1)  # [B, total size]
        else:
            cue_resp = cue_repr
        cue_resp = cue_resp[torch.arange(cue_resp.shape[0]),variable_dict["cued_item_idx"]]

        empty = torch.zeros_like(cue_resp)

        # return [flattened_coords_with_empty, empty, chosen_probe_coords, empty]
        return [joint_resp, empty, cue_resp, empty]

    def generate_diffusion_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["report_features_cart"].shape[0]
        return [torch.zeros(batch_size, *self.underlying_sensory_shape)]



class DelayedAmbiguousProbeCuingSensoryGeneratorWithMemoryPalimpsest(DelayedProbeCuingSensoryGeneratorWithMemoryPalimpsest):

    palimpsest_feature_names = ['feature0', 'feature1']

    def __init__(self, num_items: int, feature0_num_tc: int, feature1_num_tc: int, feature0_tc_width: int, feature1_tc_width: int, vectorise_input: bool = True) -> None:
        super(DelayedProbeCuingSensoryGeneratorWithMemoryPalimpsest, self).__init__(num_items, False)
        self.required_task_variable_keys = {"feature0", "feature1", "cued_item_idx"}

        # XXX: WET!

        self.feature0_num_tc = feature0_num_tc
        self.feature1_num_tc = feature1_num_tc

        self.vectorise_input = vectorise_input

        if vectorise_input:
            self.underlying_sensory_shape = [self.feature0_num_tc * self.feature1_num_tc]
        else:
            self.underlying_sensory_shape = [self.feature0_num_tc, self.feature1_num_tc]

        self.prep_sensory_shape = [self.underlying_sensory_shape] * 4
        self.diffusion_sensory_shapes = [self.underlying_sensory_shape]

        self.probe_centers = torch.linspace(-torch.pi, +torch.pi, feature0_num_tc + 1)[:-1]
        self.report_centers = torch.linspace(-torch.pi, +torch.pi, feature1_num_tc + 1)[:-1]
        self.probe_tuning_scales = torch.ones_like(self.probe_centers) * feature0_tc_width
        self.report_tuning_scales = torch.ones_like(self.report_centers) * feature1_tc_width

    def generate_prep_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        
        feature0_repr = self.generate_responses(variable_dict["feature0"], self.probe_centers, self.probe_tuning_scales, 1.0)        # [B, N, feature 0 size]
        feature1_repr = self.generate_responses(variable_dict["feature1"], self.report_centers, self.report_tuning_scales, 1.0)
        probing_feature_idx = variable_dict["probing_feature_idx"]
        joint_resp = self.generate_joint_resps(feature0_repr, feature1_repr)

        cue_repr = torch.zeros(*feature0_repr.shape, feature1_repr.shape[-1], device = feature1_repr.device) # [B, N, feature 0 size, feature 1 size]
        cue_repr[probing_feature_idx == 0] = feature0_repr[probing_feature_idx == 0].unsqueeze(-2) * torch.ones_like(feature1_repr[probing_feature_idx == 0]).unsqueeze(-1)
        cue_repr[probing_feature_idx == 1] = feature1_repr[probing_feature_idx == 1].unsqueeze(-1) * torch.ones_like(feature0_repr[probing_feature_idx == 1]).unsqueeze(-2)
        
        if self.vectorise_input:
            cue_resp = cue_repr.reshape(cue_repr.shape[0], self.num_items, -1)  # [B, total size]
        else:
            cue_resp = cue_repr
        cue_resp = cue_resp[torch.arange(cue_resp.shape[0]),variable_dict["cued_item_idx"]] # [B, size0, size1]

        # from matplotlib import pyplot as plt
        # fig, axes = plt.subplots(len(cue_resp), 2, figsize = (10, 5 * len(cue_resp)))
        # for pfi, axs, jj, cc in zip(probing_feature_idx.cpu().numpy(), axes, joint_resp.cpu().numpy(), cue_resp.cpu().numpy()):
        #     axs[0].imshow(jj)
        #     axs[1].imshow(cc)
        #     axs[1].set_title(pfi)
        # fig.savefig('asdf')

        empty = torch.zeros_like(cue_resp)

        # return [flattened_coords_with_empty, empty, chosen_probe_coords, empty]
        return [joint_resp, empty, cue_resp, empty]


    def generate_diffusion_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        # XXX: WET!
        batch_size = variable_dict["feature0"].shape[0]
        return [torch.zeros(batch_size, *self.underlying_sensory_shape)]
