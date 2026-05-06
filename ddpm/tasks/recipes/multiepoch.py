from ddpm.tasks.variable.base import *
from ddpm.tasks.sample.base import *
from ddpm.tasks.main.multiepoch import *
from ddpm.tasks.input.multiepoch import *
from torch import Tensor as _T


class JointIndexCueSpikeAndSlabSwapProbabilityTaskVariableGenerator(
    SpikeAndSlabSwapProbabilityTaskVariableGenerator
):
    """
    Task variable generator for a joint index-memory model.

    The existing cue-first and cue-delayed tasks are represented with the same
    underlying sensory shape, while the sampled task_mode selects which prep
    epoch receives the index cue.
    """

    trial_types = ["index_first", "delayed_index"]

    task_variable_keys = SpikeAndSlabSwapProbabilityTaskVariableGenerator.task_variable_keys.union(
        {"task_mode"}
    )

    def __init__(
        self,
        num_items: int,
        correct_probability: float,
        stimulus_exposure_duration: int,
        pre_index_delay_duration: int | List[int],
        index_duration: int,
        post_index_delay_duration: int,
        p_index_first: float = 0.5,
    ) -> None:
        super().__init__(
            num_items=num_items,
            correct_probability=correct_probability,
            stimulus_exposure_duration=stimulus_exposure_duration,
            pre_index_delay_duration=pre_index_delay_duration,
            index_duration=index_duration,
            post_index_delay_duration=post_index_delay_duration,
        )
        assert 0.0 <= p_index_first <= 1.0
        self.p_index_first = p_index_first

    def generate_variable_dict(
        self,
        batch_size: int,
        *_,
        override_stimulus_features_dict: Optional[Dict[str, _T]] = None,
    ) -> Dict[str, _T]:
        ret = super().generate_variable_dict(
            batch_size=batch_size,
            override_stimulus_features_dict=override_stimulus_features_dict,
        )
        task_mode = torch.bernoulli(
            torch.full((batch_size,), self.p_index_first, dtype=default_dtype)
        ).long()
        ret.update(
            {
                "task_mode": task_mode,
                "trial_type_idx": task_mode,
            }
        )
        return ret

    def generate_representative_variable_dict(
        self,
        *args,
        override_stimulus_features_dict: Optional[Dict[str, _T]] = None,
    ) -> Dict[str, _T]:
        ret = super().generate_variable_dict(
            batch_size=len(self.trial_types),
            override_stimulus_features_dict=override_stimulus_features_dict,
        )
        ret["task_mode"] = torch.tensor([0, 1]).long()
        ret["trial_type_idx"] = ret["task_mode"]
        return ret



def indexing_cue_first_fixed_probability_vectoral(
    sample_size,
    num_items,
    correct_probability,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    **sample_kwargs
):
    """
    Cue-first variant: present the index/cue before the stimulus epochs,
    keeping the same SpikeAndSlab task-variable generator and vectoral samples.
    """
    task_variable_gen = SpikeAndSlabSwapProbabilityTaskVariableGenerator(
        num_items, correct_probability, stimulus_exposure_duration,
        pre_index_delay_duration, index_duration, post_index_delay_duration
    )

    # Use the cue-first sensory generator from the multiepoch input module.
    # If the exact class name differs in your codebase, replace below with the correct class.
    sensory_gen = IndexFirstCuingSensoryGeneratorWithMemory(num_items=num_items)

    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


def delayed_indexing_cue_fixed_probability_vectoral(
    sample_size,
    num_items,
    correct_probability,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    **sample_kwargs
):
    task_variable_gen = SpikeAndSlabSwapProbabilityTaskVariableGenerator(
        num_items, correct_probability, stimulus_exposure_duration, 
        pre_index_delay_duration, index_duration, post_index_delay_duration
    )
    sensory_gen = DelayedIndexCuingSensoryGeneratorWithMemory(num_items=num_items)
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


def joint_indexing_cue_fixed_probability_vectoral(
    sample_size,
    num_items,
    correct_probability,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    p_index_first: float = 0.5,
    **sample_kwargs,
):
    """
    Joint index-cuing task that samples both cue-first and cue-delayed trials.
    """
    task_variable_gen = JointIndexCueSpikeAndSlabSwapProbabilityTaskVariableGenerator(
        num_items,
        correct_probability,
        stimulus_exposure_duration,
        pre_index_delay_duration,
        index_duration,
        post_index_delay_duration,
        p_index_first=p_index_first,
    )
    sensory_gen = JointIndexCuingSensoryGeneratorWithMemory(num_items=num_items)
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )




def delayed_probe_cue_vectoral_spike_and_slab(
    sample_size,
    num_items,
    correct_probability,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    cue_with_report,
    **sample_kwargs
):
    task_variable_gen = SpikeAndSlabSwapProbabilityTaskVariableGenerator(
        num_items, correct_probability, stimulus_exposure_duration, pre_index_delay_duration, index_duration, post_index_delay_duration
    )
    sensory_gen = DelayedProbeCuingSensoryGeneratorWithMemory(num_items=num_items, cue_with_report=cue_with_report)
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )



def delayed_probe_cue_vectoral_spike_and_slab_palimpsest(
    sample_size,
    num_items,
    correct_probability,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    probe_num_tc,
    report_num_tc,
    probe_tc_width,
    report_tc_width,
    vectorise_input: bool = True,
    limit_training_timesteps: Optional[int] = None,
    **sample_kwargs
):
    task_variable_gen = SpikeAndSlabSwapProbabilityTaskVariableGenerator(
        num_items, correct_probability, stimulus_exposure_duration, pre_index_delay_duration, index_duration, post_index_delay_duration
    )
    sensory_gen = DelayedProbeCuingSensoryGeneratorWithMemoryPalimpsest(
        num_items=num_items,
        probe_num_tc=probe_num_tc,
        report_num_tc=report_num_tc,
        probe_tc_width=probe_tc_width,
        report_tc_width=report_tc_width,
        vectorise_input=vectorise_input,
    )
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, limit_training_timesteps=limit_training_timesteps, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )



def delayed_ambiguous_probe_cue_vectoral_spike_and_slab_palimpsest(
    sample_size,
    num_items,
    correct_probability,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    feature0_num_tc,
    feature1_num_tc,
    feature0_tc_width,
    feature1_tc_width,
    vectorise_input: bool = True,
    **sample_kwargs
):
    task_variable_gen = AmbiguousSpikeAndSlabSwapProbabilityTaskVariableGenerator(
        num_items, correct_probability, stimulus_exposure_duration, pre_index_delay_duration, index_duration, post_index_delay_duration
    )
    sensory_gen = DelayedAmbiguousProbeCuingSensoryGeneratorWithMemoryPalimpsest(
        num_items=num_items,
        feature0_num_tc=feature0_num_tc,
        feature1_num_tc=feature1_num_tc,
        feature0_tc_width=feature0_tc_width,
        feature1_tc_width=feature1_tc_width,
        vectorise_input=vectorise_input,
    )
    sample_gen = AmbiguousVectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )



def delayed_probe_cue_vectoral_with_swap_function(
    sample_size,
    num_items,
    swap_function_width,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    cue_with_report,
    **sample_kwargs
):
    task_variable_gen = ProbeDistanceProbabilityTaskVariableGenerator(
        num_items, swap_function_width, stimulus_exposure_duration, pre_index_delay_duration, index_duration, post_index_delay_duration
    )
    sensory_gen = DelayedProbeCuingSensoryGeneratorWithMemory(num_items=num_items, cue_with_report=cue_with_report)
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )


def delayed_probe_cue_vectoral_with_swap_function_palimpsest(
    sample_size,
    num_items,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    probe_num_tc,
    report_num_tc,
    probe_tc_width,
    report_tc_width,
    swap_function_width: Optional[float] = None,
    swap_function_width_sharp: Optional[float] = None,
    sharp_swap_func_logit: Optional[float] = None,
    swap_function_offset: float = 0.0,
    vectorise_input: bool = True,
    **sample_kwargs
):
    task_variable_gen = ProbeDistanceProbabilityTaskVariableGenerator(
        num_items = num_items,
        swap_function_width = swap_function_width,
        swap_function_width_sharp = swap_function_width_sharp,
        sharp_swap_func_logit = sharp_swap_func_logit,
        swap_function_offset = swap_function_offset,
        stimulus_exposure_duration = stimulus_exposure_duration,
        pre_index_delay_duration = pre_index_delay_duration,
        index_duration = index_duration,
        post_index_delay_duration = post_index_delay_duration,
    )
    sensory_gen = DelayedProbeCuingSensoryGeneratorWithMemoryPalimpsest(
        num_items=num_items,
        probe_num_tc=probe_num_tc,
        report_num_tc=report_num_tc,
        probe_tc_width=probe_tc_width,
        report_tc_width=report_tc_width,
        vectorise_input=vectorise_input,
    )
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )



def delayed_ambiguous_probe_cue_vectoral_with_swap_function_palimpsest(
    sample_size,
    num_items,
    swap_function_width_0,
    swap_function_width_1,
    stimulus_exposure_duration,
    pre_index_delay_duration,
    index_duration,
    post_index_delay_duration,
    feature0_num_tc,
    feature1_num_tc,
    feature0_tc_width,
    feature1_tc_width,
    vectorise_input: bool = True,
    **sample_kwargs
):
    task_variable_gen = AmbiguousProbeDistanceProbabilityTaskVariableGenerator(
        num_items=num_items,
        swap_function_width_0=swap_function_width_0,
        swap_function_width_1=swap_function_width_1,
        stimulus_exposure_duration=stimulus_exposure_duration,
        pre_index_delay_duration=pre_index_delay_duration,
        index_duration=index_duration,
        post_index_delay_duration=post_index_delay_duration
    )
    sensory_gen = DelayedAmbiguousProbeCuingSensoryGeneratorWithMemoryPalimpsest(
        num_items=num_items,
        feature0_num_tc=feature0_num_tc,
        feature1_num_tc=feature1_num_tc,
        feature0_tc_width=feature0_tc_width,
        feature1_tc_width=feature1_tc_width,
        vectorise_input=vectorise_input,
    )
    sample_gen = AmbiguousVectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size, **sample_kwargs
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )




def sequentially_presented_vectoral_even_causal_inference(
    stimulus_exposure_durations, delay_durations,
    sample_size, min_margin_div_pi: float, **sample_kwargs
):
    task_variable_gen = SequentialCausalInferenceTaskVariableGenerator(min_margin_div_pi)
    sensory_gen = SequentialJustReportSensoryGenerator(num_items=2)
    sample_gen = VectoralEmbeddedExampleSampleGenerator(
        sample_size=sample_size,
        **sample_kwargs,
        response_location_key="response_locations_cart"
    )
    return MultiEpochDiffusionTask(
        task_variable_gen=task_variable_gen,
        sensory_gen=sensory_gen,
        sample_gen=sample_gen,
    )
