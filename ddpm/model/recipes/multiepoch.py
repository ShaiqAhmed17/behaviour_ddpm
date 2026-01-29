from ddpm.model.main.base import *
from ddpm.model.transition import *
from ddpm.model.input import *
from ddpm.model.main.multiepoch import (
    MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess,
    MultiPreparatoryRNNBaselineDDPMReverseProcess,
    MultiPreparatoryHVAEReverseProcess,
    MultiPreparatoryBounceNetworkHVAEReverseProcess
)


def teacher_forced_delayed_index_cue(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    num_items,
    sample_ambient_dim,
    recurrence_hidden_layers,
    time_embedding_size,
    sigma2x_schedule,
    indexing_embeddings_same_slots,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    assert all(
        [len(psp) == 1 for psp in prep_sensory_shape]
    )
    assert len(sample_shape) == 1 and len(underlying_sensory_shape) == 1
    under_input_model = InputModelBlock(
        underlying_sensory_shape, underlying_sensory_shape[0], device=device
    )  # i.e. sensory (and delay) epochs for actual stimulus information...
    input_model = AllowIndexInputModelBlock(
        under_input_model, num_items, indexing_embeddings_same_slots, device
    )  # ... cueing epoch for indexing
    residual_model = VectoralResidualModel(
        sample_ambient_dim,
        recurrence_hidden_layers,
        input_model.network_input_size,
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=sample_shape,
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    mse_key = "epsilon_hat"
    return ddpm_model, mse_key, "epsilon"



def rnn_baseline_delayed_index_cue(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    num_items,
    sample_ambient_dim,
    recurrence_hidden_layers,
    time_embedding_size,
    sigma2x_schedule,
    indexing_embeddings_same_slots,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    assert all(
        [len(psp) == 1 for psp in prep_sensory_shape]
    )
    assert len(sample_shape) == 1 and len(underlying_sensory_shape) == 1
    under_input_model = InputModelBlock(
        underlying_sensory_shape, underlying_sensory_shape[0], device=device
    )  # i.e. sensory (and delay) epochs for actual stimulus information...
    input_model = AllowIndexInputModelBlock(
        under_input_model, num_items, indexing_embeddings_same_slots, device
    )  # ... cueing epoch for indexing
    residual_model = VectoralResidualModel(
        sample_ambient_dim,
        recurrence_hidden_layers,
        input_model.network_input_size,
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryRNNBaselineDDPMReverseProcess(        
        sample_ambient_dim = sample_ambient_dim,
        sample_shape = sample_shape,
        sigma2xt_schedule = sigma2x_schedule,
        residual_model = residual_model,
        input_model = input_model,
        time_embedding_size = time_embedding_size,
        device = device,
        **ddpm_model_kwargs
    )
    mse_key = "subspace_trajectories"
    return ddpm_model, mse_key, "epsilon"



def hvae_delayed_index_cue(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    num_items,
    sample_ambient_dim,
    recurrence_hidden_layers,
    time_embedding_size,
    sigma2x_schedule,
    indexing_embeddings_same_slots,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    assert all(
        [len(psp) == 1 for psp in prep_sensory_shape]
    )
    assert len(sample_shape) == 1 and len(underlying_sensory_shape) == 1
    under_input_model = InputModelBlock(
        underlying_sensory_shape, underlying_sensory_shape[0], device=device
    )  # i.e. sensory (and delay) epochs for actual stimulus information...
    input_model = AllowIndexInputModelBlock(
        under_input_model, num_items, indexing_embeddings_same_slots, device
    )  # ... cueing epoch for indexing
    residual_model = VectoralResidualModel(
        sample_ambient_dim,
        recurrence_hidden_layers,
        input_model.network_input_size,
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryHVAEReverseProcess(        
        sample_ambient_dim = sample_ambient_dim,
        sample_shape = sample_shape,
        sigma2xt_schedule = sigma2x_schedule,
        residual_model = residual_model,
        input_model = input_model,
        time_embedding_size = time_embedding_size,
        device = device,
        **ddpm_model_kwargs
    )
    if ddpm_model.train_as_rnn:
        mse_key = "subspace_trajectories"
        mse_key_target = "unnoised_target"
    else:
        mse_key = "epsilon_hat"
        mse_key_target = 'kernel_target'
    return ddpm_model, mse_key, mse_key_target




def hvae_delayed_index_cue_axonal(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    num_items,
    sample_ambient_dim,
    branching_factors,
    time_embedding_size,
    sigma2x_schedule,
    indexing_embeddings_same_slots,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    assert all(
        [len(psp) == 1 for psp in prep_sensory_shape]
    )
    assert len(sample_shape) == 1 and len(underlying_sensory_shape) == 1
    under_input_model = InputModelBlock(
        underlying_sensory_shape, underlying_sensory_shape[0], device=device
    )  # i.e. sensory (and delay) epochs for actual stimulus information...
    input_model = AllowIndexInputModelBlock(
        under_input_model, num_items, indexing_embeddings_same_slots, device
    )  # ... cueing epoch for indexing
    residual_model = AxonalResidualModel(
        sample_ambient_dim,
        branching_factors,
        input_model.network_input_size,
        time_embedding_size
    )
    ddpm_model = MultiPreparatoryHVAEReverseProcess(        
        sample_ambient_dim = sample_ambient_dim,
        sample_shape = sample_shape,
        sigma2xt_schedule = sigma2x_schedule,
        residual_model = residual_model,
        input_model = input_model,
        time_embedding_size = time_embedding_size,
        device = device,
        **ddpm_model_kwargs
    )
    if ddpm_model.train_as_rnn:
        mse_key = "subspace_trajectories"
        mse_key_target = "unnoised_target"
    else:
        mse_key = "epsilon_hat"
        mse_key_target = 'kernel_target'
    return ddpm_model, mse_key, mse_key_target



def hvae_delayed_index_cue_dendritic(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    num_items,
    sample_ambient_dim,
    branching_factors,
    time_embedding_size,
    sigma2x_schedule,
    indexing_embeddings_same_slots,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    assert all(
        [len(psp) == 1 for psp in prep_sensory_shape]
    )
    assert len(sample_shape) == 1 and len(underlying_sensory_shape) == 1
    under_input_model = InputModelBlock(
        underlying_sensory_shape, underlying_sensory_shape[0], device=device
    )  # i.e. sensory (and delay) epochs for actual stimulus information...
    input_model = AllowIndexInputModelBlock(
        under_input_model, num_items, indexing_embeddings_same_slots, device
    )  # ... cueing epoch for indexing
    residual_model = DendriticResidualModel(
        sample_ambient_dim,
        branching_factors,
        input_model.network_input_size,
        time_embedding_size
    )
    ddpm_model = MultiPreparatoryHVAEReverseProcess(        
        sample_ambient_dim = sample_ambient_dim,
        sample_shape = sample_shape,
        sigma2xt_schedule = sigma2x_schedule,
        residual_model = residual_model,
        input_model = input_model,
        time_embedding_size = time_embedding_size,
        device = device,
        **ddpm_model_kwargs
    )
    if ddpm_model.train_as_rnn:
        mse_key = "subspace_trajectories"
        mse_key_target = "unnoised_target"
    else:
        mse_key = "epsilon_hat"
        mse_key_target = 'kernel_target'
    return ddpm_model, mse_key, mse_key_target





def hvae_bounceback_delayed_index_cue(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    num_items,
    sample_ambient_dim,
    time_embedding_size,
    sigma2x_schedule,
    indexing_embeddings_same_slots,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    assert all(
        [len(psp) == 1 for psp in prep_sensory_shape]
    )
    assert len(sample_shape) == 1 and len(underlying_sensory_shape) == 1
    under_input_model = InputModelBlock(
        underlying_sensory_shape, underlying_sensory_shape[0], device=device
    )  # i.e. sensory (and delay) epochs for actual stimulus information...
    input_model = AllowIndexInputModelBlock(
        under_input_model, num_items, indexing_embeddings_same_slots, device
    )  # ... cueing epoch for indexing
    residual_model = BouncePopulationResidualModel(
        sample_ambient_dim,
        input_model.network_input_size,
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryBounceNetworkHVAEReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=sample_shape,
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    if ddpm_model.train_as_rnn:
        mse_key = "subspace_trajectories"
        mse_key_target = "unnoised_target"
    else:
        mse_key = "epsilon_hat"
        mse_key_target = 'kernel_target'
    return ddpm_model, mse_key, mse_key_target





def teacher_forced_delayed_probe_cue(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    sample_ambient_dim,
    recurrence_hidden_layers,
    time_embedding_size,
    sigma2x_schedule,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    assert all(
        [len(psp) == 1 for psp in prep_sensory_shape]
    )
    assert len(sample_shape) == 1 and len(underlying_sensory_shape) == 1
    input_model = InputModelBlock(
        underlying_sensory_shape, underlying_sensory_shape[0], device=device
    )  # no indexing unlike above
    residual_model = VectoralResidualModel(
        sample_ambient_dim,
        recurrence_hidden_layers,
        underlying_sensory_shape[0],
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=sample_shape,
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    mse_key = "epsilon_hat"
    return ddpm_model, mse_key, "epsilon"



def rnn_baseline_delayed_probe_cue(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    sample_ambient_dim,
    recurrence_hidden_layers,
    time_embedding_size,
    sigma2x_schedule,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    assert all(
        [len(psp) == 1 for psp in prep_sensory_shape]
    )
    assert len(sample_shape) == 1 and len(underlying_sensory_shape) == 1
    input_model = InputModelBlock(
        underlying_sensory_shape, underlying_sensory_shape[0], device=device
    )  # no indexing unlike above
    residual_model = VectoralResidualModel(
        sample_ambient_dim,
        recurrence_hidden_layers,
        underlying_sensory_shape[0],
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryRNNBaselineDDPMReverseProcess(        
        sample_ambient_dim = sample_ambient_dim,
        sample_shape = sample_shape,
        sigma2xt_schedule = sigma2x_schedule,
        residual_model = residual_model,
        input_model = input_model,
        time_embedding_size = time_embedding_size,
        device = device,
        **ddpm_model_kwargs
    )
    mse_key = "subspace_trajectories"
    return ddpm_model, mse_key, "x_t"



def teacher_forced_delayed_probe_cue_factorised_palimpsest_representation(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    sample_ambient_dim,
    recurrence_hidden_layers,
    time_embedding_size,
    sigma2x_schedule,
    feature_projection_sizes,
    bias,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    # assert all(
    #     [len(psp) == 1 for psp in prep_sensory_shape]
    # ), [len(psp) for psp in prep_sensory_shape]
    assert len(sample_shape) == 1#  and len(underlying_sensory_shape) == 1
    input_model = FactorisedInputModelBlock(
        underlying_sensory_shape, feature_projection_sizes, bias = bias, device=device
    )  # no indexing unlike above
    residual_model = VectoralResidualModel(
        sample_ambient_dim,
        recurrence_hidden_layers,
        input_model.network_input_size,
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=sample_shape,
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    mse_key = "epsilon_hat"
    return ddpm_model, mse_key, "epsilon"


def hvae_delayed_probe_cue_factorised_palimpsest_representation(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    sample_ambient_dim,
    recurrence_hidden_layers,
    time_embedding_size,
    sigma2x_schedule,
    feature_projection_sizes,
    bias,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    # assert all(
    #     [len(psp) == 1 for psp in prep_sensory_shape]
    # ), [len(psp) for psp in prep_sensory_shape]
    assert len(sample_shape) == 1#  and len(underlying_sensory_shape) == 1
    input_model = FactorisedInputModelBlock(
        underlying_sensory_shape, feature_projection_sizes, bias = bias, device=device
    )  # no indexing unlike above
    residual_model = VectoralResidualModel(
        sample_ambient_dim,
        recurrence_hidden_layers,
        input_model.network_input_size,
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryHVAEReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=sample_shape,
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    if ddpm_model.train_as_rnn:
        mse_key = "subspace_trajectories"
        mse_key_target = "unnoised_target"
    else:
        mse_key = "epsilon_hat"
        mse_key_target = 'kernel_target'
    return ddpm_model, mse_key, mse_key_target


    
def hvae_delayed_probe_cue_factorised_palimpsest_representation_axonal(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    sample_ambient_dim,
    branching_factors,
    time_embedding_size,
    sigma2x_schedule,
    feature_projection_sizes,
    bias,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    # assert all(
    #     [len(psp) == 1 for psp in prep_sensory_shape]
    # ), [len(psp) for psp in prep_sensory_shape]
    assert len(sample_shape) == 1#  and len(underlying_sensory_shape) == 1
    input_model = FactorisedInputModelBlock(
        underlying_sensory_shape, feature_projection_sizes, bias = bias, device=device
    )  # no indexing unlike above
    residual_model = AxonalResidualModel(
        sample_ambient_dim,
        branching_factors,
        input_model.network_input_size,
        time_embedding_size
    )
    ddpm_model = MultiPreparatoryHVAEReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=sample_shape,
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    if ddpm_model.train_as_rnn:
        mse_key = "subspace_trajectories"
        mse_key_target = "unnoised_target"
    else:
        mse_key = "epsilon_hat"
        mse_key_target = 'kernel_target'
    return ddpm_model, mse_key, mse_key_target


def hvae_delayed_probe_cue_factorised_palimpsest_representation_dendritic(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    sample_ambient_dim,
    branching_factors,
    time_embedding_size,
    sigma2x_schedule,
    feature_projection_sizes,
    bias,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    # assert all(
    #     [len(psp) == 1 for psp in prep_sensory_shape]
    # ), [len(psp) for psp in prep_sensory_shape]
    assert len(sample_shape) == 1#  and len(underlying_sensory_shape) == 1
    input_model = FactorisedInputModelBlock(
        underlying_sensory_shape, feature_projection_sizes, bias = bias, device=device
    )  # no indexing unlike above
    residual_model = DendriticResidualModel(
        sample_ambient_dim,
        branching_factors,
        input_model.network_input_size,
        time_embedding_size
    )
    ddpm_model = MultiPreparatoryHVAEReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=sample_shape,
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    if ddpm_model.train_as_rnn:
        mse_key = "subspace_trajectories"
        mse_key_target = "unnoised_target"
    else:
        mse_key = "epsilon_hat"
        mse_key_target = 'kernel_target'
    return ddpm_model, mse_key, mse_key_target



def hvae_bounceback_delayed_probe_cue_factorised_palimpsest_representation(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    sample_ambient_dim,
    time_embedding_size,
    sigma2x_schedule,
    feature_projection_sizes,
    bias,
    device,
    ddpm_model_kwargs,
    residual_model_kwargs,
):
    # assert all(
    #     [len(psp) == 1 for psp in prep_sensory_shape]
    # ), [len(psp) for psp in prep_sensory_shape]
    assert len(sample_shape) == 1#  and len(underlying_sensory_shape) == 1
    input_model = FactorisedInputModelBlock(
        underlying_sensory_shape, feature_projection_sizes, bias = bias, device=device
    )  # no indexing unlike above
    residual_model = BouncePopulationResidualModel(
        sample_ambient_dim,
        input_model.network_input_size,
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryBounceNetworkHVAEReverseProcess(
        sample_ambient_dim=sample_ambient_dim,
        sample_shape=sample_shape,
        sigma2xt_schedule=sigma2x_schedule,
        residual_model=residual_model,
        input_model=input_model,
        time_embedding_size=time_embedding_size,
        device=device,
        **ddpm_model_kwargs
    )
    if ddpm_model.train_as_rnn:
        mse_key = "subspace_trajectories"
        mse_key_target = "unnoised_target"
    else:
        mse_key = "epsilon_hat"
        mse_key_target = 'kernel_target'
    return ddpm_model, mse_key, mse_key_target




def rnn_baseline_delayed_probe_cue_factorised_palimpsest_representation(
    prep_sensory_shape,
    underlying_sensory_shape,
    sample_shape,
    sample_ambient_dim,
    recurrence_hidden_layers,
    time_embedding_size,
    sigma2x_schedule,
    feature_projection_sizes,
    bias,
    device,
    residual_model_kwargs,
    ddpm_model_kwargs,
):
    # assert all(
    #     [len(psp) == 1 for psp in prep_sensory_shape]
    # ), [len(psp) for psp in prep_sensory_shape]
    assert len(sample_shape) == 1#  and len(underlying_sensory_shape) == 1
    input_model = FactorisedInputModelBlock(
        underlying_sensory_shape, feature_projection_sizes, bias = bias, device=device
    )  # no indexing unlike above
    residual_model = VectoralResidualModel(
        sample_ambient_dim,
        recurrence_hidden_layers,
        input_model.network_input_size,
        time_embedding_size,
        **residual_model_kwargs
    )
    ddpm_model = MultiPreparatoryRNNBaselineDDPMReverseProcess(        
        sample_ambient_dim = sample_ambient_dim,
        sample_shape = sample_shape,
        sigma2xt_schedule = sigma2x_schedule,
        residual_model = residual_model,
        input_model = input_model,
        time_embedding_size = time_embedding_size,
        device = device,
        **ddpm_model_kwargs
    )
    mse_key = "subspace_trajectories"
    return ddpm_model, mse_key, "x_t"
