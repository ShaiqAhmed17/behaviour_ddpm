import numpy as np
from ddpm.utils.vis.style import set_publication_style

from typing import List
from ddpm.tasks.main.multiepoch import MultiEpochDiffusionTask
from ddpm.tasks.main.multiepoch import MultiepochTrialInformation

from numpy import ndarray as _A
from torch import Tensor as _T

from matplotlib.pyplot import Axes
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


set_publication_style()


def symmetrize_and_square_axis(axes, min_size=None):
    x_max = np.max(np.abs(axes.get_xlim()))
    y_max = np.max(np.abs(axes.get_ylim()))
    xy_max = max(x_max, y_max)
    if min_size is not None:
        xy_max = max(xy_max, min_size)
    axes.set_ylim(ymin=-xy_max, ymax=xy_max)
    axes.set_xlim(xmin=-xy_max, xmax=xy_max)


def plot_mse_losses(
    mse_ax: Axes, mean_mse_ax: Axes, zoomed_mse_ax: Axes, zoomed_mean_mse_ax: Axes,
    training_step: int, plotting_start: int, num_timesteps: int, diffusion_cmap, trial_type_name: str,
    all_individual_residual_mses: _A
):
    mse_ax.set_title(f"Individual timesteps MSE\n{trial_type_name}")
    zoomed_mse_ax.set_title(f'Individual timesteps MSE (zoomed in)\n{trial_type_name}')
    mean_mse_ax.set_title(f"MSE averaged over timesteps\n{trial_type_name}")
    zoomed_mean_mse_ax.set_title(f"MSE averaged over timesteps (zoomed in)\n{trial_type_name}")

    if training_step > plotting_start:
        zoomed_start = int((training_step + 1 - plotting_start) / 3)
        for h, trace in enumerate(
            all_individual_residual_mses[: training_step + 1 - plotting_start].T
        ):
            color = diffusion_cmap.to_rgba(h + 1)
            mse_ax.plot(trace, color=color)
            zoomed_mse_ax.plot(trace[zoomed_start:], color=color)
        mean_mse_ax.plot(
            all_individual_residual_mses[: training_step + 1 - plotting_start].mean(-1)
        )
        zoomed_mean_mse_ax.plot(
            all_individual_residual_mses[zoomed_start : training_step + 1 - plotting_start].mean(-1)
        )

    cax = inset_axes(mse_ax, width="30%", height=1.0, loc=1)
    plt.colorbar(
        diffusion_cmap,
        cax=cax,
        ticks=range(1, num_timesteps, 10),
        orientation="vertical",
    )


def plot_standard_losses_multiepoch(
    mse_ax: Axes, mean_mse_ax: Axes, zoomed_mse_ax: Axes, zoomed_mean_mse_ax: Axes, prep_state_reg_axes: Axes, delay_activity_reg_axes: Axes,
    training_step: int, plotting_start: int, num_timesteps: int, diffusion_cmap, trial_type_name: str,
    all_individual_residual_mses: _A, all_prep_state_losses: _A, all_delay_activity_losses: _A
):
    mse_ax.set_title(f"Individual timesteps MSE\n{trial_type_name}")
    zoomed_mse_ax.set_title(f'Individual timesteps MSE (zoomed in)\n{trial_type_name}')
    mean_mse_ax.set_title(f"MSE averaged over timesteps\n{trial_type_name}")
    zoomed_mean_mse_ax.set_title(f"MSE averaged over timesteps (zoomed in)\n{trial_type_name}")
    prep_state_reg_axes.set_title(f"Prep state behaviour activity regulariser\n{trial_type_name}")
    delay_activity_reg_axes.set_title(f"Delay activity square mag\n{trial_type_name}")

    plot_mse_losses(
        mse_ax, mean_mse_ax, zoomed_mse_ax, zoomed_mean_mse_ax,
        training_step, plotting_start, num_timesteps, 
        diffusion_cmap, trial_type_name, all_individual_residual_mses
    )

    if training_step > plotting_start:
        prep_state_reg_axes.plot(all_prep_state_losses[: training_step + 1 - plotting_start])

    if training_step > plotting_start:
        delay_activity_reg_axes.plot(all_delay_activity_losses[: training_step + 1 - plotting_start])



def scatter_standard_responses_multiepoch(
    real_samples_axes: Axes, 
    generated_samples_axes: Axes,
    early_predictions_axes: Axes,
    start_of_prep_axes: Axes,
    pred_residual_axes: Axes,
    task_variable_axes: List[Axes],
    novel_samples_dict: dict,
    novel_samples_prep_dicts: List[dict],
    forward_process_dict: dict,
    task: MultiEpochDiffusionTask,
    trial_info: MultiepochTrialInformation,
    diffusion_cmap,
    trial_type_name: str,
    batch_idx: int
):

    real_samples_axes.set_title(f"Real sample(s)\n{trial_type_name}")
    generated_samples_axes.set_title(f"Generated sample(s)\n{trial_type_name}")
    early_predictions_axes.set_title(f"Early predictions of $x_0$\n{trial_type_name}")
    start_of_prep_axes.set_title(
        f"Samples from base distribution vs\nactual pre-diffusion behaviour states\n{trial_type_name}"
    )
    pred_residual_axes.set_title(f"$\hat\epsilon$ during generation\n{trial_type_name}")

    task.sample_gen.display_samples(
        trial_info.sample_information, real_samples_axes, batch_idx = batch_idx
    )
    task.sample_gen.display_samples(novel_samples_dict["samples"], generated_samples_axes, batch_idx = batch_idx)
    task.sample_gen.display_early_x0_pred_timeseries(
        novel_samples_dict["early_x0_preds"], early_predictions_axes, diffusion_cmap, batch_idx = batch_idx
    )
    task.sample_gen.display_samples(
        novel_samples_prep_dicts[-1]["postprep_base_samples"].detach().cpu(),
        axes = start_of_prep_axes,
        batch_idx = batch_idx,
    )
    task.sample_gen.display_samples(
        forward_process_dict["x_t"][:, :, -1, ...], start_of_prep_axes, batch_idx = batch_idx
    )
    task.sample_gen.display_early_x0_pred_timeseries(
        novel_samples_dict["epsilon_hat"].detach().cpu(),
        pred_residual_axes,
        diffusion_cmap,
        batch_idx = batch_idx
    )

    task.task_variable_gen.display_task_variables(
        trial_info.task_variable_information, *task_variable_axes, batch_idx = batch_idx
    )
    symmetrize_and_square_axis(real_samples_axes)
    symmetrize_and_square_axis(generated_samples_axes)
    symmetrize_and_square_axis(early_predictions_axes)
    symmetrize_and_square_axis(start_of_prep_axes)
    symmetrize_and_square_axis(pred_residual_axes)




def imshow_palimpsest_reprs(all_axes: List[Axes], trial_info: MultiepochTrialInformation, task: MultiEpochDiffusionTask, all_prep_epoch_indices: List[int], batch_idx: int):
    assert len(all_axes) == len(all_prep_epoch_indices)
    for ax, ind in zip(all_axes, all_prep_epoch_indices):
        ax.imshow(trial_info.prep_network_inputs[ind][batch_idx,0].cpu().reshape(task.sensory_gen.num_tcs[0], task.sensory_gen.num_tcs[1]))

