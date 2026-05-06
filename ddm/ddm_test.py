from matplotlib import pyplot as plt

import os
import math
import torch

if __name__ != "__main__":
    import pytest

    pytest.skip("ddm_test.py is a training script, not a pytest module", allow_module_level=True)

from purias_utils.util.logging import configure_logging_paths

from ddpm.model.input import InputModelBlock
from ddpm.model.transition.vectoral import VectoralResidualModel

from ddm.rt_ddpm import ReactionTimeDDPM
from ddm.ddm import generate_correct_answers, generate_sensory_batch, simulate_ddm

from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

# DDM/DDPM params
num_trials = int(1e6)
max_steps = 5000
dt = 0.001
forward_sigma = math.sqrt(dt) * 0.1
batch_size = 4
num_ddm_samples = 1024
boundary = 1.0
ddpm_sigma_scaler = 5.0


sigma2x_schedule = torch.linspace(forward_sigma, forward_sigma, max_steps)

_, savepath, _ = configure_logging_paths('results_link_sampler_ext/ddm_recovery/run', [])


sensory_shape = [1]
sample_shape = [1]
recurrence_hidden_layers = [5]
time_embedding_size = 32
sigma2x_schedule = sigma2x_schedule
device = 'cuda'
residual_model_kwargs = {
    'nonlin_first': False,
    'include_time': True
}

assert len(sensory_shape) == len(sample_shape) == 1
input_model = InputModelBlock(sensory_shape, sensory_shape[0], device=device)
residual_model = VectoralResidualModel(
    sample_shape[0],
    recurrence_hidden_layers,
    sensory_shape[0],
    time_embedding_size,
    **residual_model_kwargs
)

ddpm_model = ReactionTimeDDPM(
    ddpm_sigma_scaler,
    sample_shape,
    sigma2x_schedule,
    residual_model,
    input_model,
    time_embedding_size,
    dt,
    device,
)

ddm_sigma = ddpm_model.noise_scaler_schedule[-1].cpu()


lr = 1e-2
optim = torch.optim.Adam(ddpm_model.parameters(), lr=lr)
losses = []


sch_fig, sch_axes = plt.subplots(2, 1, figsize=(5, 8))
sch_axes[0].plot(
    ddpm_model.sigma2xt_schedule.cpu().numpy(), label="sigma2xt_schedule", alpha=0.4
)
sch_axes[0].plot(ddpm_model.a_t_schedule.cpu().numpy(), label="a_t_schedule", alpha=0.4)
sch_axes[0].plot(
    ddpm_model.root_b_t_schedule.cpu().numpy(), label="root_b_t_schedule", alpha=0.4
)
sch_axes[0].plot(
    ddpm_model.noise_scaler_schedule.cpu().numpy(),
    label="noise_scaler_schedule",
    alpha=0.4,
)
sch_axes[0].plot(
    ddpm_model.base_samples_scaler_schedule.cpu().numpy(),
    label="base_samples_scaler_schedule",
    alpha=0.4,
)
sch_axes[0].plot(
    ddpm_model.residual_scaler_schedule.cpu().numpy(),
    label="residual_scaler_schedule",
    alpha=0.4,
)
sch_axes[0].legend()

sch_axes[1].set_title("Time embeddings")
sch_axes[1].imshow(ddpm_model.time_embeddings.time_embs.detach().cpu().numpy().T)

sch_fig.savefig(os.path.join(savepath, 'schedule.png'))


for t in tqdm(range(num_trials)):


    # Generate correct answers (all upper boundary for this example)
    correct_answers = generate_correct_answers(batch_size)  # All +1 (upper boundary)
    
    mu_values = torch.randn(batch_size).unsqueeze(1).repeat(1, num_ddm_samples) / 3.0
    
    # Generate sensory inputs [batch, trials]
    sensory_inputs = generate_sensory_batch(correct_answers, mu_values)
    
    # Run simulation
    trajectories, choices, reaction_times, completed = simulate_ddm(
        sensory_inputs, max_steps, dt, ddm_sigma, boundary, use_cum_sum = True
    )


    noised_decisions_dict = ddpm_model.noise(
        x_0 = trajectories[:,:,-1].float().unsqueeze(-1).cuda(),
        rts = reaction_times.unsqueeze(-1).cuda()
    )

    if t % 10 == 0:

        # DDM generative process
        fig_traj, (axes_traj_ddm, axes_traj_noising, axes_traj_generative) = plt.subplots(3, 4, figsize = (10, 10), sharex=True)
        time_axis = torch.arange(trajectories.shape[2]) * dt


        # Generate new trajectories
        novel_samples_dict = ddpm_model.generate_samples(network_input = sensory_inputs.unsqueeze(-1), base_samples = torch.zeros([4, 1024, 1]).to(device))
    

        for b in range(4):
            traj_subset = trajectories[b, :10, :].T.cpu().numpy()
            axes_traj_ddm[b].plot(time_axis.cpu().numpy(), traj_subset, alpha=0.5, linewidth=0.8)
            axes_traj_ddm[b].axhline(y=boundary, color='r', linestyle='--', label='Upper boundary')
            axes_traj_ddm[b].axhline(y=-boundary, color='b', linestyle='--', label='Lower boundary')
            axes_traj_ddm[b].axhline(y=0, color='k', linestyle='-', alpha=0.3)    
            axes_traj_ddm[b].set_xlim(0, time_axis[-1].item())
            axes_traj_ddm[b].set_ylim(-1.0, 1.0)
            axes_traj_ddm[b].set_xlabel('Time (s)')
            axes_traj_ddm[b].set_ylabel('Evidence')
            axes_traj_ddm[b].grid(True, alpha=0.3)
            axes_traj_ddm[b].set_title('DDM generative process')


            traj_subset = noised_decisions_dict['x_t'][b, :10, :, 0]
            indices_subset = noised_decisions_dict['rts_as_remaining_indices'][b, :10]
            for tjs, ids in zip(traj_subset, indices_subset):
                axes_traj_noising[b].plot(time_axis.cpu().flip(0).numpy()[ids:], tjs[ids:].cpu().numpy(), alpha=0.5, linewidth=0.8)
            axes_traj_noising[b].axhline(y=boundary, color='r', linestyle='--', label='Upper boundary')
            axes_traj_noising[b].axhline(y=-boundary, color='b', linestyle='--', label='Lower boundary')
            axes_traj_noising[b].axhline(y=0, color='k', linestyle='-', alpha=0.3)    
            axes_traj_noising[b].set_title('DDPM adjusted noising process')


            traj_subset = novel_samples_dict['sample_trajectory'][b, :10, :, 0].T.cpu().numpy()
            axes_traj_generative[b].plot(time_axis.cpu().numpy(), traj_subset, alpha=0.5, linewidth=0.8)
            axes_traj_generative[b].axhline(y=boundary, color='r', linestyle='--', label='Upper boundary')
            axes_traj_generative[b].axhline(y=-boundary, color='b', linestyle='--', label='Lower boundary')
            axes_traj_generative[b].axhline(y=0, color='k', linestyle='-', alpha=0.3)    
            axes_traj_generative[b].set_xlim(0, time_axis[-1].item())
            # axes_traj_generative[b].set_ylim(-1.0, 1.0)
            axes_traj_generative[b].set_xlabel('Time (s)')
            axes_traj_generative[b].set_ylabel('Evidence')
            axes_traj_generative[b].grid(True, alpha=0.3)
            axes_traj_generative[b].set_title('DDPM generated trajectories')


        fig_traj.savefig(os.path.join(savepath, 'trajectories.png'))

        torch.save(ddpm_model.state_dict(), os.path.join(savepath, 'ddpm_model.pt'))
        torch.save(optim.state_dict(), os.path.join(savepath, 'optim.pt'))
    


    epsilon_hat_dict = ddpm_model.residual(
        x_samples=noised_decisions_dict["x_t"],
        network_input=sensory_inputs.unsqueeze(-1),
    )

    mask = noised_decisions_dict['relevance_mask'].bool()
    mse = torch.square(epsilon_hat_dict['epsilon_hat'][mask] - noised_decisions_dict['kernel_target'][mask])
    loss = mse.mean()

    losses.append(loss.item())

    plt.figure(figsize=(8, 4))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('DDPM Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'losses.png'))
    plt.close('all')

    # Optimization step
    optim.zero_grad()
    loss.backward()
    optim.step()



    
