#!/bin/bash

seed_max=1
n_agents=3
ep_lens=100
export WANDB_BASE_URL=https://api.bandw.top

for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
python  ../onpolicy/scripts/eval_mpe.py \
--use_valuenorm --use_popart \
--project_name "GS_GP" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seed} \
--experiment_name "check" \
--scenario_name "graph_navigation_fix" \
--hidden_size 128 \
--layer_N 2 \
--max_edge_dist 1 \
--use_wandb "False" \
--save_gifs "False" \
--use_render "True" \
--save_data "False" \
--use_curriculum "False" \
--use_policy "False" \
--gp_type "navigation" \
--num_target 3 \
--num_agents 3 \
--num_obstacle 3 \
--num_dynamic_obs 0 \
--n_rollout_threads 1 \
--use_lstm "True" \
--episode_length ${ep_lens} \
--ppo_epoch 15 --use_ReLU --gain 0.01 --lr 0 --critic_lr 0 \
--log_interval 1 \
--use_train_render "True" \
--user_name "finleygou" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--use_att_gnn "False" \
--monte_carlo_test "False" \
--num_env_steps 2000000 \
--render_episodes 20 \
--model_dir "/data/goufandi_space/Projects/GS-MARL-GP/GS-MARL-GP/onpolicy/results/GraphMPE/graph_navigation_fix/rmappo/check/wandb/run-20250902_163023-ugkd6zpo/files/"
done