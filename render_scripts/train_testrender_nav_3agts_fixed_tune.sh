#!/bin/bash

seed_max=1
n_agents=3
ep_lens=100
export WANDB_BASE_URL=https://api.bandw.top

for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
python  ../onpolicy/scripts/train_mpe.py \
--use_valuenorm --use_popart \
--project_name "GS_GP" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seed} \
--experiment_name "check" \
--scenario_name "graph_navigation_fix" \
--max_edge_dist 1 \
--clip_param 0.15 --gamma 0.99 \
--hidden_size 128 --layer_N 2 \
--num_target 3 --num_agents 3 --num_obstacle 3 --num_dynamic_obs 0 \
--gp_type "navigation" \
--save_data "False" \
--reward_file_name "r_test1" \
--cost_file_name "c_test1" \
--use_policy "False" \
--use_curriculum "False" \
--guide_cp 0.01 --cp 0.01 --js_ratio 0.0 \
--use_wandb "False" \
--n_training_threads 1 --n_rollout_threads 1 \
--use_lstm "True" \
--episode_length ${ep_lens} \
--num_env_steps 2000000 \
--log_interval 1 \
--ppo_epoch 15 --use_ReLU --gain 0.01 --lr 0 --critic_lr 0 \
--use_train_render "True" \
--user_name "finleygou" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--use_att_gnn "False" \
--model_dir "/data/goufandi_space/Projects/GS-MARL-GP/GS-MARL-GP/onpolicy/results/GraphMPE/graph_navigation_fix/rmappo/check/wandb/run-20250902_163023-ugkd6zpo/files/"
done