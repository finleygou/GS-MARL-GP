#!/bin/bash

# Run the script
seed_max=1
n_agents=6
ep_lens=150
use_curriculum="False"

for seed in $(seq ${seed_max});
do
    echo "seed: ${seed}"
    # execute the script with different params
    python ../onpolicy/scripts/eval_mpe.py \
    --use_valuenorm --use_popart \
    --project_name "GS_GP" \
    --env_name "GraphMPE" \
    --algorithm_name "rmappo" \
    --seed ${seed} \
    --experiment_name "check" \
    --scenario_name "graph_navigation_6agts" \
    --hidden_size 128 \
    --layer_N 2 \
    --use_wandb "False" \
    --save_gifs "False" \
    --use_render "True" \
    --save_data "False" \
    --use_curriculum "False" \
    --use_policy "False" \
    --gp_type "navigation" \
    --num_target 6 \
    --num_agents 6 \
    --num_obstacle 6 \
    --num_dynamic_obs 0 \
    --n_rollout_threads 1 \
    --use_lstm "True" \
    --episode_length ${ep_lens} \
    --ppo_epoch 15 --use_ReLU --gain 0.01 \
    --user_name "finleygou" \
    --use_cent_obs "False" \
    --graph_feat_type "relative" \
    --use_att_gnn "False" \
    --monte_carlo_test "False" \
    --render_episodes 20 \
    --model_dir "/data/goufandi_space/Projects/GS-MARL-GP/GS-MARL-GP/onpolicy/results/GraphMPE/graph_navigation_6agts/rmappo/check/wandb/run-20250901_154440-eas9t02l/files/"
done