#!/bin/bash
set -e
# Run the script
seed_max=1
n_agents=5
ep_lens=200
use_curriculum="False"

for seed in $(seq ${seed_max});
do
    echo "seed: ${seed}"
    # execute the script with different params
    python ../onpolicy/scripts/eval_mpe.py \
    --use_valuenorm --use_popart \
    --project_name "GS_GP" \
    --env_name "GSMPE" \
    --algorithm_name "rmappo" \
    --seed ${seed} \
    --experiment_name "check" \
    --scenario_name "graph_formation_5agts" \
    --hidden_size 64 \
    --layer_N 1 \
    --use_wandb "False" \
    --save_gifs "False" \
    --use_render "True" \
    --save_data "False" \
    --use_curriculum "False" \
    --use_policy "False" \
    --gp_type "formation" \
    --num_target 0 \
    --num_agents 5 \
    --num_obstacle 4 \
    --num_dynamic_obs 4 \
    --n_rollout_threads 1 \
    --use_lstm "True" \
    --episode_length ${ep_lens} \
    --data_chunk_length 20 \
    --ppo_epoch 15 --use_ReLU --gain 0.01 \
    --user_name "finleygou" \
    --use_cent_obs "False" \
    --graph_feat_type "relative" \
    --use_att_gnn "True" \
    --monte_carlo_test "False" \
    --render_episodes 5 \
    --model_dir "/data/goufandi_space/Projects/GS-MARL-GP/GS-MARL-GP/onpolicy/results/GSMPE/graph_formation_5agts/rmappo/check/run18/models/"
done