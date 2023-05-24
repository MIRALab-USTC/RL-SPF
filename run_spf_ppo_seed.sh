algo=PPO

gpu_id=(5 7 6 6 5 7)
env_id=(HalfCheetah Hopper Walker2d Swimmer Ant Humanoid)
gin_id=(HalfCheetah Hopper Walker2d Swimmer Ant Humanoid)
seed_id=(0 0 0 0 0 0)
update_every_id=(5 150 2 200 150 1)


for ((i=0;i<${#gpu_id[@]};i++))
do
    CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_ppo.py \
                    --policy ${algo} \
                    --env ${env_id[i]}-v2 \
                    --gin ./gins/${gin_id[i]}.gin \
                    --seed ${seed_id[i]} \
                    --fourier_type dtft \
                    --dim_discretize 128 \
                    --use_projection \
                    --projection_dim 512 \
                    --pre_train_step 1000 \
                    --target_update_freq 1000 \
                    --cosine_similarity \
                    --random_collect 4000 \
                    --tau 0.01 \
                    --update_every ${update_every_id[i]} \
                    --normalizer "layer" \
                    --save_model \
                    --dir-root "./output_${algo}" \
                    --remark "tf-${env_id[i]}, SPF" \
                    > ./my_log/exp_${algo}_${env_id[i]}_fourier_s${seed_id[i]}_up${update_every_id[i]}.log 2>&1 &
done