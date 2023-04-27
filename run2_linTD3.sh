algo=TD3
# linear_range = (100000)
# gpu_id=(6 6 6 7 5 7)
# env_id=(HalfCheetah Hopper Walker2d Swimmer Ant Humanoid)
# gin_id=(HalfCheetah Hopper Walker2d Swimmer Ant Humanoid)
# seed_id=(0 0 0 0 0 0)

# gpu_id=(7 6 5 3 1)
# env_id=(HalfCheetah Hopper Walker2d Swimmer Ant)
# gin_id=(HalfCheetah Hopper Walker2d Swimmer Ant)
# seed_id=(0 0 0 0 0)


# gpu_id=(4)
# env_id=(Swimmer)
# gin_id=(Swimmer)
# seed_id=(0)

gpu_id=(0)
env_id=(Humanoid)
gin_id=(Humanoid)
seed_id=(0)

# gpu_id=(4 4 4)
# env_id=(Hopper Walker2d Ant)
# gin_id=(Hopper Walker2d Ant)
# seed_id=(0 0 0)


for ((i=0;i<${#gpu_id[@]};i++))
do
    CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_linTD3.py \
                    --policy ${algo}linear \
                    --env ${env_id[i]}-v2 \
                    --gin ./gins_${algo}/${gin_id[i]}.gin \
                    --seed ${seed_id[i]} \
                    --fourier_type dtft \
                    --dim_discretize 128 \
                    --use_projection \
                    --projection_dim 512 \
                    --pre_train_step 1000 \
                    --target_update_freq 1000 \
                    --cosine_similarity \
                    --tau 0.01 \
                    --save_model \
                    --dir-root "./output_${algo}" \
                    --remark "tf-${env_id[i]}, FoSta, without update_every, add linear noise, weight_init_orthogonal, (400, 300) TD3, per_train_step" \
                    > ./my_log/exp_${algo}linear_${env_id[i]}_fourier1.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_ofePaper_linTD3.py \
    #                 --policy ${algo}small \
    #                 --env ${env_id[i]}-v2 \
    #                 --gin ./gins_${algo}/${gin_id[i]}.gin \
    #                 --seed ${seed_id[i]} \
    #                 --save_model \
    #                 --dir-root "./output_${algo}" \
    #                 > ./my_log/exp_small${algo}_${env_id[i]}_ofePaper.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_ofePaper.py \
    #                 --policy ${algo} \
    #                 --env ${env_id[i]}-v2 \
    #                 --seed ${seed_id[i]} \
    #                 --save_model \
    #                 --dir-root "./output_${algo}" \
    #                 > ./my_log/exp_${algo}_${env_id[i]}_raw.log 2>&1 &
    
done