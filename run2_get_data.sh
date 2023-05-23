algo=PPO

# gpu_id=(3 2 1 0)
# env_id=(HalfCheetah Walker2d Swimmer Ant)
# gin_id=(HalfCheetah Walker2d Swimmer Ant)
# seed_id=(0 0 0 0)

# gpu_id=(7 6 5 3 1)
# env_id=(HalfCheetah Hopper Walker2d Swimmer Ant)
# gin_id=(HalfCheetah Hopper Walker2d Swimmer Ant)
# seed_id=(0 0 0 0 0)


# gpu_id=(3)
# env_id=(Swimmer)
# gin_id=(Swimmer)
# seed_id=(0)

gpu_id=(4)
env_id=(HalfCheetah)
gin_id=(HalfCheetah)
seed_id=(0)

# gpu_id=(4 4 4)
# env_id=(Hopper Walker2d Ant)
# gin_id=(Hopper Walker2d Ant)
# seed_id=(0 0 0)


for ((i=0;i<${#gpu_id[@]};i++))
do
    CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u get_data.py \
                    --policy ${algo} \
                    --env ${env_id[i]}-v2 \
                    --gin ./gins_${algo}/${gin_id[i]}.gin \
                    --seed ${seed_id[i]} \
                    --fourier_type dtft \
                    --dim_discretize 128 \
                    --use_projection \
                    --projection_dim 512 \
                    --pre_train_step 10000 \
                    --target_update_freq 1000 \
                    --cosine_similarity \
                    --tau 0.01 \
                    --save_model \
                    --aux FSP \
                    --dir-root "./output_${algo}" \
                    --remark "tf-${env_id[i]}, FoSta,  update_every and linear noise, weight_init_orthogonal, (400, 300) TD3, tau=0.01, dim_output=292" \
                    > ./my_log/get_data_exp_${algo}_${env_id[i]}_fourier.log 2>&1 &

    CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u get_data.py \
                    --policy ${algo} \
                    --env ${env_id[i]}-v2 \
                    --gin ./gins_${algo}/${gin_id[i]}.gin \
                    --seed ${seed_id[i]} \
                    --save_model \
                    --aux OFE \
                    --dir-root "./output_${algo}" \
                    > ./my_log/get_data_exp_${algo}_${env_id[i]}_ofePaper.log 2>&1 &

    CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u get_data.py \
                --policy ${algo} \
                --env ${env_id[i]}-v2 \
                --seed ${seed_id[i]} \
                --save_model \
                --aux raw \
                --dir-root "./output_${algo}" \
                > ./my_log/get_data_exp_${algo}_${env_id[i]}_raw.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=${gpu_id[i]} python -u get_data.py \
    #                 --policy ${algo} \
    #                 --env ${env_id[i]}-v2 \
    #                 --seed ${seed_id[i]} \
    #                 --save_model \
    #                 --aux raw \
    #                 --dir-root "./output_${algo}" \
                    
                    

done