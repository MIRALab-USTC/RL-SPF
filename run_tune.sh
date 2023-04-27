algo=TD3

# gpu_id=(6 6 6 7 5 7)
# env_id=(HalfCheetah Hopper Walker2d Swimmer Ant Humanoid)
# gin_id=(HalfCheetah Hopper Walker2d Swimmer Ant Humanoid)
# seed_id=(0 0 0 0 0 0)

# gpu_id=(7 7 5 5 5)
# env_id=(Humanoid Hopper Walker2d Swimmer Ant)
# gin_id=(Humanoid Hopper Walker2d Swimmer Ant)
# seed_id=(0 0 0 0 0)


gpu_id=(4 5 6)
env_id=(Humanoid)
gin_id=(Humanoid)
seed_id=(0)
td3_linear_range=(10000 30000 50000 80000 100000 130000 150000 180000 200000 250000 300000 400000 500000 550000 600000 700000 800000 900000)

# gpu_id=(4 4 4)
# env_id=(Hopper Walker2d Ant)
# gin_id=(Hopper Walker2d Ant)
# seed_id=(0 0 0)

# ${#gpu_id[@]}
for ((i=0;i<6;i++))
do
    CUDA_VISIBLE_DEVICES=${gpu_id[0]} nohup python -u eager_main.py \
                    --policy ${algo} \
                    --env ${env_id[0]}-v2 \
                    --gin ./gins_${algo}/${gin_id[0]}.gin \
                    --seed ${seed_id[0]} \
                    --fourier_type dtft \
                    --dim_discretize 128 \
                    --use_projection \
                    --projection_dim 512 \
                    --pre_train_step 10000 \
                    --target_update_freq 1000 \
                    --cosine_similarity \
                    --tau 0.005 \
                    --save_model \
                    --td3_linear_range ${td3_linear_range[3*i]} \
                    --dir-root "./output_${algo}" \
                    --remark "tf-${env_id[0]}, FoSta, add low and high freq loss, update_every = 50 for both extractor and policy and add linear action noise from 1 to 0.1 during 10e6" \
                    > ./my_log/exp_${algo}_${env_id[0]}_fourier.log 2>&1 &

    CUDA_VISIBLE_DEVICES=${gpu_id[1]} nohup python -u eager_main.py \
                    --policy ${algo} \
                    --env ${env_id[0]}-v2 \
                    --gin ./gins_${algo}/${gin_id[0]}.gin \
                    --seed ${seed_id[0]} \
                    --fourier_type dtft \
                    --dim_discretize 128 \
                    --use_projection \
                    --projection_dim 512 \
                    --pre_train_step 10000 \
                    --target_update_freq 1000 \
                    --cosine_similarity \
                    --tau 0.005 \
                    --save_model \
                    --td3_linear_range ${td3_linear_range[3*i+1]} \
                    --dir-root "./output_${algo}" \
                    --remark "tf-${env_id[0]}, FoSta, add low and high freq loss, update_every = 50 for both extractor and policy and add linear action noise from 1 to 0.1 during 10e6" \
                    > ./my_log/exp_${algo}_${env_id[0]}_fourier.log 2>&1 &

    CUDA_VISIBLE_DEVICES=${gpu_id[2]} nohup python -u eager_main.py \
                    --policy ${algo} \
                    --env ${env_id[0]}-v2 \
                    --gin ./gins_${algo}/${gin_id[0]}.gin \
                    --seed ${seed_id[0]} \
                    --fourier_type dtft \
                    --dim_discretize 128 \
                    --use_projection \
                    --projection_dim 512 \
                    --pre_train_step 10000 \
                    --target_update_freq 1000 \
                    --cosine_similarity \
                    --tau 0.005 \
                    --save_model \
                    --td3_linear_range ${td3_linear_range[3*i+2]} \
                    --dir-root "./output_${algo}" \
                    --remark "tf-${env_id[0]}, FoSta, add low and high freq loss, update_every = 50 for both extractor and policy and add linear action noise from 1 to 0.1 during 10e6" \
                    > ./my_log/exp_${algo}_${env_id[0]}_fourier.log 2>&1 

    # CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_ofePaper.py \
    #                 --policy ${algo} \
    #                 --env ${env_id[i]}-v2 \
    #                 --gin ./gins_${algo}/${gin_id[i]}.gin \
    #                 --seed ${seed_id[i]} \
    #                 --save_model \
    #                 --dir-root "./output_${algo}" \
    #                 > ./my_log/exp_${algo}_${env_id[i]}_ofePaper.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_ofePaper.py \
    #                 --policy ${algo} \
    #                 --env ${env_id[i]}-v2 \
    #                 --seed ${seed_id[i]} \
    #                 --save_model \
    #                 --dir-root "./output_${algo}" \
    #                 > ./my_log/exp_${algo}_${env_id[i]}_raw.log 2>&1 &
    
done