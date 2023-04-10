algo=TD3

# gpu_id=(7 7 7 5 5)
# env_id=(HalfCheetah Hopper Walker2d Swimmer Ant)
# gin_id=(HalfCheetah Hopper Walker2d Swimmer Ant)
# seed_id=(0 0 0 0 0)

# gpu_id=(7 7 5 5 5)
# env_id=(Humanoid Hopper Walker2d Swimmer Ant)
# gin_id=(Humanoid Hopper Walker2d Swimmer Ant)
# seed_id=(0 0 0 0 0)


# gpu_id=(7 1)
# env_id=(Humanoid Swimmer)
# gin_id=(Humanoid Swimmer)
# seed_id=(0 0)

gpu_id=(7)
env_id=(Swimmer)
gin_id=(Swimmer)
seed_id=(0)



for ((i=0;i<${#gpu_id[@]};i++))
do
    CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main.py \
                    --policy ${algo} \
                    --env ${env_id[i]}-v2 \
                    --gin ./gins_${algo}/${gin_id[i]}.gin \
                    --seed ${seed_id[i]} \
                    --fourier_type dtft \
                    --dim_discretize 128 \
                    --use_projection \
                    --projection_dim 512 \
                    --pre_train_step 10000 \
                    --target_update_freq 100000 \
                    --cosine_similarity \
                    --tau 0.005 \
                    --save_model \
                    --dir-root "./output_${algo}" \
                    --remark "tf-${env_id[i]}, FoSta, add low and high freq loss, add select_next_actions and target network update, select_next_actions with actor_target" \
                    > ./my_log/exp_${algo}_${env_id[i]}_fourier2.log 2>&1 &

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