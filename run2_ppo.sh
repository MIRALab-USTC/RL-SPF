algo=PPO

# gpu_id=(7 2 5 4 3 6)
# env_id=(HalfCheetah Hopper Walker2d Swimmer Ant Humanoid)
# gin_id=(HalfCheetah Hopper Walker2d Swimmer Ant Humanoid)
# seed_id=(0 0 0 0 0 0)

# gpu_id=(3 4 5 1 4)
# gpu_id=(7 6 6 5 5)
# env_id=(Hopper Walker2d Swimmer Ant Humanoid)
# gin_id=(Hopper Walker2d Swimmer Ant Humanoid)
# seed_id=(0 0 0 0 0)


# gpu_id=(5)
# env_id=(Humanoid)
# gin_id=(Humanoid)
# seed_id=(0) 


# gpu_id=(6 6 6 6)
# env_id=(HalfCheetah HalfCheetah HalfCheetah HalfCheetah)
# gin_id=(HalfCheetah HalfCheetah HalfCheetah HalfCheetah)
# seed_id=(5 5 5 5)
# update_every_id=(1 2 5 40)

gpu_id=(6)
env_id=(HalfCheetah)
gin_id=(HalfCheetah)
seed_id=(5)
update_every_id=(1)

# gpu_id=(4 4 4)
# env_id=(Hopper Walker2d Ant)
# gin_id=(Hopper Walker2d Ant)
# seed_id=(0 0 0)


for ((i=0;i<${#gpu_id[@]};i++))
do
    CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_ppo.py \
                    --policy ${algo} \
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
                    --random_collect 4000 \
                    --tau 0.01 \
                    --update_every ${update_every_id[i]} \
                    --save_model \
                    --dir-root "./output_${algo}" \
                    --remark "tf-${env_id[i]}, FoSta, ppo, max_step=3000000" \
                    > ./my_log/exp_${algo}_${env_id[i]}_fourier_s${seed_id[i]}_up${update_every_id[i]}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=${gpu_id[i]} python -u eager_main_ofePaper_ppo.py \
    #                 --policy ${algo} \
    #                 --env ${env_id[i]}-v2 \
    #                 --gin ./gins_${algo}/${gin_id[i]}.gin \
    #                 --seed ${seed_id[i]} \
    #                 --save_model \
    #                 --dir-root "./output_${algo}" \
    #                 > ./my_log/exp_${algo}_${env_id[i]}_ofePaper_s${seed_id[i]}.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_ofePaper_ppo.py \
    #                 --policy ${algo} \
    #                 --env ${env_id[i]}-v2 \
    #                 --seed ${seed_id[i]} \
    #                 --save_model \
    #                 --dir-root "./output_${algo}" \
    #                 > ./my_log/exp_${algo}_${env_id[i]}_raw_s${seed_id[i]}.log 2>&1 &
    
done