algo=TD3

gpu_id=(4 5 4 4 5)
env_id=(HalfCheetah Hopper Walker2d Swimmer Ant)
gin_id=(HalfCheetah Hopper Walker2d Swimmer Ant)
seed_id=(1 2 10 11 12)

# gpu_id=(7 6 5 3 1)
# env_id=(HalfCheetah Hopper Walker2d Swimmer Ant)
# gin_id=(HalfCheetah Hopper Walker2d Swimmer Ant)
# seed_id=(0 0 0 0 0)


# gpu_id=(4)
# env_id=(Swimmer)
# gin_id=(Swimmer)
# seed_id=(0)

# gpu_id=(3)
# env_id=(HalfCheetah)
# gin_id=(HalfCheetah)
# seed_id=(0)

# gpu_id=(4 4 4)
# env_id=(Hopper Walker2d Ant)
# gin_id=(Hopper Walker2d Ant)
# seed_id=(0 0 0)

for ((j=0;j<${#seed_id[@]};j++))
do
    for ((i=0;i<$[${#gpu_id[@]}-1];i++))
    do
        CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main.py \
                        --policy ${algo} \
                        --env ${env_id[i]}-v2 \
                        --gin ./gins_${algo}/${gin_id[i]}.gin \
                        --seed ${seed_id[j]} \
                        --fourier_type dtft \
                        --dim_discretize 128 \
                        --use_projection \
                        --projection_dim 512 \
                        --pre_train_step 10000 \
                        --target_update_freq 1000 \
                        --cosine_similarity \
                        --tau 0.01 \
                        --save_model \
                        --dir-root "./output_${algo}" \
                        --remark "tf-${env_id[i]}, FoSta, add low and high freq loss, without update_every and linear noise, weight_init_orthogonal, (400,300) TD3, tau=0.01" \
                        > ./my_log/exp_${algo}_${env_id[i]}_fourier_s${seed_id[j]}.log 2>&1 &

        # CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_ofePaper.py \
        #                 --policy ${algo} \
        #                 --env ${env_id[i]}-v2 \
        #                 --gin ./gins_${algo}/${gin_id[i]}.gin \
        #                 --seed ${seed_id[j]} \
        #                 --save_model \
        #                 --dir-root "./output_${algo}" \
        #                 > ./my_log/exp_${algo}_${env_id[i]}_ofePaper_s${seed_id[j]}.log 2>&1 &

        # CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_ofePaper.py \
        #                 --policy ${algo} \
        #                 --env ${env_id[i]}-v2 \
        #                 --seed ${seed_id[i]} \
        #                 --save_model \
        #                 --dir-root "./output_${algo}" \
        #                 > ./my_log/exp_${algo}_${env_id[i]}_raw_s${seed_id[j]}.log 2>&1 &
        
    done
    h=$[${#seed_id[@]}-1]
    CUDA_VISIBLE_DEVICES=${gpu_id[h]} nohup python -u eager_main.py \
                        --policy ${algo} \
                        --env ${env_id[h]}-v2 \
                        --gin ./gins_${algo}/${gin_id[h]}.gin \
                        --seed ${seed_id[j]} \
                        --fourier_type dtft \
                        --dim_discretize 128 \
                        --use_projection \
                        --projection_dim 512 \
                        --pre_train_step 10000 \
                        --target_update_freq 1000 \
                        --cosine_similarity \
                        --tau 0.01 \
                        --save_model \
                        --dir-root "./output_${algo}" \
                        --remark "tf-${env_id[h]}, FoSta, add low and high freq loss, without update_every and linear noise, weight_init_orthogonal, (400,300) TD3, tau=0.01" \
                        > ./my_log/exp_${algo}_${env_id[h]}_fourier_s${seed_id[j]}.log 2>&1

done