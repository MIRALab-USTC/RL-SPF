# bash tool_for_plot/run_img_PPO.sh

algo=PPO
suffix=_low15-high15-freq-loss
data_source=evaluation
dim=512
file_name=(record_sa_data predictor_visualize projection_visualize projection_layers_visualize IDTFT_predict_s2_concat video)

# gpu_id=(0 0 0 0 1)
# env_id=(HalfCheetah Hopper Swimmer Ant Walker2d)
# gin_id=(HalfCheetah Hopper Swimmer Ant Walker2d)
# seed_id=(0 0 0 0 0)
# update_every_id=(5 150 200 150 2)

# gpu_id=(1)
# env_id=(Humanoid)
# gin_id=(Humanoid)
# seed_id=(5)
# update_every_id=(1) Humanoid不太行，跑不了1000步

gpu_id=(0)
env_id=(Swimmer)
gin_id=(Swimmer)
seed_id=(0)
update_every_id=(200)

# > ./my_log/visualize_${algo}_${file_name[4]}.log 2>&1 &   nohup 
# suffix: "_Proj2_LowFreqLoss", "_low15-high15-freq-loss", "_low15-med15+20-high15-freq-loss", "_low20-med25-high20-freq-loss" "_low15-freq-loss"

for ((i=0;i<${#gpu_id[@]};i++))
do
    # CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u ${file_name[0]}.py \
    #                 --policy ${algo} \
    #                 --env ${env_id[i]}-v2 \
    #                 --gin ./gins/${gin_id[i]}.gin \
    #                 --seed ${seed_id[i]} \
    #                 --fourier_type dtft \
    #                 --dim_discretize 128 \
    #                 --use_projection \
    #                 --projection_dim ${dim} \
    #                 --pre_train_step 2 \
    #                 --target_update_freq 1000 \
    #                 --cosine_similarity \
    #                 --tau 0.01 \
    #                 --update_every ${update_every_id[i]} \
    #                 --dir-root "./output_${algo}_img" \
    #                 --suffix ${suffix} \
    #                 --data_source ${data_source} \
    #                 --remark "tf-${env_id[i]}, FoSta, visualization" \
    #                 > ./my_log/visualize_${algo}_${env_id[i]}_${file_name[0]}2.log 2>&1

    # CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u ${file_name[1]}.py \
    #                 --policy ${algo} \
    #                 --env ${env_id[i]}-v2 \
    #                 --gin ./gins/${gin_id[i]}.gin \
    #                 --seed ${seed_id[i]} \
    #                 --fourier_type dtft \
    #                 --dim_discretize 128 \
    #                 --use_projection \
    #                 --projection_dim ${dim} \
    #                 --pre_train_step 2 \
    #                 --target_update_freq 1000 \
    #                 --cosine_similarity \
    #                 --tau 0.01 \
    #                 --update_every ${update_every_id[i]} \
    #                 --dir-root "./output_${algo}_img" \
    #                 --suffix ${suffix} \
    #                 --data_source ${data_source} \
    #                 --remark "tf-${env_id[i]}, FoSta, visualization" \
    #                 > ./my_log/visualize_${algo}_${env_id[i]}_${file_name[1]}2.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u ${file_name[4]}.py \
    #                 --policy ${algo} \
    #                 --env ${env_id[i]}-v2 \
    #                 --gin ./gins/${gin_id[i]}.gin \
    #                 --seed ${seed_id[i]} \
    #                 --fourier_type dtft \
    #                 --dim_discretize 128 \
    #                 --use_projection \
    #                 --projection_dim ${dim} \
    #                 --pre_train_step 2 \
    #                 --target_update_freq 1000 \
    #                 --cosine_similarity \
    #                 --tau 0.01 \
    #                 --update_every ${update_every_id[i]} \
    #                 --dir-root "./output_${algo}_img" \
    #                 --suffix ${suffix} \
    #                 --data_source ${data_source} \
    #                 --remark "tf-${env_id[i]}, FoSta, visualization" \
    #                 > ./my_log/visualize_${algo}_${env_id[i]}_${file_name[4]}.log 2>&1 &
    
    # python concat_img.py --suffix ${suffix} --data_source ${data_source} --policy ${algo} --env ${env_id[i]}-v2 &

    CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u ${file_name[5]}.py \
                    --policy ${algo} \
                    --env ${env_id[i]}-v2 \
                    --gin ./gins/${gin_id[i]}.gin \
                    --seed ${seed_id[i]} \
                    --fourier_type dtft \
                    --dim_discretize 128 \
                    --use_projection \
                    --projection_dim ${dim} \
                    --pre_train_step 2 \
                    --target_update_freq 1000 \
                    --cosine_similarity \
                    --tau 0.01 \
                    --update_every ${update_every_id[i]} \
                    --dir-root "./output_${algo}_img" \
                    --suffix ${suffix} \
                    --data_source ${data_source} \
                    --remark "tf-${env_id[i]}, FoSta, visualization" \
                    > ./my_log/video_${algo}_${env_id[i]}_${file_name[5]}2.log 2>&1 &
                    
done