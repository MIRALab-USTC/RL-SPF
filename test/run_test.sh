algo=SAC

file_name=(record_sa_data predictor_visualize projection_visualize projection_layers_visualize)
gpu_id=(7)
env_id=(HalfCheetah)
gin_id=(HalfCheetah)
seed_id=(0)

for ((i=0;i<${#gpu_id[@]};i++))
do
    CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u ${file_name[1]}.py \
                    --policy ${algo} \
                    --env ${env_id[i]}-v2 \
                    --gin ./gins/${gin_id[i]}.gin \
                    --seed ${seed_id[i]} \
                    --fourier_type dtft \
                    --dim_discretize 128 \
                    --use_projection \
                    --projection_dim 512 \
                    --pre_train_step 2 \
                    --target_update_freq 1000 \
                    --cosine_similarity \
                    --tau 0.01 \
                    --dir-root "./output_img" \
                    --remark "tf-${env_id[i]}, FoSta, visualization" \
                    > ./my_log/exp_${algo}_${file_name[1]}_visualize.log 2>&1 &

done