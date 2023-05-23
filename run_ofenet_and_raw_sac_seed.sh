algo=SAC

gpu_id=(7 7 6 6 5 5)
env_id=(HalfCheetah Hopper Walker2d Swimmer Ant Humanoid)
gin_id=(HalfCheetah Hopper Walker2d Swimmer Ant Humanoid)
seed_id=(0 0 0 0 0 0)

for ((i=0;i<${#gpu_id[@]};i++))
do
    CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_ofePaper.py \
                    --policy ${algo} \
                    --env ${env_id[i]}-v2 \
                    --gin ./gins/${gin_id[i]}.gin \
                    --seed ${seed_id[i]} \
                    --save_model \
                    --dir-root "./output_${algo}" \
                    > ./my_log/exp_${algo}_${env_id[i]}_ofePaper_s${seed_id[i]}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_ofePaper.py \
                    --policy ${algo} \
                    --env ${env_id[i]}-v2 \
                    --seed ${seed_id[i]} \
                    --save_model \
                    --dir-root "./output_${algo}" \
                    > ./my_log/exp_${algo}_${env_id[i]}_raw_s${seed_id[i]}.log 2>&1 &
done