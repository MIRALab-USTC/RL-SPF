algo=PPO

gpu_id=(4 4 3 3 2 2)
env_id=(HalfCheetah Hopper Walker2d Swimmer Ant Humanoid)
gin_id=(HalfCheetah Hopper Walker2d Swimmer Ant Humanoid)
seed_id=(0 0 0 0 0 0)
update_every_id=(5 150 2 200 150 1)

for ((i=0;i<${#gpu_id[@]};i++))
do
    CUDA_VISIBLE_DEVICES=${gpu_id[i]} python -u eager_main_ofePaper_ppo.py \
                    --policy ${algo} \
                    --env ${env_id[i]}-v2 \
                    --gin ./gins/${gin_id[i]}.gin \
                    --seed ${seed_id[i]} \
                    --normalizer "layer" \
                    --save_model \
                    --update_every ${update_every_id[i]} \
                    --dir-root "./output_${algo}" \
                    > ./my_log/exp_${algo}_${env_id[i]}_ofePaper_s${seed_id[i]}_up${update_every_id[i]}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=${gpu_id[i]} nohup python -u eager_main_ofePaper_ppo.py \
                    --policy ${algo} \
                    --env ${env_id[i]}-v2 \
                    --seed ${seed_id[i]} \
                    --save_model \
                    --dir-root "./output_${algo}" \
                    > ./my_log/exp_${algo}_${env_id[i]}_raw_s${seed_id[i]}.log 2>&1 &
done