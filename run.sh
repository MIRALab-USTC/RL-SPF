# CUDA_VISIBLE_DEVICES=7 nohup python -u eager_main.py \
#                     --policy SAC \
#                     --env Hopper-v2 \
#                     --gin ./gins/Hopper.gin \
#                     --seed 0 \
#                     --original \
#                     --remark 'hopper origin sac' \
#                     > ./my_log/exp_hopper_raw.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 nohup python -u eager_main_addReward.py \
#                     --policy SAC \
#                     --env Walker2d-v2 \
#                     --gin ./gins/Walker2d.gin \
#                     --seed 0 \
#                     --fourier_type dtft \
#                     --only_reward \
#                     --record_grad \
#                     --remark 'dtft, only reward' \
#                     > ./my_log/exp_r_walker.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python -u eager_main.py \
#                     --policy SAC \
#                     --env HalfCheetah-v2 \
#                     --gin ./gins/HalfCheetah.gin \
#                     --seed 0 \
#                     --fourier_type dtft \
#                     --use_projection \
#                     --projection_dim 256 \
#                     --cosine_similarity \
#                     --record_grad \
#                     --remark 'dtft, tau=0.005' \
#                     > ./my_log/exp0.log 2>&1 &


# CUDA_VISIBLE_DEVICES=6 nohup python -u eager_main.py \
#                     --policy SAC \
#                     --env Hopper-v2 \
#                     --gin ./gins/Hopper.gin \
#                     --seed 0 \
#                     --fourier_type dtft \
#                     --use_projection \
#                     --projection_dim 256 \
#                     --cosine_similarity \
#                     --record_grad \
#                     --tau 0.01 \
#                     --target_update_interval 100 \
#                     --remark 'dtft, tau=0.01, update_target_freq=100' \
#                     > ./my_log/exp11.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python -u eager_main.py \
#                     --policy SAC \
#                     --env Hopper-v2 \
#                     --gin ./gins/Hopper.gin \
#                     --seed 2 \
#                     --fourier_type dtft \
#                     --use_projection \
#                     --projection_dim 256 \
#                     --cosine_similarity \
#                     --record_grad \
#                     --remark 'dtft, tau=0.005, update_target_freq=100' \
#                     > ./my_log/exp_hopper.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python -u eager_main.py \
#                     --policy SAC \
#                     --env Walker2d-v2 \
#                     --gin ./gins/Walker2d.gin \
#                     --seed 2 \
#                     --fourier_type dtft \
#                     --use_projection \
#                     --projection_dim 256 \
#                     --cosine_similarity \
#                     --record_grad \
#                     --remark 'dtft, tau=0.005, update_target_freq=100' \
#                     > ./my_log/exp_walker.log 2>&1 &


# CUDA_VISIBLE_DEVICES=7 nohup python -u eager_main.py \
#                     --policy SAC \
#                     --env Walker2d-v2 \
#                     --gin ./gins/Walker2d.gin \
#                     --seed 2 \
#                     --fourier_type dtft \
#                     --use_projection \
#                     --projection_dim 256 \
#                     --cosine_similarity \
#                     --record_grad \
#                     --tau 0.01 \
#                     --target_update_interval 1000 \
#                     --remark 'dtft, tau=0.01, update_target_freq=1000' \
#                     > ./my_log/exp_walker.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4 python -u eager_main.py \
#                     --policy SAC \
#                     --env HopperP-v2 \
#                     --gin ./gins/Hopper.gin \
#                     --seed 0 \
#                     --fourier_type dtft \
#                     --dim_discretize 128 \
#                     --use_projection \
#                     --projection_dim 256 \
#                     --cosine_similarity \
#                     --record_grad \
#                     --tau 0.01 \
#                     --target_update_freq 1000 \
#                     --remark 'hopper fosta' \
#                     > ./my_log/exp_human.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 python -u eager_main.py \
#                     --policy SAC \
#                     --env Swimmer-v2 \
#                     --gin ./gins/Swimmer.gin \
#                     --seed 0 \
#                     --fourier_type dtft \
#                     --use_projection \
#                     --projection_dim 256 \
#                     --cosine_similarity \
#                     --record_grad \
#                     --tau 0.01 \
#                     --target_update_interval 1000 \
#                     --remark 'swim fosta' \
#                     > ./my_log/exp_swim.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 python -u eager_main.py \
#                     --policy SAC \
#                     --env Swimmer-v2 \
#                     --gin ./gins/Swimmer.gin \
#                     --seed 0 \
#                     --original \
#                     --remark 'raw swim' \
#                     > ./my_log/exp_swim_origin.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 python -u eager_main.py \
#                     --policy SAC \
#                     --env Humanoid-v2 \
#                     --gin ./gins/Humanoid.gin \
#                     --seed 0 \
#                     --original \
#                     --remark 'raw human' \
#                     > ./my_log/exp_human_origin.log 2>&1 &   _complexLoss

# CUDA_VISIBLE_DEVICES=5 nohup python -u eager_main.py \
#                     --policy SAC \
#                     --env HalfCheetah-v2 \
#                     --gin ./gins/HalfCheetah.gin \
#                     --seed 2 \
#                     --fourier_type dtft \
#                     --use_projection \
#                     --projection_dim 256 \
#                     --cosine_similarity \
#                     --save_model \
#                     --tau 0.01 \
#                     --target_update_freq 1000 \
#                     --remark 'dtft, tau=0.01, update_target_freq=1000' \
#                     > ./my_log/exp_hc.log 2>&1 &


# CUDA_VISIBLE_DEVICES=6 nohup python -u eager_main.py \
#                     --policy SAC \
#                     --env Walker2d-v2 \
#                     --gin ./gins/Walker2d.gin \
#                     --seed 0 \
#                     --fourier_type dtft \
#                     --use_projection \
#                     --projection_dim 256 \
#                     --cosine_similarity \
#                     --record_grad \
#                     --remark 'dtft, tau=0.005' \
#                     > ./my_log/exp2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python -u eager_main.py \
#                     --policy SAC \
#                     --env Humanoid_modify2-v2 \
#                     --gin ./gins/Humanoid.gin \
#                     --seed 0 \
#                     --fourier_type dtft \
#                     --use_projection \
#                     --projection_dim 256 \
#                     --cosine_similarity \
#                     --record_grad \
#                     --remark 'dtft, target3, hidden_layer_size=output_dim*4' \
#                     > ./my_log/exp0.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 python -u test_model.py \
#                     --policy SAC \
#                     --env HalfCheetah-v2 \
#                     --gin ./gins_test_model/HalfCheetah.gin \
#                     --seed 0 \
#                     --fourier_type dtft \
#                     --dim_discretize 128 \
#                     --use_projection \
#                     --projection_dim 256 \
#                     --cosine_similarity \
#                     --record_grad \
#                     --tau 0.01 \
#                     --target_update_freq 1000 \
#                     --remark 'hc transfer, masses & frictions' \
#                     > ./my_log/exp_hc_transfer.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 python -u test_model_record_state.py \
#                     --policy SAC \
#                     --env HalfCheetah-v2 \
#                     --gin ./gins_ofePaper/HalfCheetah.gin \
#                     --seed 0 \
#                     --fourier_type dtft \
#                     --dim_discretize 128 \
#                     --use_projection \
#                     --projection_dim 256 \
#                     --cosine_similarity \
#                     --record_grad \
#                     --tau 0.01 \
#                     --target_update_freq 1000 \
#                     --remark 'hc transfer, masses & frictions' \
#                     > ./my_log/exp_hc_period.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python -u eager_main.py \
                    --policy SAC \
                    --env HalfCheetah-v2 \
                    --gin ./gins/HalfCheetah.gin \
                    --seed 0 \
                    --fourier_type dtft \
                    --dim_discretize 128 \
                    --use_projection \
                    --projection_dim 512 \
                    --cosine_similarity \
                    --tau 0.01 \
                    --target_update_freq 1000 \
                    --td3_linear_range 1000 \
                    --pre_train_step 1 \
                    --remark 'hc test fourier, test fourier' \
                    --dir-root "./output_SAC" \
                    > ./my_log/exp_fourier_test.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 python concat_img.py \
#                     > ./my_log/exp_concat_img.log 2>&1 &

