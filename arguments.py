import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG")
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--steps", default=1000000, type=int)
    parser.add_argument("--sac-units", default=256, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gin", default=None)
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--force", default=False, action="store_true",
                        help="remove existed directory")
    parser.add_argument("--dir-root", default="output", type=str)
    parser.add_argument("--img_save_dir", default='./results/img')

    # fourier
    # parser.add_argument("--block", default="densenet", type=str, choices=['densenet', 'mlp', 'mlp_cat'])
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--dim_discretize", default=128, type=int)
    parser.add_argument("--pred_stoch", default=False, action="store_true",
                        help="predict the state sequence under the stochastic policy")
    parser.add_argument("--fourier_type", default='dtft', type=str)
    # parser.add_argument("--original", default=False, action="store_true",
    #                     help="run original rl policy without ofe and fourier")

    # loss
    parser.add_argument("--use_projection", default=False, action="store_true")
    parser.add_argument("--projection_dim", default=256, type=int)
    parser.add_argument("--cosine_similarity", default=False, action="store_true")

    # auxiliary records
    parser.add_argument("--qval_img", default=False, action="store_true")
    parser.add_argument("--tsne", default=False, action="store_true")
    parser.add_argument("--record_grad", default=False, action="store_true")
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--record_state", default=False, action="store_true")
    # parser.add_argument("--visual_buffer", default=False, action="store_true")
    # parser.add_argument("--record_rb_ind", default=False, action="store_true")

    # intervals
    parser.add_argument("--summary_freq", default=1000, type=int) #
    parser.add_argument("--eval_freq", default=5000, type=int) #
    parser.add_argument("--value_eval_freq", default=20000, type=int) #
    parser.add_argument("--sne_freq", default=10000, type=int)
    # parser.add_argument("--record_state_freq", default=500, type=int)
    parser.add_argument("--grad_freq", default=10000, type=int)  #
    parser.add_argument("--random_collect", default=10000, type=int)
    parser.add_argument("--pre_train_step", default=10000, type=int)
    parser.add_argument("--save_freq", default=100000, type=int)
    # parser.add_argument("--visual_buffer_freq", default=5000, type=int)

    # target
    parser.add_argument("--target_update_freq", default=100, type=int)
    parser.add_argument("--tau", default=0.005, type=float)
    

    parser.add_argument("--remark", default="dtft, use projection")
    # "dtft, tune the hyperparameter dim_discretize from 128 to 64"
    # "dtft, after debug: pi -> 2*pi, remove FT network"

    # network_weight
    parser.add_argument("--weight", default=0.5, type=float)

    args = parser.parse_args()

    return args