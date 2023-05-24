import os

import PIL.Image as Image
import re
import sys
from arguments import parse_args
from src.util.misc import parse_cmdline_kwargs

dir_of_env = {'HalfCheetah-v2': 'hc', 'HalfCheetah_noise-v2': 'hcNoise', 'HalfCheetah_transfer-v2': 'hcTrans', 
            'Hopper-v2': 'hopper',  'HopperP-v2': 'hopperP', 'HopperV-v2': 'hopperV', 'Hopper_noise-v2': 'hopperNoise', 
            'Walker2d-v2': 'walker', 'Walker2dP-v2': 'walkerP',  'Walker2dV-v2': 'walkerV', 'Walker2d_noise-v2': 'walkerNoise', 
            'Ant-v2': 'ant', 'AntP-v2': 'antP', 'AntV-v2': 'antV', 'Ant_noise-v2': 'antNoise', 
            'Swimmer-v2': 'swim', 
            'Humanoid-v2': 'human'}


def resize_by_width(infile, image_size):
    im = Image.open(infile)
    (x, y) = im.size
    lv = round(x / image_size, 2) + 0.01
    x_s = int(x // lv)
    y_s = int(y // lv)
    print("x_s", x_s, y_s)
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    return out


def get_new_img_xy(infile, image_size):
    im = Image.open(infile)
    (x, y) = im.size
    lv = round(x / image_size, 2) + 0.01
    x_s = x // lv
    y_s = y // lv
    # print("x_s", x_s, y_s)
    # out = im.resize((x_s, y_s), Image.ANTIALIAS)
    return x_s, y_s


def image_compose(image_colnum, image_size, image_rownum, image_names, image_save_path, x_new, y_new):
    to_image = Image.new('RGB', (image_colnum * x_new, image_rownum * y_new), "white")
    total_num = 0
    for y in range(1, image_rownum + 1):
        for x in range(1, image_colnum + 1):
            from_image = resize_by_width(image_names[image_colnum * (y - 1) + x - 1], image_size)
            # from_image = Image.open(image_names[image_colnum * (y - 1) + x - 1]).resize((image_size,image_size ), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * x_new, (y - 1) * y_new))
            total_num += 1
            if total_num == len(image_names):
                break
    return to_image.save(image_save_path)


def get_image_list_fullpath(dir_path):
    file_name_list = os.listdir(dir_path)
    words = re.findall(r'\D+', file_name_list[0])

    if len(words) > 1:
        start = re.findall(r'\D+', file_name_list[0])[0]
        L = len(words)
        end = re.findall(r'\D+', file_name_list[0])[L - 1]
    else:
        start = re.findall(r'\D+', file_name_list[0])[0]
        end = ''
        
    remove_list = []

    for i in range(len(file_name_list)):
        if file_name_list[i].endswith('.csv'):
            remove_list.append(i)
        elif len(re.findall(r'\d+', file_name_list[i])) == 0:
            remove_list.append(i)
        else:
            number = re.findall(r'\d+', file_name_list[i])[0]
            print(number + '\n')
            file_name_list[i] = number + '.png'
    for i in remove_list:
        file_name_list.pop(i)

    file_name_list.sort(key= lambda x:int(x[:-4]))

    for i in range(len(file_name_list)):
        file_name_list[i] = start + file_name_list[i].strip('.png') + end

    image_fullpath_list = []
    for file_name_one in file_name_list:
        file_one_path = os.path.join(dir_path, file_name_one)
        if os.path.isfile(file_one_path):
            image_fullpath_list.append(file_one_path)
        else:
            img_path_list = get_image_list_fullpath(file_one_path)
            image_fullpath_list.extend(img_path_list)

    return image_fullpath_list


def merge_images(image_dir_path,image_size,image_colnum, env, dir_name, env_name, policy_name, up_image_dir_path):
    image_fullpath_list = get_image_list_fullpath(image_dir_path)
    if env_name == "Ant-v2":
        image_fullpath_list = image_fullpath_list[15:21]
    elif env_name == "Humanoid-v2":
        image_fullpath_list = image_fullpath_list[200:206]
    print("image_fullpath_list", len(image_fullpath_list), image_fullpath_list)

    image_save_path = os.path.join(up_image_dir_path, policy_name + '-SPF-' + dir_of_env[env_name] + '-' +  dir_name + '.pdf')
    # image_rownum = 4 
    image_rownum_yu = len(image_fullpath_list) % image_colnum
    if image_rownum_yu == 0:
        image_rownum = len(image_fullpath_list) // image_colnum
    else:
        image_rownum = len(image_fullpath_list) // image_colnum + 1

    x_list = []
    y_list = []
    
    for img_file in image_fullpath_list:
        img_x, img_y = get_new_img_xy(img_file, image_size)
        x_list.append(img_x)
        y_list.append(img_y)
    print("x_list", sorted(x_list))
    print("y_list", sorted(y_list))
    x_new = int(x_list[len(x_list) // 5 * 4])
    y_new = int(y_list[len(y_list) // 5 * 4])
    print(" x_new, y_new", x_new, y_new)
    image_compose(image_colnum, image_size, image_rownum, image_fullpath_list, image_save_path, x_new, y_new)
    # for img_file in image_fullpath_list:
    #     resize_by_width(img_file,image_size)

if __name__ == '__main__':

    # env_names = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Ant-v2', 'Swimmer-v2', 'Humanoid-v2']

    # dir_names = ['true and predicted_next_state', 'true and predicted_fourier_module']
    dir_names = ['true and predicted_next_state']

    parse = parse_args()
    args, extra_args = parse.parse_known_args(sys.argv)
    extra_args = parse_cmdline_kwargs(extra_args) # dict

    if extra_args['data_source'] == 'replay_buffer':
        suffix = extra_args['suffix'] + '_rb'             
    elif extra_args['data_source'] == 'evaluation':
        suffix = extra_args['suffix'] + '_eval'
    else:
        raise ValueError("invalid data source {}".format(extra_args['data_source']))

    env_names = [args.env]
    for env_name in env_names:
        for dir_name in dir_names:
            image_dir_path = os.path.join('./test/img_ablation/periodicity', args.policy+'-FoSta-'+dir_of_env[env_name] + suffix, dir_name)
            up_image_dir_path = os.path.join('./test/img_ablation/periodicity')
            image_size = 720 
            image_colnum = 3 
            merge_images(image_dir_path, image_size, image_colnum, dir_of_env[env_name], dir_name, env_name, args.policy, up_image_dir_path)
    
