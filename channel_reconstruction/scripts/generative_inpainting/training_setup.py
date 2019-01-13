from random import shuffle
from tqdm import tqdm
import argparse
import os


import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='../../data/generative_inpainting/', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='../../objects/generative_inpainting/train_shuffled.flist', type=str,
                    help='The output filename.')
parser.add_argument('--validation_filename', default='../../objects/generative_inpainting/validation_shuffled.flist', type=str,
                    help='The output filename.')
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to shuffle')

if __name__ == "__main__":

    args = parser.parse_args()
    dirs = args.folder_path
    training_file_names = []
    validation_file_names = []

    training_folder = os.listdir(dirs + "/training")
    for training_item in tqdm(training_folder):
        training_item = dirs + "/training" + "/" + training_item
        training_file_names.append(training_item)

    validation_folder = os.listdir(dirs + "/validation")
    for validation_item in tqdm(validation_folder):
        validation_item = dirs + "/validation" + "/" + validation_item
        validation_file_names.append(validation_item)


    # This would print all the files and directories

    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(training_file_names)
        shuffle(validation_file_names)

    # make output file if not existed
    # if not os.path.exists(args.train_filename):
    #    os.mknod(args.train_filename)

    #if not os.path.exists(args.validation_filename):
    #    os.mknod(args.validation_filename)
    #

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename, ", is_shuffle: ", args.is_shuffled)
