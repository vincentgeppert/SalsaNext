#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

from tasks.semantic.modules.user import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')

if __name__ == '__main__':
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=os.path.expanduser("~") + '/logs/' +
                datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
        help='Directory to put the predictions. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--uncertainty', '-u',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Uncertainty Version'
    )

    parser.add_argument(
        '--monte-carlo', '-c',
        type=int, default=30,
        help='Number of samplings per scan'
    )

    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        default=None,
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )

    parser.add_argument(
        '--epistemic', '-e',
        type=str2bool,
        required=False,
        default=False,
        help='Set this if you dont want to use Uncertainty Version but still want to get epistemic uncertainty',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("Uncertainty", FLAGS.uncertainty)
    print("Monte Carlo Sampling", FLAGS.monte_carlo)
    print("infering", FLAGS.split)
    print("epistemic uncertainty", FLAGS.epistemic)
    print("----------\n")
    #print("Commit hash (training version): ", str(
    #    subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file from %s" % FLAGS.model)
        ARCH = yaml.safe_load(open(FLAGS.model + "/kitti360_arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        DATA = yaml.safe_load(open(FLAGS.model + "/kitti_odometry_data_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # create log folder
    try:
        if not os.path.exists(os.path.join(FLAGS.log, 'SalsaNext_semantics')):
            os.makedirs(os.path.join(FLAGS.log, 'SalsaNext_semantics'))
        #os.makedirs(os.path.join(FLAGS.log, "sequences"))
        if FLAGS.split == 'train':
            for seq in DATA["split"]["train"]:
                seq = '{0:04d}'.format(int(seq)) #KITTI odometry
                #seq = '2013_05_28_drive_%04d_sync' %seq #KITTI-360
                print("train", seq)
                os.makedirs(os.path.join(FLAGS.log, 'SalsaNext_semantics', seq))
                os.makedirs(os.path.join(FLAGS.log, 'SalsaNext_semantics', seq, "predictions"))
        if FLAGS.split == 'valid':
            for seq in DATA["split"]["valid"]:
                seq = '{0:04d}'.format(int(seq)) #KITTI odometry
                #seq = '2013_05_28_drive_%04d_sync' %seq #KITTI-360
                print("valid", seq)
                os.makedirs(os.path.join(FLAGS.log, 'SalsaNext_semantics', seq))
                os.makedirs(os.path.join(FLAGS.log, 'SalsaNext_semantics', seq, "predictions"))
        if FLAGS.split == 'test':
            for seq in DATA["split"]["test"]:
                seq = '{0:04d}'.format(int(seq)) #KITTI odometry
                #seq = '2013_05_28_drive_%04d_sync' %seq #KITTI-360
                print("test", seq)
                os.makedirs(os.path.join(FLAGS.log, 'SalsaNext_semantics', seq))
                os.makedirs(os.path.join(FLAGS.log, 'SalsaNext_semantics', seq, "predictions"))
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    # create user and infer dataset
    FLAGS.log = os.path.join(FLAGS.log, 'SalsaNext_semantics')
    user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model,FLAGS.split, FLAGS.uncertainty, FLAGS.epistemic, FLAGS.monte_carlo)
    user.infer()
