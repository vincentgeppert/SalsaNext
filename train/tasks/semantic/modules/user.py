#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np

from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.postproc.KNN import KNN

""" Function to enable the dropout layers during test-time """  
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir, split, uncertainty, only_epistemic, mc=30):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.uncertainty = uncertainty
    self.split = split
    self.mc = mc
    self.only_epistemic = only_epistemic

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=False,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
        torch.nn.Module.dump_patches = True
        if self.uncertainty:
            self.model = SalsaNextUncertainty(self.parser.get_n_classes())
            #self.model = nn.DataParallel(self.model)
            w_dict = torch.load(modeldir + "/SalsaNext_valid_best",
                                map_location=lambda storage, loc: storage)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
        else:
            self.model = SalsaNext(self.parser.get_n_classes())
            #self.model = nn.DataParallel(self.model)
            w_dict = torch.load(modeldir + "/SalsaNext_valid_best",
                                map_location=lambda storage, loc: storage)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
        
    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    print(torch.cuda.is_available())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    cnn = []
    knn = []
    if self.split == None:

        self.infer_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
        # do valid set
        self.infer_subset(loader=self.parser.get_valid_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
        # do test set
        self.infer_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)


    elif self.split == 'valid':
        self.infer_subset(loader=self.parser.get_valid_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    elif self.split == 'train':
        self.infer_subset(loader=self.parser.get_train_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    else:
        self.infer_subset(loader=self.parser.get_test_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")

    return

  def infer_subset(self, loader, to_orig_fn,cnn,knn):
    # switch to evaluate mode
    if not self.uncertainty:
      self.model.eval() #so that dropout for mc sampling still works if uncertainty version 
      if self.only_epistemic: #set dropout layers to train mode
        enable_dropout(self.model)

    total_time=0
    total_frames=0
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      counter = 0

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]
        if self.gpu:
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        #compute output
        if self.uncertainty:
            if counter == 0:
                print('-----')
                print('uncertainty version')
                print('-----')

            # output: outputs_mean, outputs_variance
            proj_output_r,log_var_r = self.model(proj_in)
            for i in range(self.mc):    #monte carlo sampling with dropout
                proj_output, log_var = self.model(proj_in) 
                log_var_r = torch.cat((log_var, log_var_r))
                proj_output_r = torch.cat((proj_output, proj_output_r))

            proj_output2, log_var2 = self.model(proj_in)
            proj_output = proj_output_r.var(dim=0, keepdim=True).mean(dim=1) # model uncertainty
            log_var2 = log_var_r.mean(dim=0, keepdim=True).mean(dim=1) # data uncertainty
            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_time = time.time() - end
            print("Infered seq", path_seq, "scan", path_name,
                  "in", frame_time, "sec")
            total_time += frame_time
            total_frames += 1
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            #pred_np = pred_np.reshape((-1)).astype(np.int32)
            pred_np = pred_np.reshape((-1)).astype(np.int16)

            # log_var2 = log_var2[0][p_y, p_x]
            # log_var2 = log_var2.cpu().numpy()
            # log_var2 = log_var2.reshape((-1)).astype(np.float32)

            log_var2 = log_var2[0][p_y, p_x]
            log_var2 = log_var2.cpu().numpy()
            log_var2 = log_var2.reshape((-1)).astype(np.float32)
            # assert proj_output.reshape((-1)).shape == log_var2.reshape((-1)).shape == pred_np.reshape((-1)).shape

            # map to original label
            pred_np = to_orig_fn(pred_np)

            # save scan
            path = os.path.join(self.logdir,
                                path_seq, "predictions", path_name)
            pred_np.tofile(path)

            path = os.path.join(self.logdir,
                                path_seq, "log_var", path_name)
            if not os.path.exists(os.path.join(self.logdir,
                                               path_seq, "log_var")):
                os.makedirs(os.path.join(self.logdir,
                                         path_seq, "log_var"))
            log_var2.tofile(path)

            proj_output = proj_output[0][p_y, p_x]
            proj_output = proj_output.cpu().numpy()
            proj_output = proj_output.reshape((-1)).astype(np.float32)

            path = os.path.join(self.logdir, path_seq, "uncert", path_name)
            if not os.path.exists(os.path.join(self.logdir, path_seq, "uncert")):
                os.makedirs(os.path.join(self.logdir, path_seq, "uncert"))
            proj_output.tofile(path)

            print(total_time / total_frames)

            counter += 1
        
        #######
        elif self.only_epistemic:
            if counter == 0:
                print('-----')
                print('own uncertainty version')
                print('-----')

            proj_output_r = self.model(proj_in)
            for i in range(self.mc):    #monte carlo sampling with dropout
                proj_output = self.model(proj_in)
                proj_output_r = torch.cat((proj_output, proj_output_r))

            #ensemble voting
            proj_output_mc = []
            for mc_sampling in proj_output_r[:][:][:][:]:
                proj_output_mc.append(mc_sampling.argmax(dim=0))
            proj_output_mc = torch.stack(proj_output_mc)
            proj_argmax, _ = torch.mode(proj_output_mc, dim=0)

            #proj_output = proj_output_r.var(dim=0, keepdim=True).mean(dim=1) # model uncertainty -> first variance and then mean over all classes per pixel
            var_per_class = proj_output_r.var(dim=0, keepdim=True) #variance over mc sampling -> torch.Size([1, 20, 64, 2048])

            #get uncertainty value for the predicted class
            uncer_values = torch.ones(1, proj_argmax.shape[0], proj_argmax.shape[1])
            for i in range(proj_argmax.shape[0]):
                for j in range(proj_argmax.shape[1]):
                    sem_class = proj_argmax[i][j]
                    uncer_values[0][i][j] = var_per_class[0][sem_class][i][j] #variance for predicted class per pixel -> torch.Size([64, 2048])


            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_time = time.time() - end
            print("Infered seq", path_seq, "scan", path_name,
                  "in", frame_time, "sec")
            total_time += frame_time
            total_frames += 1
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            #pred_np = pred_np.reshape((-1)).astype(np.int32)
            pred_np = pred_np.reshape((-1)).astype(np.int16)

            # map to original label
            pred_np = to_orig_fn(pred_np)

            # save scan
            path = os.path.join(self.logdir,
                                path_seq, "predictions", path_name)
            pred_np.tofile(path)

            uncer_values = uncer_values[0][p_y, p_x]
            uncer_values = uncer_values.cpu().numpy()
            #proj_output = proj_output.reshape((-1)).astype(np.float32)
            uncer_values = uncer_values.reshape((-1)).astype(np.float32)

            #round uncer values
            rounded_uncer = []
            for value in uncer_values:
                rnd=round(value,7)
                rounded_uncer.append(rnd)
            uncer_values_rounded = np.asarray(rounded_uncer, dtype=np.float32)

            path = os.path.join(self.logdir, path_seq, "uncert", path_name)
            if not os.path.exists(os.path.join(self.logdir, path_seq, "uncert")):
                os.makedirs(os.path.join(self.logdir, path_seq, "uncert"))
            uncer_values_rounded.tofile(path)

            print(total_time / total_frames)

            counter += 1
        #######

        else:
            if counter == 0:
                print('-----')
                print('no uncertainty version')
                print('-----')

            proj_output = self.model(proj_in)
            proj_argmax = proj_output[0].argmax(dim=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("Network seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            end = time.time()
            cnn.append(res)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("Network seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            end = time.time()
            cnn.append(res)

            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("KNN Infered seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            knn.append(res)
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            #pred_np = pred_np.reshape((-1)).astype(np.int32)
            pred_np = pred_np.reshape((-1)).astype(np.int16)

            # map to original label
            pred_np = to_orig_fn(pred_np)

            # save scan
            path = os.path.join(self.logdir, path_seq, "predictions", path_name)
            pred_np.tofile(path)

            counter += 1
