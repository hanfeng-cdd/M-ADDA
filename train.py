from torch import optim

import models
import pandas as pd
import numpy as np
import datasets
import misc as ms
import fit_src as ts
import experiments
import test
import fit_tgt as tg
import torch
import os
import test

def train(exp_dict):
  history = ms.load_history(exp_dict)

  # Source
  src_trainloader, src_valloader= ms.load_src_loaders(exp_dict)
  #####################
  ## Train source model
  #####################
  src_model, src_opt = ms.load_model_src(exp_dict)

  # Train Source
  for e in range(history["src_train"][-1]["epoch"], 
                 exp_dict["src_epochs"]):
    train_dict = ts.fit_src(src_model, src_trainloader, src_opt)

    loss = train_dict["loss"]
    print("Source ({}) - Epoch [{}/{}] - loss={:.2f}".format(
                type(src_trainloader).__name__, e, 
                exp_dict["src_epochs"], loss))

    history["src_train"] += [{"loss":loss, "epoch":e}]

    if e % 50 == 0:
      ms.save_model_src(exp_dict, history, src_model, src_opt)
      

  # Test Source
  src_acc = test.validate(src_model, 
                          src_model, 
                          src_trainloader, 
                          src_valloader)

  print("{} TEST Accuracy = {:2%}\n".format(exp_dict["src_dataset"], 
                                            src_acc))
  history["src_acc"] = src_acc

  ms.save_model_src(exp_dict, history, src_model, src_opt)

  #####################
  ## Train Target model
  #####################
  tgt_trainloader, tgt_valloader= ms.load_tgt_loaders(exp_dict)

  # load models
  tgt_model, tgt_opt, disc_model, disc_opt = ms.load_model_tgt(exp_dict)
  tgt_model.load_state_dict(src_model.state_dict())


  for e in range(history["tgt_train"][-1]["epoch"], exp_dict["tgt_epochs"]+1):
      # 1. Train disc
      if exp_dict["options"]["disc"] == True:
        tg.fit_disc(src_model, tgt_model, disc_model,
                      src_trainloader, tgt_trainloader, 
                      opt_tgt=tgt_opt,
                      opt_disc=disc_opt, 
                      epochs=3, verbose=0)

      acc_tgt = test.validate(src_model, tgt_model, 
                              src_trainloader, 
                              tgt_valloader)

      history["tgt_train"] += [{"epoch":e,
                       "acc_src":src_acc, 
                       "acc_tgt":acc_tgt,
                       "n_train - "+ exp_dict["src_dataset"]:len(src_trainloader.dataset), 
                       "n_train - "+ exp_dict["tgt_dataset"]:len(tgt_trainloader.dataset), 
                       "n_test - " + exp_dict["tgt_dataset"]:len(tgt_valloader.dataset)}] 

      print("\n>>> Methods: {} - Source: {} -> Target: {}".format(None, 
                                                                 exp_dict["src_dataset"], 
                                                                 exp_dict["tgt_dataset"]))
      print(pd.DataFrame([history["tgt_train"][-1]]))

      if (e % 5)==0:
        ms.save_model_tgt(exp_dict, history, tgt_model, tgt_opt,
                          disc_model, disc_opt)
        #ms.test_latest_model(exp_dict)

      # 2. Train center-magnet
      if exp_dict["options"]["center"] == True:
        tg.fit_center(src_model, tgt_model, 
                        src_trainloader, tgt_trainloader,
                        tgt_opt, epochs=1)

  ms.save_model_tgt(exp_dict, history, tgt_model, tgt_opt,
                    disc_model, disc_opt)

  exp_dict["reset_src"] = 0
  exp_dict["reset_tgt"] = 0
  ms.test_latest_model(exp_dict)