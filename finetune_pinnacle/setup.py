# Set up model environment, parameters, and data

import os
import pandas as pd
import argparse
import json
import torch

from read_data import read_labels_from_evidence


def create_parser():
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument("--hidden_dim_1", type=int, default=64, help="1st hidden dim size")
    parser.add_argument("--hidden_dim_2", type=int, default=32, help="2nd hidden dim size, discard if 0")
    parser.add_argument("--hidden_dim_3", type=int, default=0, help="3rd hidden dim size, discard if 0")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--norm", type=str, default=None, help="normalization layer")
    parser.add_argument("--actn", type=str, default="relu", help="activation type")
    parser.add_argument("--order", type=str, default="nd", help="order of normalization and dropout")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--num_epoch", type=int, default=1, help="epoch num")
    parser.add_argument("--batch_size", type=int, help="batch size")

    # Input data for finetuning task
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--data_split_path", type=str, default="targets/data_split")
    parser.add_argument('--positive_proteins_prefix', type=str, default="../data/therapeutic_target_task/positive_proteins")
    parser.add_argument('--negative_proteins_prefix', type=str, default="../data/therapeutic_target_task/negative_proteins")
    
    # Input PINNACLE representations
    parser.add_argument("--embeddings_dir", type=str)
    parser.add_argument("--embed", type=str, default="pinnacle")

    # Output directories
    parser.add_argument("--metrics_output_dir", type=str, default="./tmp_evaluation_results/")
    parser.add_argument("--models_output_dir", type=str, default="./tmp_model_outputs/")
    parser.add_argument("--random_state", type=int, default=1)
    parser.add_argument("--random", action="store_true", help="random runs without fixed seeds")
    parser.add_argument("--overwrite", action="store_true", help="whether to overwrite the label data or not")
    parser.add_argument("--train_size", type=float, default=0.6)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--weigh_sample", action="store_true", help="whether to weigh samples or not")  # default = False
    parser.add_argument("--weigh_loss", action="store_true", help="whether to weigh losses or not")  # default = False
    
    args = parser.parse_args()
    return args


def get_hparams(args):

    hparams = {
               "lr": args.lr, 
               "wd": args.wd, 
               "hidden_dim_1": args.hidden_dim_1, 
               "hidden_dim_2": args.hidden_dim_2, 
               "hidden_dim_3": args.hidden_dim_3, 
               "dropout": args.dropout, 
               "actn": args.actn, 
               "order": args.order, 
               "norm": args.norm,
               "task_name": args.task_name
              }
    return hparams


def setup_paths(args):
    random_state = args.random_state if args.random_state >= 0 else None
    if random_state == None:
        models_output_dir = args.models_output_dir + args.embed + "/"
        metrics_output_dir = args.metrics_output_dir + args.embed + "/"
    else:
        models_output_dir = args.models_output_dir + args.embed + ("_seed=%s" % str(random_state)) + "/"
        metrics_output_dir = args.metrics_output_dir + args.embed + ("_seed=%s" % str(random_state)) + "/"
    if not os.path.exists(models_output_dir): os.makedirs(models_output_dir)
    if not os.path.exists(metrics_output_dir): os.makedirs(metrics_output_dir)
    
    embed_path = args.embeddings_dir + args.embed + "_protein_embed.pth"
    labels_path = args.embeddings_dir + args.embed + "_labels_dict.txt"
    return models_output_dir, metrics_output_dir, random_state, embed_path, labels_path
