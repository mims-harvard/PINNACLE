import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Learning node embeddings.")

    # Input
    parser.add_argument("--G_f", type=str, default="../data/networks/global_ppi_edgelist.txt/", help="Directory to global reference PPI network")
    parser.add_argument("--ppi_dir", type=str, default="../data/networks/ppi_edgelists/", help="Directory to PPI layers")
    parser.add_argument("--mg_f", type=str, default="../data/networks/mg_edgelist.txt", help="Directory to metagraph")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train")
    parser.add_argument("--resume_run", type=str, default="", help="Model hyperparameters")
    
    # Parameters
    parser.add_argument("--loader", type=str, default="graphsaint", choices=["neighbor", "graphsaint"], help="Loader for minibatching.")

    # Hyperparameters
    parser.add_argument("--feat_mat", type=int, default=2048, help="Random Gaussian vectors of shape (1 x 2048)")
    parser.add_argument("--output", type=int, default=8, help="Output size")
    parser.add_argument("--hidden", type=int, default=16, help="Output size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--theta", type=float, default=0.1, help="Theta (for PPI loss)")
    parser.add_argument("--lmbda", type=float, default=0.01, help="Lambda (for center loss function)")
    parser.add_argument("--lr_cent", type=float, default=0.01, help="Learning rate for center loss")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--norm", type=str, default=None, help="Type of normalization layer to use in up-pooling")
    parser.add_argument("--pc_att_channels", type=int, default=8, help="Type of normalization layer to use in up-pooling")
    
    # Save
    parser.add_argument('--save_prefix', type=str, default='../data/pinnacle_embeds/pinnacle', help='Prefix of all saved files')
    parser.add_argument('--plot', type=bool, default=False, help='Boolean to fit and plot a UMAP')
    
    args = parser.parse_args()
    return args


def get_hparams(args):
    
    hparams = {
               'pc_att_channels': args.pc_att_channels,
               'feat_mat': args.feat_mat, 
               'output': args.output,
               'hidden': args.hidden,
               'lr': args.lr,
               'wd': args.wd,
               'dropout': args.dropout,
               'gradclip': 1.0,
               'n_heads': args.n_heads,
               'lambda': args.lmbda,
               'theta': args.theta,
               'lr_cent': args.lr_cent,
               'loss_type': "BCE",
               'plot': args.plot,
              }
    print("Hyperparameters:", hparams)    

    return hparams
