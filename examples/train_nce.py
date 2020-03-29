import torch
import argparse
import os
import pickle
import datetime
from torch import nn
import torch.nn.functional as F
from binding_prediction.models import BindingModel
from binding_prediction.dataset import DrugProteinDataset, collate_fn
from binding_prediction import pretrained_language_models
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def _parse_args():
    parser = argparse.ArgumentParser(description='Train on a binding database.')
    parser.add_argument('--num-epoch', '-e', type=int, default=256,
                        help='Number of epochs to train (default: 255)')
    parser.add_argument('--dir', '-d', type=str, default='.', help='Directory to which to save the model.')
    parser.add_argument('--batch-size', '-b', type=int, default=20,
                        help='Mini-batch size (default: 25)')
    parser.add_argument('--train_dataset', '-t', type=str, default='data/molecules_train_qm9.json',
                        help='Training dataset')
    parser.add_argument('--valid_dataset', '-v', type=str, default='data/molecules_valid_qm9.json',
                        help='Validation dataset')
    parser.add_argument('--merge_molecule_channels', '-m', type=int, default=10,
                        help='Number of channels to use in the hidden layers')
    parser.add_argument('--merge_prot_channels', '-p', type=int, default=10,
                        help='Number of channels to use in the hidden layers')
    parser.add_argument('--hidden_channels', '-c', nargs='*', type=int, default=[10],
                        help='Number of channels to use in the hidden layers')
    parser.add_argument('--conv_kernel_sizes', '-k', nargs='*', type=int, default=None,
                        help='Number of channels to use in the hidden layers')
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--lmarch', '-a', type=str, default='elmo',
                        help='Language Model Architecture')
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Use CUDA (default)')
    parser.add_argument('--no-cuda', '--cpu', dest='cuda', action='store_false',
                        help='Use CPU')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()
    return args


def calculate_loss(output, batch, device='cuda'):
    targets = batch['is_true'].to(device=device).int().unsqueeze(-1)
    bce_loss = F.binary_cross_entropy_with_logits(output, targets)
    return bce_loss


def run_model_on_batch(model, batch, device='cuda'):
    adj_mat = batch['adj_mat'].to(device=device)
    features = batch['node_features'].to(device=device)
    sequences = batch['protein']
    out_features = model(adj_mat, features, sequences)
    return out_features


def initialize_logging(root_dir='./', logging_path=None):
    if logging_path is None:
        basename = "logdir"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        logging_path = "_".join([basename, suffix])
    full_path = root_dir + logging_path
    writer = SummaryWriter(full_path)
    return writer


def main():
    args = _parse_args()

    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    writer = initialize_logging(args.dir, '/logs/')

    # Save the construction arguments for future reference.
    with open(args.dir + '/training_args.pkl', 'wb') as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

    lm, path = pretrained_language_models[args.lmarch]

    #### NEEDS CHANGING WITH DATASET? ####
    train_dataset = DrugProteinDataset(args.train_dataset, prob_fake=0.5)
    valid_dataset = DrugProteinDataset(args.valid_dataset, prob_fake=0.5)
    ######################################

    cfxn = lambda x : collate_fn(x, prots_are_sequences=True)

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=cfxn)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=True, collate_fn=cfxn)

    in_channels_nodes = train_dataset[0]['node_features'].shape[-1]
    in_channels_seq = 512
    in_channels = in_channels_seq + in_channels_nodes
    out_channels = 1
    model = BindingModel(in_channels_nodes, in_channels_seq, args.merge_molecule_channels, 
                         args.merge_prot_channels, args.hidden_channels, out_channels)
    model = model.to(device=device)
    model.load_language_model(lm, path)

    writer.add_text("Log", "Initialized Model.")

    if os.path.isfile(args.dir + '/model_best.pt'):
        writer.add_text("Log", "Previous Model found.  Attempting to load previous best model...")
        model_param_dict = torch.load('models/model_best.pt')
        model.load_state_dict(model_param_dict)
        writer.add_text("Log", "Succesfully loaded previous model")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fxn = nn.BCEWithLogitsLoss()
    best_valid_loss = 1E8

    for n in range(args.num_epoch):
        model.train()
        total_train_loss = 0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = run_model_on_batch(model, batch, device=device).squeeze(-1)
            targets = batch['is_true'].to(device=device).float()
            loss = loss_fxn(targets, output)
            # loss = calculate_loss(output, batch)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Batch {}/{}.  Batch loss: {}".format(i, len(train_dataloader), loss))

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader):
                output = run_model_on_batch(model, batch, device=device).squeeze(-1)
                targets = batch['is_true'].to(device=device).float()
                loss = loss_fxn(targets, output)
                total_valid_loss += loss

        avg_train_loss = total_train_loss / len(train_dataset)
        avg_valid_loss = total_valid_loss / len(valid_dataset)
        print("Epoch {} Complete. Train loss: {}.  Valid loss: {}.".format(n, avg_train_loss, avg_valid_loss))
        writer.add_scalar('training_loss', avg_train_loss)
        writer.add_scalar('validation_loss', avg_valid_loss)

        torch.save(model.state_dict(), args.dir + '/model_current.pt')
        if avg_valid_loss < best_valid_loss:
            writer.add_text("Log", "Best validation loss achieved at %d." % n)
            torch.save(model.state_dict(), args.dir + '/model_best.pt')
            best_valid_loss = avg_valid_loss


if __name__ == "__main__":
    main()
