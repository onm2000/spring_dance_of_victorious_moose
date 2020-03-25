import torch
import argparse
import os
import pickle
import logging
from binding_prediction.models import GraphAndConvStack
from binding_prediction.dataset import DrugProteinDataset
from torch import optim
from torch.utils.data import DataLoader


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
    parser.add_argument('--hidden_channels', '-c', nargs='*', type=int, default=[10],
                        help='Number of channels to use in the hidden layers')
    parser.add_argument('--conv_kernel_sizes', '-k', nargs='*', type=int, default=None,
                        help='Number of channels to use in the hidden layers')
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Use CUDA (default)')
    parser.add_argument('--no-cuda', '--cpu', dest='cuda', action='store_false',
                        help='Use CPU')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()
    return args


def calculate_loss(output, batch):
    #### NEEDS CHANGING WITH DATASET / TASK ! ####
    return -1
    ##############################################


def run_model_on_batch(model, batch):
    #### NEEDS CHANGING WITH DATASET! ####
    adj_mat = batch['adj_mat']
    features = batch['features']
    out_features = model(adj_mat, features)[-1]
    return out_features
    ######################################


def main():
    args = _parse_args()
    logging.basicConfig(filename=args.dir + '/training.log', filemode='a', level=logging.DEBUG)

    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    # Save the construction arguments for future reference.
    with open(args.dir + '/training_args.pkl', 'wb') as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

    #### NEEDS CHANGING WITH DATASET! ####
    train_dataset = DrugProteinDataset(args.train_dataset)
    valid_dataset = DrugProteinDataset(args.valid_dataset)
    ######################################

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=True)

    in_channels = train_dataset[0].shape[-1]
    out_channels = 1
    model = GraphAndConvStack(in_channels, args.hidden_channels, out_channels)
    model = model.to(device=device)
    logging.info("Initialized Model.")

    if os.path.isfile(args.dir + '/model_best.pt'):
        logging.info("Previous Model found.  Attempting to load previous best model...")
        model_param_dict = torch.load('models/model_best.pt')
        model.load_state_dict(model_param_dict)
        logging.info("Succesfully loaded previous model")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    best_valid_loss = 1E8

    for n in range(args.num_epoch):
        model.train()
        total_train_loss = 0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = run_model_on_batch(batch, model)
            loss = calculate_loss(output, batch)
            total_train_loss += loss
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                logging.info("Batch {}/{}.  Batch loss: {}".format(i, len(train_dataloader), loss))

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader):
                output = run_model_on_batch(batch, model)
                loss = calculate_loss(output, batch)
                total_valid_loss += loss

        avg_train_loss = total_train_loss / len(train_dataset)
        avg_valid_loss = total_valid_loss / len(valid_dataset)
        logging.info("Epoch {} Complete. Train loss: {}.  Valid loss: {}.".format(avg_train_loss, avg_valid_loss))
        torch.save(model.state_dict(), 'models/model_current.pt')
        if avg_valid_loss < best_valid_loss:
            logging.info("Best validation loss achieved!")
            torch.save(model.state_dict(), 'models/model_best.pt')
            best_valid_loss = avg_valid_loss
