import torch
import argparse
import os
import pickle
import datetime
import torch.nn.functional as F
from binding_prediction.models import GraphAndConvStack
from binding_prediction.dataset import DrugProteinDataset
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class BindingModelNCE(torch.nn.Module):
    """
    Model for predicting weather or not we achieve binding.

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the input tensor
    hidden_channel_list : iterable of ints
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of hidden layers.
    out_channels : int
        number of output channels.
    conv_kernel_sizes : iterable of ints, optional
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of layers.
        If not provided, defaults to 1 for every layer
    """

    def __init__(self, in_channels, hidden_channel_list, out_channels,
                 conv_kernel_sizes=None, nonlinearity=None):
        super(BindingModelNCE, self).__init__()
        self.gcs_stack = GraphAndConvStack(in_channels, hidden_channel_list,
                                           out_channels, conv_kernel_sizes,
                                           nonlinearity)
        total_number_inputs = sum(hidden_channel_list) + out_channels
        self.final_mix = torch.nn.Linear(total_number_inputs, 1, bias=False)

    def forward(self, adj, x):
        x_all = self.gcs_stack(adj, x)
        x_all = torch.cat(x_all, dim=-1)
        x_out = self.final_mix(x_all)
        x_out = torch.sum(torch.sum(x_out, dim=2), dim=1)
        return x_out


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
    targets = batch['is_true']
    bce_loss = F.binary_cross_entropy_with_logits(output, targets)
    return bce_loss


def run_model_on_batch(model, batch):
    #### NEEDS CHANGING WITH DATASET! ####
    adj_mat = batch['adj_mat']
    features = batch['features']
    out_features = model(adj_mat, features)[-1]
    return out_features
    ######################################


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

    #### NEEDS CHANGING WITH DATASET! ####
    train_dataset = DrugProteinDataset(args.train_dataset, prob_fake=0.5)
    valid_dataset = DrugProteinDataset(args.valid_dataset, prob_fake=0.5)
    ######################################

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=True)

    in_channels = train_dataset[0].shape[-1]
    out_channels = 1
    model = BindingModelNCE(in_channels, args.hidden_channels, out_channels)
    model = model.to(device=device)
    writer.add_text("Initialized Model.")

    if os.path.isfile(args.dir + '/model_best.pt'):
        writer.add_text("Previous Model found.  Attempting to load previous best model...")
        model_param_dict = torch.load('models/model_best.pt')
        model.load_state_dict(model_param_dict)
        writer.add_text("Succesfully loaded previous model")

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
                print.info("Batch {}/{}.  Batch loss: {}".format(i, len(train_dataloader), loss))

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader):
                output = run_model_on_batch(batch, model)
                loss = calculate_loss(output, batch)
                total_valid_loss += loss

        avg_train_loss = total_train_loss / len(train_dataset)
        avg_valid_loss = total_valid_loss / len(valid_dataset)
        print("Epoch {} Complete. Train loss: {}.  Valid loss: {}.".format(avg_train_loss, avg_valid_loss))
        writer.add_scalar('training_loss', avg_train_loss)
        writer.add_scalar('validation_loss', avg_valid_loss)

        torch.save(model.state_dict(), 'models/model_current.pt')
        if avg_valid_loss < best_valid_loss:
            writer.add_text("Best validation loss achieved at %d." % n)
            torch.save(model.state_dict(), 'models/model_best.pt')
            best_valid_loss = avg_valid_loss
