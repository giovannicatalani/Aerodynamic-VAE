import argparse
import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from Model_CVAE import ConditionalVariationalAutoencoder
from utilities_CVAE import run_epoch, run_val, predict_test, reconstruct, plot_writer
from DataLoader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='CVAE Reduced-Order Model for Transonic Aerodynamics'
    )
    parser.add_argument('--train', action='store_true',
                        help='Run training. If not set, runs inference with saved model.')
    parser.add_argument('--data_dir', type=str, default='../data/',
                        help='Path to data directory (default: ../data/)')
    parser.add_argument('--model_path', type=str, default='../models/best_model_CVAE.pt',
                        help='Path to save/load model checkpoint (default: ../models/best_model_CVAE.pt)')
    parser.add_argument('--res', type=int, default=128,
                        help='Grid resolution (default: 128). Must match preprocessed data.')
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='Latent space dimension (default: 10)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--kl_weight', type=float, default=1.0,
                        help='Weight for KL divergence term / Beta (default: 1.0)')
    parser.add_argument('--recon_idx', type=int, default=20,
                        help='Sample index to plot for reconstruction (default: 20)')
    parser.add_argument('--test_idx', type=int, default=44,
                        help='Sample index to plot for test prediction (default: 44)')
    return parser.parse_args()


def train(args, model, train_loader, valid_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    state = {'best_valid_loss': float('inf')}
    aux_loss = np.zeros((6, args.epochs))

    print(f'Starting training for {args.epochs} epochs...\n')

    for epoch in range(args.epochs):
        tr_loss, tr_recon, tr_kl = run_epoch(
            epoch, model, train_loader, optimizer,
            kl_weight=args.kl_weight, device=device
        )
        val_loss, val_recon, val_kl = run_val(
            epoch, model, valid_loader,
            kl_weight=args.kl_weight, device=device
        )

        aux_loss[:, epoch] = [tr_loss, tr_recon, tr_kl, val_loss, val_recon, val_kl]

        if val_loss < state['best_valid_loss']:
            state['best_valid_loss'] = val_loss
            state['epoch'] = epoch
            state['state_dict_CVAE'] = model.state_dict()
            state['optimizer_CVAE'] = optimizer.state_dict()
            torch.save(state, args.model_path)
            print(f'  [Epoch {epoch}] New best model saved (val_loss={val_loss:.6f})')

        print(
            f'Epoch {epoch:4d} | '
            f'Train loss: {tr_loss:.4f} (recon={tr_recon:.4f}, kl={tr_kl:.4f}) | '
            f'Val loss: {val_loss:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f})'
        )

    # Plot loss curves
    epochs = np.arange(args.epochs)
    plt.figure()
    plt.plot(epochs, aux_loss[0], label='Train total')
    plt.plot(epochs, aux_loss[1], label='Train recon')
    plt.plot(epochs, aux_loss[2], label='Train KL')
    plt.plot(epochs, aux_loss[3], label='Val total')
    plt.plot(epochs, aux_loss[4], label='Val recon')
    plt.plot(epochs, aux_loss[5], label='Val KL')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training curves')
    plt.tight_layout()
    plt.show()

    print('Training complete.')


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Data directory: {args.data_dir}')
    print(f'Model path: {args.model_path}')

    # Load data
    train_loader, valid_loader, test_loader, p_mean, p_std, ctrl_mean, ctrl_std = DataLoader(
        args.data_dir, args.res, args.batch_size
    )
    print('Dataset loaded.\n')

    # Build model
    control_dim = 2
    model = ConditionalVariationalAutoencoder(args.latent_dim, control_dim, args.res)
    model = model.to(device)
    print(f'Model: latent_dim={args.latent_dim}, res={args.res}x{args.res}, '
          f'kl_weight={args.kl_weight}\n')

    # Train or infer
    if args.train:
        train(args, model, train_loader, valid_loader, device)

    # Reconstruction on training set + latent space plot
    pred_train, true_train, ctrl_train, mu_train, _ = reconstruct(
        model, args.model_path, train_loader,
        p_mean, p_std, ctrl_mean, ctrl_std, device='cpu'
    )
    plot_writer(
        args.data_dir, args.res,
        pred_train, true_train, ctrl_train,
        args.recon_idx, mu=mu_train, latent_plot=True
    )

    # Prediction on test set
    pred_test, true_test, ctrl_test = predict_test(
        model, args.model_path, test_loader,
        p_mean, p_std, ctrl_mean, ctrl_std, device='cpu'
    )
    plot_writer(
        args.data_dir, args.res,
        pred_test, true_test, ctrl_test,
        args.test_idx
    )


if __name__ == '__main__':
    main()
