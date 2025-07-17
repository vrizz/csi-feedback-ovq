import argparse
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch import optim

from cost_loader import get_cost_dataset
from metrics import normalized_mean_square_error, cosine_similarity
from models.crnet_ovq import crnet_ovq
from models.transnet_ovq import transnet_ovq


def parse_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Parse model arguments for TransnetOVQ model.")

    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help="Dataset type indoor or outdoor.")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Name or path of the base model.")
    parser.add_argument('-r', '--reduction', type=int, required=True,
                        help="Reduction value to determine latent dimensions (e.g., 4, 8, 16, 32).")
    parser.add_argument('-b', '--bits', type=int, required=True,
                        help="Batch size or bits for training or inference.")
    parser.add_argument('-e', '--embedding_dim', type=int, required=True,
                        help="Dimensionality of embeddings used in the model.")

    # Add fine-tuning flag
    parser.add_argument('-ft', '--fine_tuning', action='store_true',
                        help="Enable fine-tuning mode if set.")

    # Parse the command-line arguments
    args = parser.parse_args()

    return args


def transform_data(x_hat):
    img_height = 32
    img_width = 32

    x_hat = x_hat - 0.5

    x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    x_hat_C = x_hat_real + 1j * x_hat_imag
    x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
    X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257 - img_width))), axis=2), axis=2)
    X_hat = X_hat[:, :, 0:125]

    data_real = np.real(X_hat)
    data_imag = np.imag(X_hat)
    data_combined = np.stack([data_real, data_imag], axis=1)

    return data_combined


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed}")


def get_model(model_name, reduction, vq_params, fine_tuning):
    # Map reduction values to latent_dim
    reduction_map = {
        2: 1024,
        4: 512,
        8: 256,
        16: 128,
        32: 64
    }

    model_map = {
        "transnet": transnet_ovq(vq_params=vq_params, reduction=reduction, fine_tuning=fine_tuning),
        "crnet": crnet_ovq(vq_params=vq_params, reduction=reduction, fine_tuning=fine_tuning),
    }

    print("Latent space size is ", reduction_map[reduction])
    return model_map[model_name]



def test(model, test_loader, device):
    model.to(device)
    model.eval()

    all_metrics = []

    embedding_dim = model.embedding_dim
    latent_dim = model.latent_dim


    my_range = np.arange(embedding_dim, latent_dim + 1, embedding_dim)
    b = int(model.b)

    for start_idx in my_range:

        B = b * (start_idx // embedding_dim)
        print(B)
        inputs_list = []
        outputs_list = []
        inputs_raw_list = []

        enc_idx_list = []

        with torch.no_grad():
            for idx, (inputs, inputs_raw) in enumerate(test_loader):
                inputs = inputs.to(device)
                outputs, _ = model.evaluate(inputs, start_idx=start_idx)

                # enc_idx_np = vq_params['encoding_indices'].cpu().numpy()

                inputs_np = inputs.cpu().numpy()
                inputs_raw_np = inputs_raw.cpu().numpy()
                outputs_np = outputs.cpu().numpy()

                # enc_idx_list.append(enc_idx_np)

                inputs_list.append(inputs_np)
                inputs_raw_list.append(inputs_raw_np)
                outputs_list.append(outputs_np)

        inputs_combined = np.concatenate(inputs_list, axis=0)
        inputs_raw_combined = np.concatenate(inputs_raw_list, axis=0)
        outputs_combined = np.concatenate(outputs_list, axis=0)

        # enc_idx_comb = np.concatenate(enc_idx_list, axis=0)

        nmse = normalized_mean_square_error(outputs_combined - 0.5, inputs_combined - 0.5)

        outputs_transformed = transform_data(outputs_combined)

        gcs = cosine_similarity(outputs_transformed, inputs_raw_combined)

        metrics = {'B': B, 'NMSE [dB]': 10 * math.log10(nmse), 'GCS': gcs}
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('test_metrics.csv', index=False)
    print(f"Test metrics saved to 'test_metrics.csv'")

    table = wandb.Table(dataframe=metrics_df)
    wandb.log({"Test Metrics": table})


def train(model, config, train_loader, val_loader, device):
    learning_rate = config['learning_rate']
    epochs = config['epochs']

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    criterion = nn.MSELoss().to(device)

    best_val_loss = float('inf')
    best_model_path = "best_model.pth"


    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_perplexity = 0  # To track the total perplexity over the epoch
        perplexity_count = 0  # To track the number of perplexities we sum


        for inputs, in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs, vq_returns = model(inputs)

            # Calculate reconstruction loss and vector quantization loss
            reco_loss = criterion(outputs, inputs)
            vq_loss = vq_returns["loss"]
            loss = reco_loss + vq_loss
            loss.backward()
            optimizer.step()

            # Perplexity and loss tracking
            perplexity = vq_returns["perplexity"]
            total_loss += loss.item()
            total_perplexity += perplexity.item()
            perplexity_count += 1

        avg_train_loss = total_loss / len(train_loader)
        avg_train_perplexity = total_perplexity / perplexity_count  # Calculate average perplexity

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, in val_loader:
                inputs = inputs.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, inputs)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # scheduler.step(avg_val_loss)

        # Log average training loss, validation loss, and perplexity
        log_data = {
            'Train Loss': avg_train_loss,
            'Validation Loss': avg_val_loss,
            'Avg Train Perplexity': avg_train_perplexity
        }
        wandb.log(log_data)

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Avg Train Perplexity: {avg_train_perplexity}')

        # Save the best model if the validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                torch.save(model.state_dict(), best_model_path)
                print(f"Model saved to {best_model_path}")
            except PermissionError as e:
                print(f"PermissionError: {e}")
                print("Ensure you have write permissions to the directory.")

            print(f'New best model saved at epoch {epoch + 1} with val_loss: {best_val_loss}')

    # Upload the best model artifact to WandB
    artifact = wandb.Artifact("best_model.pth", type='model')
    artifact.add_file(best_model_path)
    wandb.log_artifact(artifact)
    wandb.finish()


def get_run_id_by_name(project_name=None, run_name=None):
    """
    Fetch the run_id of a specific run by its name or print all available runs in the project.

    Parameters:
    - project_name: The name of the W&B project.
    - run_name: The name of the specific run (optional). If provided, returns the run_id of that run.

    Returns:
    - run_id: The run ID of the matched run or None if not found.
    """
    api = wandb.Api()
    runs = api.runs(project_name)

    if run_name:
        # Search for the run by name
        for run in runs:
            if run.name == run_name:
                print(f"Found run: {run.name} with run_id: {run.id}")
                return run.id
        print(f"Run with name '{run_name}' not found.")
        return None
    else:
        # List all available runs
        print("Available runs:")
        for run in runs:
            print(f"Run name: {run.name}, Run ID: {run.id}")
        return None


def load_best_model_from_wandb(model, artifact_name="best_model.pth", project_name=None, run_name=None):
    """
    Load the best model saved in the wandb artifact from a specific run without starting a new run.

    Parameters:
    - model: The model to load the state_dict into.
    - artifact_name: The name of the artifact to download.
    - run_name: The name of the specific run to fetch the artifact from.
    """
    # Fetch the run_id of the specific run
    run_id = get_run_id_by_name(project_name, run_name)

    if run_id:
        # Use the specific run_id to access the artifact directly
        api = wandb.Api()

        run = api.run(f"{project_name}/{run_id}")

        # Retrieve all artifacts logged in this run
        artifacts = run.logged_artifacts()
        artifact = None

        # Find the artifact matching the given name
        for art in artifacts:
            if artifact_name in art.name:
                artifact = art
                break

        if artifact is None:
            print(f"Artifact '{artifact_name}' not found in run '{run_name}'.")
            return None

        artifact_dir = artifact.download()

        # Load the checkpoint with non-strict loading
        checkpoint = torch.load(f"{artifact_dir}/{artifact_name}", map_location=torch.device('cpu'))

        # Load the state dict with missing or unexpected keys
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        if missing_keys:
            print(f"Missing keys in model: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {unexpected_keys}")

        print(f"Best model loaded from run {run_id} at {artifact_dir}/{artifact_name}")
    else:
        print(f"Run with name '{run_name}' not found, could not load model.")

    return model


if __name__ == "__main__":
    seed = 42
    set_seed(seed)

    args = parse_arguments()

    # Access parsed arguments with args.<argument_name>
    scenario = args.dataset
    base_model = args.model
    reduction = args.reduction
    b = args.bits
    embedding_dim = args.embedding_dim

    fine_tuning = args.fine_tuning

    # Now base_model, reduction, b, and embedding_dim contain the parsed values
    print(f"Base Model: {base_model}")
    print(f"Reduction: {reduction}")
    print(f"Bits b: {b}")
    print(f"Embedding dim: {embedding_dim}")

    print(f"Fine tuning: {fine_tuning}")

    if base_model == 'transnet':
        if fine_tuning:
            epochs = 1000
        else:
            epochs = 1000
    else:
        if fine_tuning:
            epochs = 1000
        else:
            epochs = 2000

    base_model_config = {
        'transnet': {'batch_size': 200, 'learning_rate': 1e-4, 'epochs': epochs},
        'crnet': {'batch_size': 200, 'learning_rate': 1e-3, 'epochs': epochs},
    }

    # define vq_params
    vq_params = {
        "embedding_dim": embedding_dim,
        "num_embeddings": 2 ** b,
        "commitment_cost": 0.25,
    }

    model = get_model(base_model, reduction, vq_params, fine_tuning)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    train_loader, val_loader, test_loader = get_cost_dataset(scenario=scenario,
                                                             batch_size=base_model_config[base_model]['batch_size'])

    project_name_pretrain = "csi-feedback-ovq-pretrain"
    project_name_fine_tune = "csi-feedback-ovq-fine-tune"

    run_name = f"{base_model}_env_{scenario}_reduction_{reduction}_bits_{b}_embedding_dim_{embedding_dim}"

    if fine_tuning:
        project_name = project_name_fine_tune
        wandb.init(project=project_name, name=run_name)

        # load the best model checkpoint from the pretrained project
        model = load_best_model_from_wandb(model, artifact_name="best_model.pth", project_name=project_name_pretrain,
                                           run_name=run_name)

    else:
        project_name = project_name_pretrain
        wandb.init(project=project_name, name=run_name)

    train(model, base_model_config[base_model], train_loader, val_loader, device)

    run_id = get_run_id_by_name(project_name, run_name)

    if run_id:
        # Initialize wandb and resume the previous run using the run_id
        wandb.init(project=project_name, id=run_id, resume="must", reinit=False)

        # checkpoint = torch.load('best_model.pth')
        # model.load_state_dict(checkpoint, strict=False)

        # Load your model and test it
        model = load_best_model_from_wandb(model, artifact_name="best_model.pth", project_name=project_name,
                                           run_name=run_name)
        test(model, test_loader, device)

        # artifact = wandb.Artifact("best_model.pth", type='model')
        # artifact.add_file("best_model.pth")
        # wandb.log_artifact(artifact)
        # wandb.finish()

    else:
        print(f"Run not found. Exiting.")
