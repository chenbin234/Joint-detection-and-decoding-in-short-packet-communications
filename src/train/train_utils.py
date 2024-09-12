"""
This files contains the gengeral-purpose functions to train the model.

"""

import torch
import torchvision
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
import wandb
import torch.nn as nn


def training_loop(
    model_type,
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    num_epochs,
    start_epoch,
    print_every,
    save_model_name,
    save_every,
):
    """
    Training loop for the transformer_encoder_decoder model.

    Args:
        model_type (str): The type of model to be trained.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loss_fn (torch.nn.Module): The loss function used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation dataset.
        num_epochs (int): The number of epochs to train the model.
        start_epoch (int): The epoch to start training from.
        print_every (int): The interval for printing training progress.
        save_model_name (str): The name of the model to be saved.
        save_every (int): The interval for saving the model checkpoints.

    Returns:
        torch.nn.Module: The trained model.
    """

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_fn, log="all", log_freq=5000)

    # record the number of trainable parameters in the model
    wandb.config.update(
        {"trainable_parameters": count_parameters(model)}, allow_val_change=True
    )
    print(f"Number of trainable parameters in the model: {count_parameters(model)}")

    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(start_epoch, num_epochs + 1):

        # train for one epoch
        if model_type == "CNN_AutoEncoder":
            model, train_loss_batches = train_one_epoch(
                model,
                optimizer,
                loss_fn,
                train_loader,
                val_loader,
                device,
                print_every,
                model_type="Transformer",
            )
            # compute the validation loss
            val_loss = CNN_AutoEncoder_validate(model, loss_fn, val_loader, device)

        # calculate the loss for the specific epoch
        train_loss_one_epoch = sum(train_loss_batches) / len(train_loss_batches)
        val_loss_one_epoch = val_loss

        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"Train loss: {train_loss_one_epoch:.3f}, "
            f"Val. loss: {val_loss_one_epoch:.3f}"
        )

        # Log the train_loss_one_epoch and val_loss_one_epoch to wandb
        wandb.log(
            {"train_loss": train_loss_one_epoch, "val_loss": val_loss_one_epoch},
            step=epoch,
        )

        # Saving model
        if (epoch) % save_every == 0:
            # Saving model, loss and error log files
            print(f"Saving model [epoch {epoch}]")
            # torch.save(model.state_dict(), os.path.join(save_location, save_model_name, 'epoch{}.pth'.format(epoch)))
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss_one_epoch,
                },
                f"./models/{save_model_name}/{save_model_name}_epoch{epoch}.pth",
            )

            # print(f"Saving checkpoint [epoch {epoch}]")
            # CHECKPOINT_PATH = f"./models/{save_model_name}/checkpoint.tar"
            # # Save our checkpoint loc
            # torch.save(
            #     {
            #         "epoch": epoch,
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "loss": train_loss_one_epoch,
            #     },
            #     CHECKPOINT_PATH,
            # )
            # wandb.save(CHECKPOINT_PATH)  # saves checkpoint to wandb

            # Save the model to wandb
            # torch.onnx.export(model, torch.randn(1, 12, 256), f'./models/{save_model_name}/{save_model_name}_epoch{epoch}.onnx', verbose=True)
            # wandb.save(f'./models/{save_model_name}/{save_model_name}_epoch{epoch}.onnx')

    print("Finished Training!")
    return model


def train_one_epoch(
    model, optimizer, loss_fn, train_loader, val_loader, device, print_every, model_type
):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loss_fn (callable): The loss function used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        val_loader (torch.utils.data.DataLoader): The data loader for validation data.
        device (torch.device): The device to run the training on.
        print_every (int): The interval for printing the training progress.
        model_type (str): The type of model to be trained.

    Returns:
        tuple: A tuple containing the accuracy and average loss for the epoch.
    """
    #
    model = model.train()

    train_loss_batches = []
    num_batches = len(train_loader)

    for batch_index, data in enumerate(train_loader, 1):

        # get the features and targets, X has shape (batch_size, 12, 256), y has shape (batch_size, 4, 256)
        X = data.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        if model_type == "CNN_AutoEncoder":
            predictions = model.forward(X)

        loss = loss_fn(predictions.reshape(X.size(0), -1), X.reshape(X.size(0), -1))

        # add the loss to the list
        train_loss_batches.append(loss.item())

        # backpropagate the loss
        loss.backward()

        # update the parameters
        optimizer.step()

        # If you want to print your progress more often than every epoch you can
        # set `print_every` to the number of batches you want between every status update.
        # Note that the print out will trigger a full validation on the full val. set => slows down training
        if print_every is not None and batch_index % print_every == 0:

            # compute the validation loss
            if model_type == "CNN_AutoEncoder":
                val_loss = CNN_AutoEncoder_validate(model, loss_fn, val_loader, device)

            # switch back to training mode (since when calling validate() the model is switched to eval mode)
            model.train()

            print(
                f"\tBatch {batch_index}/{num_batches}: "
                f"\tTrain loss: {sum(train_loss_batches[-print_every:])/print_every:.3f}, "
                f"\tVal. loss: {val_loss:.3f}"
            )

    return model, train_loss_batches


def CNN_AutoEncoder_validate(model, loss_fn, val_loader, device):
    """
    Function to validate the model on the whole validation dataset.

    Args:
        model (torch.nn.Module): The trained model.
        loss_fn (torch.nn.Module): The loss function.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to perform computations on.

    Returns:
        float: The average validation loss throughout the whole val dataset.
    """
    val_loss_cum = 0

    # switch to evaluation mode
    model.eval()

    # turn off gradients
    with torch.no_grad():

        for batch_index, data_val in enumerate(val_loader, 1):

            # get the features and targets, X has shape (batch_size, 15, 256), y has shape (batch_size, 5, 256)
            X_val = data_val.to(device)

            # prediction by the model, shape (batch_size, 1280), 5*256 = 1280
            predictions = model.forward(X_val)

            # compute the loss
            batch_loss = loss_fn(
                predictions.contiguous().view(X_val.size(0), -1),
                X_val.view(X_val.size(0), -1),
            )

            # update the cummulative loss
            val_loss_cum += batch_loss.item()

    return val_loss_cum / len(val_loader)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model to count the parameters of.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
