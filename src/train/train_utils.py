"""
This files contains the gengeral-purpose functions to train the model.

"""

import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from features.build_pytorch_dataset import InfobitDataset


def training_loop(
    model_type,
    model,
    optimizer,
    loss_fn,
    num_epochs,
    training_steps,
    batch_size,
    start_epoch,
    print_every,
    save_model_name,
    save_every,
    snr_min,
    snr_max,
    k,
):
    """
    Training loop for the transformer_encoder_decoder model.

    Args:
        model_type (str): The type of model to be trained.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loss_fn (torch.nn.Module): The loss function used for training.
        num_epochs (int): The number of epochs to train the model.
        training_steps (int): The number of training steps to train the model.
        start_epoch (int): The epoch to start training from.
        print_every (int): The interval for printing training progress.
        save_model_name (str): The name of the model to be saved.
        save_every (int): The interval for saving the model checkpoints.
        snr_min (float): The minimum value of the snr.
        snr_max (float): The maximum value of the snr.


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

    # ramdonly generate a SNR_db list between snr_min and snr_max
    # use 1 value for each epoch
    training_snr_db = (
        torch.rand((num_epochs, training_steps)) * (snr_max - snr_min) + snr_min
    )

    for epoch in range(start_epoch, num_epochs + 1):

        train_loss_batches_per_epoch = []

        # generate random information bits of size (batch_size, 1, k)
        train_dataset = InfobitDataset(num_samples=1e6, k=k)
        val_dataset = InfobitDataset(num_samples=1e4, k=k)

        if epoch == 1:
            print("number of training samples: ", len(train_dataset))
            print("number of validation samples: ", len(val_dataset))

        # create the train and val dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for T in range(training_steps):

            # # generate random information bits of size (batch_size, 1, k)
            # train_dataset = InfobitDataset(num_samples=500, k=k)
            # val_dataset = InfobitDataset(num_samples=500, k=k)

            # # create the train and val dataloaders
            # train_dataloader = DataLoader(
            #     train_dataset, batch_size=batch_size, shuffle=True
            # )
            # val_dataloader = DataLoader(
            #     val_dataset, batch_size=batch_size, shuffle=False
            # )

            # train for one epoch
            if model_type == "CNN_AutoEncoder":
                model, train_loss_batches_per_training_step = train_one_training_step(
                    model,
                    optimizer,
                    loss_fn,
                    train_dataloader,
                    val_dataloader,
                    device,
                    print_every,
                    model_type="CNN_AutoEncoder",
                    training_snr_db=training_snr_db[epoch - 1, T],
                    snr_min=snr_min,
                    snr_max=snr_max,
                )

            # add the loss to the list, the list contains training_steps elements
            train_loss_batches_per_epoch.append(
                sum(train_loss_batches_per_training_step)
                / len(train_loss_batches_per_training_step)
            )

        # compute the average training loss for the epoch
        train_loss_one_epoch = sum(train_loss_batches_per_epoch) / len(
            train_loss_batches_per_epoch
        )

        # compute the validation loss
        val_loss_one_epoch = CNN_AutoEncoder_validate(
            model, loss_fn, val_dataloader, device, snr_min, snr_max
        )

        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"Train loss: {train_loss_one_epoch:.5f}, "
            f"Val. loss: {val_loss_one_epoch:.5f}"
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

    print("Finished Training!")
    return model


def train_one_training_step(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    device,
    print_every,
    model_type,
    training_snr_db,
    snr_min,
    snr_max,
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

    train_loss_batches_per_training_step = []
    num_batches = len(train_loader)

    for batch_index, data in enumerate(train_loader, 1):

        # get the features and targets, X has shape (batch_size, 12, 256), y has shape (batch_size, 4, 256)
        X = data.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        if model_type == "CNN_AutoEncoder":
            predictions = model.forward(X, SNR_db=training_snr_db)

        loss = loss_fn(predictions, X)

        # add the loss to the list
        train_loss_batches_per_training_step.append(loss.item())

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
                val_loss = CNN_AutoEncoder_validate(
                    model, loss_fn, val_loader, device, snr_min, snr_max
                )

            # switch back to training mode (since when calling validate() the model is switched to eval mode)
            model.train()

            print(
                f"\tBatch {batch_index}/{num_batches}: "
                f"\tTrain loss: {sum(train_loss_batches_per_training_step[-print_every:])/print_every:.3f}, "
                f"\tVal. loss: {val_loss:.3f}"
            )

    return model, train_loss_batches_per_training_step


def CNN_AutoEncoder_validate(
    model,
    loss_fn,
    val_loader,
    device,
    snr_min,
    snr_max,
):
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
    # val_loss_cum = 0
    val_loss_snr_list = []  # list to store the validation loss for each SNR_db value

    # generate a list of SNR_db values between snr_min and snr_max for validation
    val_snr_db = torch.arange(snr_min, snr_max + 0.5, 0.5)

    # switch to evaluation mode
    model.eval()

    # turn off gradients
    with torch.no_grad():

        # loop over the validation dataset for each SNR_db value
        for T_val in range(len(val_snr_db)):

            val_loss_cum = 0

            for batch_index, data_val in enumerate(val_loader, 1):

                # get the features and targets, X has shape (batch_size, 1, k)
                X_val = data_val.to(device)

                # prediction by the model, shape (batch_size, 1, k)
                predictions = model.forward(X_val, SNR_db=val_snr_db[T_val])

                # compute the loss
                batch_loss = loss_fn(predictions, X_val)

                # update the cummulative loss
                val_loss_cum += batch_loss.item()

            val_loss_snr_list.append(val_loss_cum / len(val_loader))

    return sum(val_loss_snr_list) / len(val_loss_snr_list)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model to count the parameters of.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
