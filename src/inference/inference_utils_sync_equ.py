"""
This files contains the gengeral-purpose functions to train the model.

"""

import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from features.build_pytorch_dataset import InfobitDataset
from train.train_utils_sync_equ import generate_random_delay


def inference_loop(
    model_type,
    model,
    batch_size,
    test_dataloader,
    test_snr_min,
    test_snr_max,
    delay_max,
):
    """
    inference loop for the well trained model.


    """

    print("Starting inference ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # inference snr_db values is a list from test_snr_min to test_snr_max with step 0.5, contains 2 sides of the SNR_db
    inference_snr_db = torch.arange(test_snr_min, test_snr_max + 0.5, 2)

    # initialize the list to store the bit error rate and block error rate
    infer_ber_list = []  # list to store the bit error rate
    infer_bler_list = []  # list to store the block error rate

    # switch to evaluation mode
    model.eval()

    # turn off gradients
    with torch.no_grad():

        for i in range(len(inference_snr_db)):

            # initialize the list to store the bit error rate and block error rate
            ber_snr_i = []  # list to store the bit error rate
            bler_snr_i = []  # list to store the block error rate

            for batch_index, data in enumerate(test_dataloader, 1):

                # randomly generate delay for each message
                # delay has the size (batch_size, 1),
                # delay_onehot has the size (batch_size, 1, delay_max + 1)
                true_delay, true_delay_onehot = generate_random_delay(
                    batch_size=batch_size, delay_max=delay_max, device=device
                )

                # get the features and targets, X has shape (batch_size, 1, k)
                X = data.to(device)

                # prediction by the model, shape (batch_size, 1, k)
                estimated_delay, predictions = model.forward(
                    X,
                    true_delay,
                    true_delay_onehot,
                    SNR_db=inference_snr_db[i],
                    device=device,
                    training=False,
                )

                # round the predictions to 0 or 1
                predictions_round = torch.round(predictions)

                # compute the bit error rate and block error rate
                infer_ber_batch, infer_bler_batch = compute_ber_bler(
                    predictions_round, X
                )

                # append the bit error rate and block error rate to the list
                ber_snr_i.append(infer_ber_batch)
                bler_snr_i.append(infer_bler_batch)

            # compute the average bit error rate and block error rate
            infer_ber_avg = sum(ber_snr_i) / len(ber_snr_i)
            infer_bler_avg = sum(bler_snr_i) / len(bler_snr_i)

            # print
            print(
                f"For SNR = {inference_snr_db[i]}, the average BER is {infer_ber_avg}, the average BLER is {infer_bler_avg}"
            )

            # append the average bit error rate and block error rate to the list
            infer_ber_list.append(infer_ber_avg.item())
            infer_bler_list.append(infer_bler_avg.item())

    # define our custom x axis metric
    wandb.define_metric("custom_step")

    # define which metrics will be plotted against it
    # first plot for BER
    wandb.define_metric(
        f"BER v.s. SNR (Joint Sync and Decoding, Block Fading Channel)",
        step_metric="custom_step",
    )

    # second plot for BLER
    wandb.define_metric(
        f"BLER v.s. SNR (Joint Sync and Decoding, Block Fading Channel)",
        step_metric="custom_step",
    )

    for i in range(len(inference_snr_db)):

        log_dict = {
            f"BER v.s. SNR (Joint Sync and Decoding, Block Fading Channel)": infer_ber_list[
                i
            ],
            "custom_step": inference_snr_db[i],
        }
        wandb.log(log_dict)

        log_dict = {
            f"BLER v.s. SNR (Joint Sync and Decoding, Block Fading Channel)": infer_bler_list[
                i
            ],
            "custom_step": inference_snr_db[i],
        }
        wandb.log(log_dict)

    print("The snr_db values for inference are: ", inference_snr_db)
    print("The average bit error rate is: ", infer_ber_list)
    print("The average block error rate is: ", infer_bler_list)

    print("Finished inference!")

    return model


def compute_ber_bler(predictions, targets):
    """
    Compute the bit error rate and block error rate.

    Args:
        predictions (torch.Tensor): The predicted values of shape (batch_size, 1, k).
        targets (torch.Tensor): The target values of shape (batch_size, 1, k).

    Returns:
        float: The bit error rate.
        float: The block error rate.
    """
    # compute the bit error rate
    ber = torch.sum(torch.abs(predictions - targets)) / (
        targets.shape[0] * targets.shape[2]
    )

    # compute the block error rate
    bler = (
        torch.sum(torch.any(torch.abs(predictions - targets), dim=2)) / targets.shape[0]
    )

    return ber, bler
