# copied directly from https://gitlab.cern.ch/LHCb-Reco-Dev/pv-finder/-/blob/master/model/training.py
import time
import torch
from collections import namedtuple
import sys
import os

from .utilities import tqdm_redirect, import_progress_bar

Results = namedtuple("Results", ["epoch", "cost", "val", "time"])

def trainNet(
    model,
    optimizer,
    loss,
    train_loader,
    val_loader,
    n_epochs,
    *,
    notebook=None,
    epoch_start=0,
):
    """
    If notebook = None, no progress bar will be drawn. If False, this will be a terminal progress bar.
    """

    # Print all of the hyperparameters of the training iteration
    if not notebook:
        print("{0:=^80}".format(" HYPERPARAMETERS "))
        print(
            f"""\
n_epochs: {n_epochs}
batch_size: {train_loader.batch_size} events
dataset_train: {train_loader.dataset.tensors[0].size()[0]} events
dataset_val: {val_loader.dataset.tensors[0].size()[0]} events
loss: {loss}
optimizer: {optimizer}
model: {model}"""
        )
        print("=" * 80)

    # Set up notebook or regular progress bar (or none)
    progress = import_progress_bar(notebook)
    
    print(f"Number of batches: train = {len(train_loader)}, val = {len(val_loader)}")

    epoch_iterator = progress(
        range(epoch_start, n_epochs),
        desc="Epochs",
        postfix="train=start, val=start",
        dynamic_ncols=True,
        position=0,
        file=sys.stderr,
    )

    # Loop for n_epochs
    for epoch in epoch_iterator:
        #print("Epoch: ", epoch)
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = validate(model, loss, val_loader)
        val_epoch = total_val_loss / len(val_loader)

        # Record total time
        time_epoch = time.time() - training_start_time

        # Pretty print a description
        if hasattr(epoch_iterator, "postfix"):
            epoch_iterator.postfix = f"train={cost_epoch:.4}, val={val_epoch:.4}"

        # Redirect stdout if needed to avoid clash with progress bar
        write = getattr(progress, "write", print)
        write(
            f"Epoch {epoch}: train={cost_epoch:.6}, val={val_epoch:.6}, took {time_epoch:.5} s"
        )

        yield Results(epoch, cost_epoch, val_epoch, time_epoch)


def train(model, loss, loader, optimizer, progress):
    total_loss = 0.0

    # switch to train mode
    model.train()
    loader = progress(
        loader,
        postfix="train=start",
        desc="Training",
        mininterval=0.5,
        dynamic_ncols=True,
        position=1,
        leave=False,
        file=sys.stderr,
    )
    for idx, batch in enumerate(loader):
        # Set the parameter gradients to zero
        optimizer.zero_grad()
        # Forward pass, backward pass, optimize
        outputs = model(batch.data, batch.mask)
        loss_output = loss(
            input=outputs[0],
            target=batch.target,
            mask=batch.mask,
            length=batch.length
        )
        loss_output.backward()
        optimizer.step()

        total_loss += loss_output.data.item()
        if hasattr(loader, "postfix"):
            loader.postfix = f"train={loss_output.data.item():.4g}"

    return total_loss


def validate(model, loss, loader):
    total_loss = 0

    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(loader):

            # Forward pass
            val_outputs = model(batch.data, batch.mask)
            loss_output = loss(
                input=val_outputs[0],
                target=batch.target,
                mask=batch.mask,
                length=batch.length
            )

            total_loss += loss_output.data.item()

    return total_loss
