"""
Helper functions for PyTorch-Apps
04/17/2020 Bandhav Veluri, Lin Cheng
"""

import argparse
import copy
import numpy as np
import random
import time
import torch
from tqdm import tqdm


def parse_model_args(workload_args=None):
    """
    Parse command line options.
    If a workload has options that are specific to it, it should pass in a
    function which adds those arguments
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Common options
    parser.add_argument('--nepoch', default=30, type=int,
                        help="number of training epochs")
    parser.add_argument('--nbatch', default=-1, type=int,
                        help="number of training/inference batches")
    parser.add_argument('--batch-size', default=32, type=int,
                        help="size of each batch")
    parser.add_argument('--hammerblade', default=False, action='store_true',
                        help="run MLP MNIST on HammerBlade")
    parser.add_argument('--training', default=False, action='store_true',
                        help="run training phase")
    parser.add_argument('--inference', default=False, action='store_true',
                        help="run inference phase")
    parser.add_argument("-v", "--verbose", default=0, action='count',
                        help="increase output verbosity")
    parser.add_argument("--save-model", default=False, action='store_true',
                        help="save trained model to file")
    parser.add_argument("--load-model", default=False, action='store_true',
                        help="load trained model from file")
    parser.add_argument('--model-filename', default="trained_model", type=str,
                        help="filename of the saved model")
    parser.add_argument('--seed', default=42, type=int,
                        help="manual random seed")
    parser.add_argument("--dry", default=False, action='store_true',
                        help="dry run")

    # Inject workload specific options
    if workload_args is not None:
        workload_args(parser)

    # Parse arguments
    args = parser.parse_args()

    # By default, we do both training and inference
    if (not args.training) and (not args.inference):
        args.training = True
        args.inference = True

    # If nbatch is set, nepoch is forced to be 1
    if args.nbatch == -1:
        args.nbatch = None
    else:
        args.nepoch = 1

    # Set random number seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)
    random.seed(args.seed + 2)

    # Dump configs
    if args.verbose > 0:
        print(args)

    return args

# -------------------------------------------------------------------------
# Model saving
# -------------------------------------------------------------------------


def save_model(model, model_filename):
    print("Saving model to " + model_filename)
    model_cpu = copy.deepcopy(model)
    model_cpu.to(torch.device("cpu"))
    torch.save(model_cpu.state_dict(), model_filename)

# -------------------------------------------------------------------------
# Common training routine
# -------------------------------------------------------------------------


def train(model, loader, optimizer, loss_func, args):
    print('Training {} for {} epoch(s)...'.format(
        type(model).__name__,
        args.nepoch
    ))

    # Timer
    training_start = time.time()

    # Prep model for *training*
    model.train()

    for epoch in range(args.nepoch):
        losses = []

        for batch_idx, (data, labels) in \
                tqdm(enumerate(loader, 0), total=len(loader)):
            if args.hammerblade:
                data, labels = data.hammerblade(), labels.hammerblade()
            optimizer.zero_grad()
            outputs = model(data)
            if args.verbose > 1:
                print("outputs:")
                print(outputs)
            loss = loss_func(outputs, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if (args.nbatch is not None) and (batch_idx + 1 >= args.nbatch):
                break

        print('epoch {} : Average Loss={:.6f}\n'.format(
            epoch,
            np.mean(losses)
        ))

    print("--- %s seconds ---" % (time.time() - training_start))

# -------------------------------------------------------------------------
# Common inference routine
# -------------------------------------------------------------------------


@torch.no_grad()
def inference(model, loader, loss_func, collector_func, args):
    test_loss = []

    print('Predicting with {} ...'.format(type(model).__name__))

    # Timer
    inference_start = time.time()

    # Prep model for *evaluation*
    model.eval()

    for batch_idx, (data, labels) in \
            tqdm(enumerate(loader, 0), total=len(loader)):
        if args.hammerblade:
            data, labels = data.hammerblade(), labels.hammerblade()
        outputs = model(data)
        if args.verbose > 1:
            print("outputs:")
            print(outputs)
        loss = loss_func(outputs, labels)
        test_loss.append(loss.item())
        collector_func(outputs, labels)

        if batch_idx + 1 == args.nbatch:
            break

    print("--- %s seconds ---" % (time.time() - inference_start))

    print('Test set: Average loss={:.4f}'.format(np.mean(test_loss)))
