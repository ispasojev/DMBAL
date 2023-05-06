# %%
import argparse
import pathlib
import random
from typing import Dict, Any

import numpy as np
import torch

from src.data.DataLoader import get_dataset, get_handler_and_args
from src.database.mlflow import MLFlowClient
from src.model.resnet import ResNet34
from src.strategies import Centroids, BoundaryPoints
from src.strategies.random import Random

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def dmbal_loop(settings: Dict[str, Any]):
    # Überprüft, ob wir GPUs finden & wenn ja dann device = gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'----- device: {device}')

    # database
    database = MLFlowClient()
    run_id, output_path = database.init_experiment(hyper_parameters=settings)
    # ordner in dem ich alles speichern kann , irgendwas mit torch.save
    # jupyter notebook erstellen und torch.load(path) oder sowas um das aus output_path anzeigen zu lassen
    output_path = pathlib.Path(output_path)

    # load data
    X_tr, Y_tr, X_te, Y_te = get_dataset("CIFAR10")
    data_handler, d_args = get_handler_and_args("CIFAR10")
    d_args["max_epochs"] = settings["max_epochs"]
    d_args["loader_tr_args"]["batch_size"] = settings["batch_size"]
    n_pool = len(Y_tr)

    # initialize net
    model = ResNet34

    # initial random sampling
    idxs_lb = np.zeros(n_pool, dtype=bool)  # all points in pool False = not labelled
    idxs_tmp = np.arange(n_pool)  # idx 0-len(pool) for idx
    np.random.shuffle(idxs_tmp)  # randomly shuffle
    idxs_lb[idxs_tmp[:settings["sampling_size"]]] = True  # set to True = labelled

    strategy = set_diversityStrategy(X_tr,
                                      Y_tr,
                                      idxs_lb,
                                      model,
                                      data_handler,
                                      {
                                          "optimizer": settings["optimizer"],
                                          "learning_rate": settings["learning_rate"]
                                      },
                                      d_args,
                                      settings)
    strategy.set_path(output_path)

    # Pre-train on the initial subset
    strategy.train()
    P = strategy.predict(X_te, Y_te)
    current_test_acc = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    currently_labeled = np.sum(idxs_lb)
    result = {"acc": current_test_acc, "samples_labeled": currently_labeled}
    counter = currently_labeled
    database.log_results(result=result, step=counter)

    # Budget-Loop starts
    # Aus budget dann sample definieren
    remaining_budget = settings["budget"]
    remaining_budget = remaining_budget - currently_labeled

    print(f"Start AL loop")
    iter = 0
    while remaining_budget > 0:
        iter += 1
        strategy.set_current_round(iter)
        print(f"Remaining budget: {remaining_budget}")
        print(f"Currently labeled samples: {currently_labeled}")

        # query
        q_idxs = strategy.query(settings["sampling_size"])
        idxs_lb[q_idxs] = True

        # update
        strategy.update(idxs_lb)
        strategy.train()

        # accuracy
        P = strategy.predict(X_te, Y_te)
        currently_labeled = np.sum(idxs_lb)
        current_test_acc = 1.0 * (Y_te == P).sum().item() / len(Y_te)

        # log
        result = {"acc": current_test_acc, "samples_labeled": currently_labeled}
        counter = currently_labeled
        database.log_results(result=result, step=counter)

        remaining_budget = remaining_budget - settings["sampling_size"]  # update budget

    print('Finished Training')
    # While-Loop ends

    torch.save(model.state_dict(), output_path / './cifar_net.pth')
    database.finalise_experiment()
    print("dmbal finished successfully")

def set_diversityStrategy(X_tr, Y_tr, idxs_lb, model, data_handler, optimizer, d_args, settings):
    if settings["diversity_strategy"] == 'centroids':
        strategy = Centroids(
            X_tr,
            Y_tr,
            idxs_lb,
            model,
            data_handler,
            optimizer,
            d_args,
            k=settings["k"],
            beta=settings["beta"],
            n_clusters=settings["number_clusters"],
            uncertainty=settings["uncertainty"],
            weighted=True if settings["clustering"] == "w_kmeans" else False
        )
    elif settings["diversity_strategy"] == 'random':
        strategy = Random(
            X_tr,
            Y_tr,
            idxs_lb,
            model,
            data_handler,
            optimizer,
            d_args
        )
    else:
        strategy = BoundaryPoints(
            X_tr,
            Y_tr,
            idxs_lb,
            model,
            data_handler,
            optimizer,
            d_args,
            settings["diversity_strategy"],
            k=settings["k"],
            beta=settings["beta"],
            n_clusters=settings["number_clusters"],
            uncertainty=settings["uncertainty"],
            weighted=True if settings["clustering"] == "w_kmeans" else False
        )
    return strategy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST'])
    parser.add_argument('--model', type=str, default='resnet34', choices=['resnet34'])
    parser.add_argument('--strategy', type=str, default='boundary_points',
                        choices=['centroids', 'boundary_points', 'bp_smo_canceled', 'bp_lmo_canceled',
                                 'bp_prop_clustersize', 'random'])
    parser.add_argument('--k', type=int, default=1_000)
    parser.add_argument('--beta', type=int, default=10)
    parser.add_argument('--budget', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=110)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sampling_size', type=int, default=1_000)
    parser.add_argument('--number_clusters', type=int, default=100)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=int, default=0.01)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--clustering', type=str, default='kmeans', choices=['kmeans', 'w_kmeans'])
    parser.add_argument('--uncertainty', type=str, default='margin', choices=['margin', 'least_confident', 'entropy'])
    parser.add_argument('--validation', type=bool, default=False)
    args = parser.parse_args()

    config = {
        "budget": args.budget,
        "beta": args.beta,
        "k": args.k,
        "batch_size": args.batch_size,
        "sampling_size": args.sampling_size,
        "number_clusters": args.number_clusters,
        "diversity_strategy": args.strategy,
        "dataset": args.ds,
        "model": args.model,
        "random_seed": args.seed,
        "validation": args.validation,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "clustering": args.clustering,
        "uncertainty": args.uncertainty
    }
    set_seed(config['random_seed'])
    dmbal_loop(settings=config)
