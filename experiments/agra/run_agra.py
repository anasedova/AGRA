import argparse
import os
import logging
import torch

from experiments.agra.utils import define_data_encoding_agra, \
    load_train_data_for_agra
from experiments.utils import define_eval_metric
from src.AGRA.logreg_model_with_AGRA import LogRegModelWithAGRA
from src.utils import set_seed, compute_weights

logger = logging.getLogger(__name__)

lr = 0.1
weight_decay = 0.00001
batch_size = 512
agra_threshold = 0
num_epochs = 10

# set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--train_labels_path", type=str, default=None)
    parser.add_argument("--gold_label_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default='youtube',
                        choices=['youtube', 'sms', 'trec', 'yoruba', 'hausa', 'cifar', 'chexpert'])
    parser.add_argument("--num_valid_samples", type=int, default=5000,
                        help="Size of a valid set to be sampled from the test set if no valid set is available")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoding", type=str, default=None)
    parser.add_argument("--finetuning", type=bool, default=False)
    parser.add_argument("--adaptive_threshold", type=bool, default=False)
    parser.add_argument("--include_bias", type=bool, default=False)
    parser.add_argument("--finetuning_epochs", type=int, default=2)
    parser.add_argument("--finetuning_batch_size", type=int, default=2)
    parser.add_argument("--modification", type=str, default="last")
    parser.add_argument("--other", type=int, default=None)
    parser.add_argument("--closs", type=str, default='F1', choices=['CE', 'F1'])
    parser.add_argument("--weights", type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--save', type=bool, default=True)

    args = parser.parse_args()
    logger.info(args)

    # set the seed
    if args.seed is not None:
        print(f"Seed {args.seed} is fixed.")
        set_seed(args.seed)

    # define the way how the dataset will be encoded
    encoding = define_data_encoding_agra(args)

    # define the path from which the dataset will be loaded
    dataset_path = args.data_path if args.data_path else \
        os.path.join(os.path.split(os.path.abspath(__file__))[0], 'datasets')

    storing_loc = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'training_plots', args.dataset, f"comp_loss{args.closs}_weights{args.weights}")
    os.makedirs(storing_loc, exist_ok=True)

    # define the name of the folder where the results will be stored
    results_folder = os.path.join(args.output_path, "results", 'single_run', 'agra', args.dataset)
    os.makedirs(results_folder, exist_ok=True)
    output_file = open(os.path.join(results_folder, "test_performance.txt"), "a+")

    # define metric for the evaluation
    metric = define_eval_metric(args.dataset)

    # load dataset and training labels (mv or separately provided)
    train_dataset, valid_dataset, test_dataset, train_labels = load_train_data_for_agra(
        args.dataset, dataset_path, args.train_labels_path, args.num_valid_samples, args.finetuning_batch_size, encoding,
        finetuning=args.finetuning, finetuning_epochs=args.finetuning_epochs, metric=metric
    )

    # calculate num_classes
    if args.other is not None:
        num_classes = int(max(args.other, max(train_labels), max(valid_dataset.labels))) + 1
    else:
        num_classes = int(max(max(train_labels), max(valid_dataset.labels))) + 1

    # compute weights for comparison batch sampling
    agra_weights = compute_weights(train_labels) if args.weights == 'True' else None

    # initialize the model
    model = LogRegModelWithAGRA(
        num_classes=num_classes,
        agra_weights=agra_weights,
        other=args.other,
        agra_threshold=agra_threshold,
        adaptive_threshold=args.adaptive_threshold,
        storing_loc=storing_loc,
        include_bias=args.include_bias
    )

    history = model.fit(
        dataset_train=train_dataset,
        dataset_valid=valid_dataset,
        y_train=train_labels,
        comp_loss=args.closs,
        lr=lr,
        l2=weight_decay,
        batch_size=batch_size,
        num_epochs=num_epochs,

        modification=args.modification,
        metric=metric,
        verbose=True,

        patience=50            # 20 by default
    )

    # test the trained model on the test set
    test_metric_value = model.test(test_dataset, batch_size, metric)
    print(test_metric_value)

    if args.save is True:
        output_file.write("AGRA with LogReg \n")
        output_file.write("\t".join(["model", "logreg"]) + "\n")
        output_file.write("\t".join(["comp_loss", args.closs]) + "\n")
        output_file.write("\t".join(["weights", args.weights]) + "\n")
        output_file.write("\t".join(["alternative_label", str(args.other)]) + "\n")
        output_file.write("\t".join(["lr", str(lr)]) + "\n")
        output_file.write("\t".join(["batch_size", str(batch_size)]) + "\n")
        output_file.write("\t".join(["FT_batch_size", str(args.finetuning_batch_size)]) + "\n")
        output_file.write("\t".join(["agra_threshold", str(agra_threshold)]) + "\n")
        output_file.write("\t".join(["finetuning", str(args.finetuning)]) + "\n")
        output_file.write("\t".join(["finetuning_epochs", str(args.finetuning_epochs)]) + "\n")
        output_file.write("\t".join([metric, str(test_metric_value)]) + "\n" + "\n")
