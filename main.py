from random import shuffle
from typing import Annotated

import torch
import typer
from loguru import logger

from config.logging import setUpLogger
import constant.enums as cli_enum
from data.loader import load_data
from data.utils import merge_and_split, normalization_wrt_subject, zscore_wrt_subject
from model.rpacq import RPACQ
from train.train import train
from utils.metric import Metric
from utils.random_seed import setup_seed

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    dataset: Annotated[
        cli_enum.DatasetName, typer.Option(help='dataset name')
    ] = cli_enum.DatasetName.SEED,
    dataset_path: Annotated[
        str, typer.Option(help='path to the dataset')
    ] = '../data/SEED',
    cache_dir: Annotated[str | None, typer.Option(
        help='cache directory for loaded dataset (disabled if not set)')] = "./cache",
    device: Annotated[str, typer.Option(
        help='device to run the model on')] = 'cuda',
    label_type: Annotated[
        str, typer.Option(help='type of label to use for DEAP (valence, arousal)')
    ] = 'valence',
    task_type: Annotated[
        cli_enum.TaskTypeName,
        typer.Option(help='type of experimental task (dep, indep)'),
    ] = cli_enum.TaskTypeName.SUBJECT_INDEPENDENT,
    split_type: Annotated[
        cli_enum.SplitTypeName,
        typer.Option(help='type of data split (kfold, loso, ttv)'),
    ] = cli_enum.SplitTypeName.LEAVE_ONE_SUBJECT_OUT,
    split_ratio: Annotated[float, typer.Option(help='ratio for train data size')] = 0.6,
    batch_size: Annotated[int, typer.Option(help='batch size for training')] = 32,
    epochs: Annotated[int, typer.Option(help='number of epochs for training')] = 1000,
    data_random: Annotated[bool, typer.Option(help='whether to shuffle the data')] = False,
    only_one_experiment: Annotated[bool, typer.Option(help='run only one experiment (for debugging)')] = False,
    run_session: Annotated[str, typer.Option(help='which session to run (e.g., "0", "all")')] = "0",
    random_seed: Annotated[int | None, typer.Option(help='random seed (None = random)')] = 42,
    learning_rate: Annotated[float, typer.Option(help='learning rate')] = 0.001,
    early_stop_patience: Annotated[int, typer.Option(help='early stop after N epochs without improvement (0 = disabled)')] = 0,
    use_gcn: Annotated[bool, typer.Option(help='use GCN feature extractor (False = MLP)')] = True,
    pool_capacity: Annotated[int, typer.Option(help='target prototype FIFO pool capacity')] = 256,
    xconf_ramp_epochs: Annotated[int, typer.Option(help='epochs over which xconf lam3 ramps from 0 to 0.2')] = 200,
    norm: Annotated[str, typer.Option(help='normalization type: minmax or zscore')] = 'minmax',
    level: Annotated[cli_enum.LevelName, typer.Option('-l', help='logging level')] = cli_enum.LevelName.INFO,
):
    """RPACQ: Reliability-weighted Prototype Alignment with Cosine Quantization."""

    setUpLogger(level=level)
    setup_seed(random_seed)

    logger.info('CUDA Available: {}', torch.cuda.is_available())
    logger.info('Device Count: {}', torch.cuda.device_count())
    if torch.cuda.is_available():
        logger.info('GPU Name: {}', torch.cuda.get_device_name(0))

    logger.info(
        'Launching....\ndataset: {}\ndataset_path: {}\ncache_dir: {}'
        '\ndevice: {}\nlogging level: {}\ntask type: {}'
        '\nsplit type: {}\nbatch_size: {}\nepochs: {}'
        '\ndata_random: {}\nrandom_seed: {}'
        '\nonly_one_experiment: {}\nrun_session: {}'
        '\nlearning_rate: {}\nearly_stop_patience: {}'
        '\nuse_gcn: {}\npool_capacity: {}\nsinkhorn_warmup_epochs: {}'
        '\nlabel_type: {}',
        dataset, dataset_path, cache_dir,
        device, level, task_type,
        split_type, batch_size, epochs,
        data_random, random_seed,
        only_one_experiment, run_session,
        learning_rate, early_stop_patience,
        use_gcn, pool_capacity,
        label_type,
    )

    data, labels, num_subjects, num_electrodes, num_features, num_classes = load_data(
        dataset_name=dataset,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        label_type=label_type,
    )
    logger.info(
        'num_electrodes: {} num_features: {} num_classes: {}',
        num_electrodes, num_features, num_classes,
    )

    if norm == 'zscore':
        zscore_wrt_subject(data)
    else:
        normalization_wrt_subject(data)

    num_sessions = len(labels)
    subject_ids = list(range(num_subjects))
    if data_random:
        shuffle(subject_ids)

    metric = Metric(num_subjects, num_sessions)

    logger.debug('num_sessions {} num_subjects {}', num_sessions, num_subjects)

    for session_id in range(num_sessions):
        if run_session != "all" and session_id != int(run_session):
            continue
        for subject_id in subject_ids:
            setup_seed(random_seed)

            train_data, train_labels, test_data, test_labels = merge_and_split(
                data, labels, task_type, session_id, subject_id, split_ratio, data_random,
            )

            model = RPACQ(
                num_electrodes=num_electrodes,
                in_features=num_features,
                num_classes=num_classes,
                use_gcn=use_gcn,
                max_iter=epochs,
                pool_capacity=pool_capacity,
            ).to(device)

            train(
                model, metric, train_data, train_labels, test_data, test_labels,
                batch_size, num_classes, device, epochs, subject_id,
                session_id, learning_rate, early_stop_patience,
                xconf_ramp_epochs=xconf_ramp_epochs,
            )

            logger.info(
                "\n--------------> Finished training subject {} session {} acc {:<.4f}",
                subject_id, session_id, metric.accuracy[subject_id, session_id],
            )

            if only_one_experiment:
                break
        if only_one_experiment:
            break

    logger.info('\n-----------> Finished training for all subjects!!!!')
    one_mean, one_std = metric.one_best_session_mean_acc()
    logger.info('One best session acc: mean {:<.4f} std {:<.4f}\n', one_mean, one_std)


if __name__ == '__main__':
    app()
