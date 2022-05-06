import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from comet_ml import Optimizer
import torch
import gc
import models
from utils.Dataset import Dataset
from utils.Evaluator import Evaluator
from utils.HPShelper import conf_dict_generator
from utils.Logger import Logger
from utils.Params import Params
from utils.Trainer import Trainer
from utils.io import load_dataframe_csv, save_dataframe_csv

DATA_PATH = "./data"
LOG_PATH = Path("./saves")
CONFIG_PATH = Path("./conf_hp_search")
TABLE_PATH = "./tables"


def fit(experiment_, model_name, dataset_, log_directory, device_, skip_eval, plot_graph):
    d = conf_dict_generator[model_name](experiment_)
    d['skip_eval'] = skip_eval
    conf_dict = Params()
    conf_dict.update_dict(d)

    model_base = getattr(models, model_name)
    if 'contrast' in model_name:
        model_ = model_base(conf_dict, dataset_.num_users, dataset_.num_items, dataset_.num_keyphrases, device_)
    else:
        model_ = model_base(conf_dict, dataset_.num_users, dataset_.num_items, device_)

    evaluator = Evaluator(rec_atK=[5, 10, 15, 20, 50], explain_atK=[5, 10, 15, 20, 50])
    logger = Logger(log_directory)
    logger.info(conf_dict)
    logger.info(dataset_)

    trainer = Trainer(
        dataset=dataset_,
        model=model_,
        evaluator=evaluator,
        logger=logger,
        conf=conf_dict,
        experiment=experiment_,
        plot_graph=plot_graph  # plot the stats for embeddings
    )

    trainer.train()
    return (trainer.best_rec_score, trainer.best_uk_score,
            trainer.best_epoch, model_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='VAEcontrast_multilayer')
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--fold_name', type=str, default='fold0_valid')
    parser.add_argument('--top_items', type=int, help='used to indicate top labels for each item')
    parser.add_argument('--top_users', type=int, help='if cuting the matrix with top user numbers')
    parser.add_argument('--rating_threshold', type=float,
                        help='used to indicate user liked items for generating uk matrices')
    parser.add_argument('--plot_graph', action='store_true', help='Whether plotting the statistical graphs')
    parser.add_argument('--conf', type=str, default='VAEmultilayer_contrast1.config')
    parser.add_argument('--seed', type=int, default=201231)
    p = parser.parse_args()

    np.random.seed(p.seed)
    torch.random.manual_seed(p.seed)

    # where the training data files are stored
    data_dir = "{}/{}/{}/".format(DATA_PATH, p.data_name, p.fold_name)
    log_dir = LOG_PATH / p.data_name / p.model_name
    config_dir = CONFIG_PATH / p.data_name / p.conf
    table_dir = "{}/{}/{}/{}/".format(TABLE_PATH, p.data_name, p.model_name, 'hp_search')
    print('config_dir:', config_dir, 'table_dir:', table_dir)

    project_name = p.data_name + '-' + p.model_name + '-' + 'valid-grid'
    print(project_name)

    opt = Optimizer(config_dir)  # pass configuration file to cometml optimizer
    dataset = Dataset(data_dir=data_dir, top_keyphrases=p.top_items, rating_threshold=p.rating_threshold,
                      top_users=p.top_users)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # skip existing experiment results
    df = None
    try:
        df = load_dataframe_csv(table_dir, p.conf.split('.')[0] + '.csv')
    except:
        print('Not resuming any experiments')

    for experiment in opt.get_experiments(project_name=project_name):
        result_dict = conf_dict_generator[p.model_name](experiment)
        if df is None:
            df = pd.DataFrame(columns=result_dict.keys())

        # check experiment duplication
        if ('contrast' not in p.model_name and ((df['act'] == experiment.get_parameter("act")) &
                                                (df['anneal_cap'] == experiment.get_parameter("anneal_cap")) &
                                                (df['hidden_dim'] == experiment.get_parameter("hidden_dim")) &
                                                (df['dropout_ratio'] == experiment.get_parameter("dropout_ratio")) &
                                                (df['learning_rate'] == experiment.get_parameter("learning_rate")) &
                                                (df['weighted_recon'] == experiment.get_parameter("weighted_recon")) &
                                                (df['weight_decay'] == experiment.get_parameter(
                                                    "weight_decay"))).any()):
            print('Existing records: Skip')
            experiment.end()
            continue
        elif 'contrast' in p.model_name and ((df['act'] == experiment.get_parameter("act")) &
                                             (df['anneal_cap'] == experiment.get_parameter("anneal_cap")) &
                                             (df['hidden_dim'] == experiment.get_parameter("hidden_dim")) &
                                             (df['dropout_ratio'] == experiment.get_parameter("dropout_ratio")) &
                                             (df['learning_rate'] == experiment.get_parameter("learning_rate")) &
                                             (df['weight_decay'] == experiment.get_parameter("weight_decay")) &
                                             (df['kernel_method'] == experiment.get_parameter("kernel_method")) &
                                             (df['temperature_tau_u'] == experiment.get_parameter(
                                                 "temperature_tau_u")) &
                                             (df['temperature_tau_k'] == experiment.get_parameter(
                                                 "temperature_tau_k")) &
                                             (df['hp_contrastive_u'] == experiment.get_parameter("hp_contrastive_u")) &
                                             (df['pos_uk_num'] == experiment.get_parameter("pos_uk_num")) &
                                             (df['neg_uk_num'] == experiment.get_parameter("neg_uk_num")) &
                                             (df['pos_kk_num'] == experiment.get_parameter("pos_kk_num")) &
                                             (df['neg_kk_num'] == experiment.get_parameter("neg_kk_num")) &
                                             (df['weighted_recon'] == experiment.get_parameter("weighted_recon")) &
                                             (df['use_default_hp'] == experiment.get_parameter("use_default_hp"))).any():
            print('Existing records: Skip')
            experiment.end()
            continue

        try:
            rec_score, uk_score, epoch, model = fit(experiment, p.model_name, dataset, log_dir, device,
                                                skip_eval=False, plot_graph=p.plot_graph)
        except:
            continue

        # experiment.log_metric("best_mse", rec_score['RMSE'][0])
        experiment.log_metric("best_epoch", epoch)
        experiment.log_metrics({k: v[0] for k, v in rec_score.items()})
        if uk_score is not None:
            experiment.log_metrics({k: v[0] for k, v in uk_score.items()})

        experiment.log_others({
            "model_desc": p.model_name
        })

        result_dict['best_epoch'] = epoch

        for name in rec_score.keys():
            result_dict[name] = [round(rec_score[name][0], 4), round(rec_score[name][1], 4)]
        if uk_score is not None:
            for name in uk_score.keys():
                result_dict[name] = [round(uk_score[name][0], 4), round(uk_score[name][1], 4)]

        df = df.append(result_dict, ignore_index=True)

        save_dataframe_csv(df, table_dir, p.conf.split('.')[0])

        del model
        gc.collect()
        torch.cuda.empty_cache()
