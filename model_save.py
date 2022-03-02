import pandas as pd
from comet_ml import Experiment
import numpy as np
import argparse
import torch
from pathlib import Path
import json
import models
from utils.Dataset import Dataset
from utils.Params import Params
from utils.Logger import Logger
from utils.Evaluator import Evaluator
from utils.Trainer import Trainer
from utils.HPShelper import conf_dict_generator
import gc
from utils.io import save_dataframe_csv

DATA_PATH = "./data"
LOG_PATH = Path("./saves") 
CONFIG_PATH = Path("./conf")
TABLE_PATH = "./tables"


def fit(experiment_, model_name, dataset_, log_directory, device_, skip_eval, plot_graph=False):
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
    parser.add_argument('--data_name', type=str, default='yelp_SIGIR')
    parser.add_argument('--data_dir', type=str, default='fold0')
    parser.add_argument('--top_items', type=int, help='used to indicate top labels for each item')
    parser.add_argument('--top_users', type=int, help='if cuting the matrix with top user numbers')
    parser.add_argument('--rating_threshold', type=float,
                        help='used to indicate user liked items for generating uk matrices, NOT USED NOW ')
    parser.add_argument('--log_dir', type=str, default='VAEcontrast',
                        help='used to name the csv file for the resulting model')
    parser.add_argument('--conf', type=str, default='VAEmultilayer_contrast1.config')
    parser.add_argument('--seed', type=int, default=201231)
    p = parser.parse_args()

    np.random.seed(p.seed)
    torch.random.manual_seed(p.seed)

    data_dir = "{}/{}/{}/".format(DATA_PATH, p.data_name, p.data_dir)
    log_dir = LOG_PATH / p.data_name / p.log_dir
    config_dir = CONFIG_PATH / p.data_name / p.conf
    print(config_dir)
    table_dir = "{}/{}/{}/{}/".format(TABLE_PATH, p.data_name, p.model_name, 'model_save')

    with open(config_dir) as f:
        conf = json.load(f)
    project_name = p.data_name + '-' + p.model_name + '-' + 'save'

    dataset = Dataset(data_dir=data_dir, top_keyphrases=p.top_items, rating_threshold=p.rating_threshold,
                      top_users=p.top_users)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    try:
        df = pd.read_csv(str(table_dir) + '.csv')
    except:
        df = pd.DataFrame(columns=['model', 'fold'])

    experiment = Experiment(project_name=project_name)
    experiment.log_parameters(conf)
    experiment.log_parameter("fold", data_dir)

    gc.collect()
    torch.cuda.empty_cache()

    rec_score, uk_score, epoch, model = fit(experiment, p.model_name, dataset, log_dir, device,
                                            skip_eval=True, plot_graph=False)
    experiment.log_metric("best_epoch", epoch)
    experiment.log_metrics({k: v[0] for k, v in rec_score.items()})
    if uk_score is not None:
        experiment.log_metrics({k: v[0] for k, v in uk_score.items()})

    experiment.log_others({
        "model_desc": p.log_dir,
        "fold": data_dir
    })

    result_dict = {
        'model': p.log_dir,
        'fold': p.data_dir,
        'best_epoch': epoch
    }

    # add in the recommendation score records
    for name in rec_score.keys():
        result_dict[name] = [round(rec_score[name][0], 4), round(rec_score[name][1], 4)]
    if uk_score is not None:
        for name in uk_score.keys():
            result_dict[name] = [round(uk_score[name][0], 4), round(uk_score[name][1], 4)]

    if torch.cuda.is_available:
        model = model.to(torch.device('cpu'))
    torch.save(model, str(log_dir)+'.pt')
    experiment.end()

    df = df.append(result_dict, ignore_index=True)
    save_dataframe_csv(df, str(log_dir), 'metrics')
