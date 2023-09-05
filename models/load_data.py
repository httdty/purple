# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 19:47
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : load_data.py
# @Software: PyCharm
import json

from tqdm import tqdm

from models.few_shot import RandomFewShotPrompter, PurpleFewShotPrompter
from models.utils import load_data, load_schema, load_processed_data


def data(args):
    load_data_strategies = {
        'default': load_data_default,
    }
    return load_data_strategies[args.data](args)


def load_data_default(args):
    dev = load_data(args.dev_file)
    instances = load_few_shot(dev, args)

    return instances

def load_ori_data(args):
    dev = load_data(args.ori_dev_file)
    return dev

def load_few_shot(dev, args):
    # Preprocess dev set
    with open(args.pred_skeleton, 'r') as f:
        pred_skeleton = json.load(f)
        if args.toy:
            pred_skeleton = pred_skeleton[::3]
    assert len(pred_skeleton) == len(dev), "pred_skeleton must match the dev number"
    with open(args.pred_domain, 'r') as f:
        pred_domain = json.load(f)
        if args.toy:
            pred_domain = pred_domain[::3]
    assert len(pred_domain) == len(dev), "pred_domain must match the dev number"
    for ins, skeleton, domain in zip(dev, pred_skeleton, pred_domain):
        # ins['sql_skeleton'] = skeleton[0]['generated_text']
        ins['sql_skeleton'] = skeleton
        ins['domain_linking'] = domain

    # Init prompter
    demonstrations = load_data(args.train_file)
    # prompter = None
    if args.prompt == 'random':
        prompter = RandomFewShotPrompter(
            demonstrations,
            max_length=args.prompt_length,
            model_name=args.model_name,
            db_dir=args.db_dir
        )
    elif args.prompt == 'purple':
        prompter = PurpleFewShotPrompter(
            demonstrations,
            max_length=args.prompt_length,
            model_name=args.model_name,
            enable_domain=args.enable_domain,
            enable_skeleton=args.enable_skeleton,
            enable_distinct=args.enable_distinct,
        )
    else:
        raise NotImplementedError(f"Do not supply such prompt strategy `{args.prompt}`")

    # Prompt gen
    instances = []
    print("Prepare prompt for the input...")
    for instance in tqdm(dev):
        prompt = prompter.get_prompt(instance)
        print(prompt)
        instances.append(
            {
                "prompt": prompt,
                "gold": instance["sql"],
                "db_id": instance["db_id"],
            }
        )
    return instances

