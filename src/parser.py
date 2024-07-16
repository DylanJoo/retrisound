#!/usr/bin/env python
# coding=utf-8

import argparse
from transformers import HfArgumentParser

def main():

    from options import ModelOptions, DataOptions, TrainOptions
    parser = HfArgumentParser((ModelOptions, DataOptions, TrainOptions))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()
    if data_opt.config_file is not None:
        model_opt, data_opt, train_opt = parser.parse_yaml_file(data_opt.config_file)

    print(train_opt)

if __name__ == '__main__':
    main()
