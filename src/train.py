import logging
logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger()

import json
import os
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

torch.set_num_threads(4)
# torch.cuda.set_device(hvd.local_rank())

def main():

    parser = HfArgumentParser(TrainerArgs)
    parser.add_argument('--data_path', type=str, help='path the training data, could be a folder')
    parser.add_argument('--valid_data_path', type=str, help='path the valid data, could be a folder')
    parser.add_argument('--test_data_path', type=str, help='path the test data, could be a folder')
    parser.add_argument('--validate_only', action='store_true', default=False, help='Skip training and run only validation. (default: False)')
    parser.add_argument('--working_dir', type=str, default='.', help='working dir, should be a dir with t5-experiments repo (default: .)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--show_valid_examples', type=int, default=0, help='how many valid examples to show during training (default: 0)')
    parser.add_argument('--subsample', type=str, default='', help='balanced or as in original data (default: '').')
    parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
    parser.add_argument('--target_seq_len', type=int, default=16, help='target sequnce length, should be set to ' 'max(len(target))+1 for EOS (default: 16).')
    parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

    parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
    # parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
    parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
    parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining', help='model class name to use (default: transformers:BertForPreTraining)')
    parser.add_argument('--backbone_cls', type=str, default=None, help='backbone class name to use for RMT')
    parser.add_argument('--backbone_trainable', action='store_true', default=False, help='make all model weights trainable, not only task-specific head.')
    parser.add_argument('--model_type', type=str, default='encoder-decoder', help='model type, encoder, encoder-decoder, decoder, affects preprocessing ' '(default: encoder-decoder)')

    parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
    parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
    parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
    parser.add_argument('--sum_loss', action='store_true', default=False, help='with this flag task loss from all segments is summed')
    # parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
    parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular', choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])

    parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
    args = parser.parse_args()
    # set current working dir
    # args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    # os.chdir(args.working_dir)

    # prepare_run(args, logger, logger_fmt)

    # [load models]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    from modeling.rmt import RMTEncoder
    from modeling.rife import Contriever
    if args.num_mem_tokens is not None:
        rmt_config = {
            'num_mem_tokens': args.num_mem_tokens,
            'max_n_segments': args.max_n_segments,
            'input_size': args.input_size,
            'bptt_depth': args.bptt_depth,
            'sum_loss': args.sum_loss,
            'tokenizer': tokenizer,
            # 'segment_ordering': args.segment_ordering,
        }
        rmt_cls = get_cls_by_name(args.model_cls)
    else:
        rmt_config = {}

    # logger.info(f'Wrapping in: {rmt_cls}')
    model = Contriever.from_pretrained('bert-base-uncased')
    d_encoder = deepcopy(model)
    ada_encoder = RMTEncoder(base_model=model, tokenizer=tokenizer, **rmt_config)

    from inbatch import InBatchInteraction
    inbatch = InBatchInteraction(
        model_opt, 
        q_encoder=ada_encoder,
        d_encoder=encoder,
        fixed_d_encoder=True
    )

    # [load datasets]
    from data.qampari import ContextQADataset, ContextQACollator
    train_dataset = ContextQADataset(path)
    eval_dataset = qampari(path)
    collator = ContextQACollator()

    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    if hvd.rank() == 0:
        logger.info(f'Using optimizer class: {optimizer_cls}')

    optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # # for encoder only classification
    # def keep_for_metrics_fn(batch, output):
    #     # select data from batch and model output that would be used to compute metrics
    #     data = {}
    #     if args.model_type == 'encoder':
    #         data['labels'] = batch['labels']
    #         data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
    #     elif args.model_type == 'encoder-decoder' and 'generation_outputs' in output:
    #         # logger.info(f'{output["generation_outputs"].shape}')
    #         data['labels'] = batch['labels']
    #         data['generation_outputs'] = output['generation_outputs']
    #     return data
    #
    # def metrics_fn(data):
    #     # compute metrics based on stored labels, predictions, ...
    #     metrics = {}
    #     y, p = None, None
    #     if args.model_type == 'encoder':
    #         y, p = data['labels'], data['predictions']
    #     elif args.model_type == 'encoder-decoder' and 'generation_outputs' in data:
    #         y = tokenizer.batch_decode(data['labels'], skip_special_tokens=True)
    #         p = tokenizer.batch_decode(data['generation_outputs'], skip_special_tokens=True)
    #         for _y, _p in zip(y, p):
    #             logger.info(f'{_y}: {labels_map.get(_y, 0)},  {_p}: {labels_map.get(_p, 0)}')
    #         # map to labels
    #         y = [labels_map.get(normalize_answer(_y), 0) for _y in y]
    #         p = [labels_map.get(normalize_answer(_p), 0) for _p in p]
    #     if y is not None and p is not None:
    #         # accuracy
    #         metrics['accuracy'] = accuracy_score(y, p)
    #         # f1, precision, recall, mcc
    #         metrics['f1'] = f1_score(y, p)
    #         metrics['precision'] = precision_score(y, p)
    #         metrics['recall'] = recall_score(y, p)
    #     return metrics

    ## booydar
    batch_metrics_fn = lambda _, y: {key: y[key] for key in y.keys() if (('loss' in key) or ('!log' in key))}
    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, train_sampler,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      ##
                      batch_metrics_fn=batch_metrics_fn,
                      generate_kwargs=generate_kwargs if args.use_generate_on_valid else {})

    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
        hvd.barrier()
        # run validation after training

        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best.pth')
            if hvd.rank() == 0:
                logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        if args.valid_data_path:
            if hvd.rank() == 0:
                logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
        if args.test_data_path:
            if hvd.rank() == 0:
                logger.info('Runnning validation on test data:')
            trainer.validate(test_dataloader, split='test', write_tb=True)
    else:
        # run validation, do not write to tensorboard
        if hvd.rank() == 0:
            logger.info('Running validation on train set:')
        trainer.validate(train_dataloader, write_tb=False)
        if args.valid_data_path:
            if hvd.rank() == 0:
                logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
        if args.test_data_path:
            if hvd.rank() == 0:
                logger.info('Running validation on test data:')
            trainer.validate(test_dataloader, split='test', write_tb=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)
