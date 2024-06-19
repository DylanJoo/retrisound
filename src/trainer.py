import os
import torch
import torch.distributed as dist
from transformers import Trainer as hf_trainer
from transformers.utils import logging 
from transformers.trainer_utils import seed_worker 
from torch.utils.data import DataLoader, RandomSampler

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

class Trainer(hf_trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.curret_epoch = 0

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self.model.encoder.from_pretrained(resume_from_checkpoint)

    # def _get_train_sampler(self):
    #     if self.args.do_tas_doc: # as we dont have query actually.
    #         from .sampling.samplers import BinSampler
    #         return BinSampler(self.train_dataset, self.args.train_batch_size)
    #     else:
    #         return RandomSampler(self.train_dataset)
    #
    # def get_train_dataloader(self) -> DataLoader:
    #     """
    #     Returns the training [`~torch.utils.data.DataLoader`].
    #     Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler 
    #     (adapted to distributed training if necessary) otherwise.
    #     Subclass and override this method if you want to inject some custom behavior.
    #     """
    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")
    #
    #     train_dataset = self.train_dataset
    #     data_collator = self.data_collator
    #     data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
    #
    #     dataloader_params = {
    #         "batch_size": self._train_batch_size,
    #         "collate_fn": data_collator,
    #         "num_workers": self.args.dataloader_num_workers,
    #         "pin_memory": self.args.dataloader_pin_memory,
    #         "persistent_workers": self.args.dataloader_persistent_workers,
    #         "sampler": self._get_train_sampler()
    #     }
    #
    #     # add sampler here
    #     if not isinstance(train_dataset, torch.utils.data.IterableDataset):
    #         # dataloader_params["sampler"] = self._get_train_sampler()
    #         dataloader_params["drop_last"] = self.args.dataloader_drop_last
    #         dataloader_params["worker_init_fn"] = seed_worker
    #         dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
    #
    #     return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    # def train(
    #     self,
    #     resume_from_checkpoint: Optional[Union[str, bool]] = None,
    #     trial: Union["optuna.Trial", Dict[str, Any]] = None,
    #     ignore_keys_for_eval: Optional[List[str]] = None,
    #     **kwargs,
    # ):
    #     """
    #     Main training entry point.
    #
    #     Args:
    #         resume_from_checkpoint (`str` or `bool`, *optional*):
    #             If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
    #             `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
    #             of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
    #         trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
    #             The trial run or the hyperparameter dictionary for hyperparameter search.
    #         ignore_keys_for_eval (`List[str]`, *optional*)
    #             A list of keys in the output of your model (if it is a dictionary) that should be ignored when
    #             gathering predictions for evaluation during the training.
    #         kwargs (`Dict[str, Any]`, *optional*):
    #             Additional keyword arguments used to hide deprecated arguments
    #     """
    #     if resume_from_checkpoint is False:
    #         resume_from_checkpoint = None
    #
    #     # memory metrics - must set up as early as possible
    #     self._memory_tracker.start()
    #
    #     args = self.args
    #
    #     self.is_in_train = True
    #
    #     # Attach NEFTune hooks if necessary
    #     if self.neftune_noise_alpha is not None:
    #         self.model = self._activate_neftune(self.model)
    #
    #     # do_train is not a reliable argument, as it might not be set and .train() still called, so
    #     # the following is a workaround:
    #     if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
    #         self._move_model_to_device(self.model, args.device)
    #
	# ### [commented] ###
    #     # if "model_path" in kwargs:
    #     #     resume_from_checkpoint = kwargs.pop("model_path")
    #     #     warnings.warn(
    #     #         "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
    #     #         "instead.",
    #     #         FutureWarning,
    #     #     )
    #     # if len(kwargs) > 0:
    #     #     raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
    #     # This might change the seed so needs to run first.
    #     self._hp_search_setup(trial)
    #     self._train_batch_size = self.args.train_batch_size
    #
	# ### [commented] ###
    #     # Model re-init 
    #     # model_reloaded = False
    #     # if self.model_init is not None:
    #     #     # Seed must be set before instantiating the model when using model_init.
    #     #     enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
    #     #     self.model = self.call_model_init(trial)
    #     #     model_reloaded = True
    #     #     # Reinitializes optimizer and scheduler
    #     #     self.optimizer, self.lr_scheduler = None, None
    #
    #     # Load potential model checkpoint
    #     if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
    #         resume_from_checkpoint = get_last_checkpoint(args.output_dir)
    #         if resume_from_checkpoint is None:
    #             raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")
    #
    #     if resume_from_checkpoint is not None:
    #         if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
    #             self._load_from_checkpoint(resume_from_checkpoint)
    #         # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
    #         state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
    #         if state.train_batch_size is not None:
    #             self._train_batch_size = state.train_batch_size
    #
	# ### [commented] ###
    #     # If model was re-initialized, put it on the right device and update self.model_wrapped
    #     # if model_reloaded:
    #     #     if self.place_model_on_device:
    #     #         self._move_model_to_device(self.model, args.device)
    #     #     self.model_wrapped = self.model
    #
    #     inner_training_loop = find_executable_batch_size(
    #         self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
    #     )
	# ### [commented] ###
    #     # if args.push_to_hub:
    #     #     try:
    #     #         # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
    #     #         hf_hub_utils.disable_progress_bars()
    #     #         return inner_training_loop(
    #     #             args=args,
    #     #             resume_from_checkpoint=resume_from_checkpoint,
    #     #             trial=trial,
    #     #             ignore_keys_for_eval=ignore_keys_for_eval,
    #     #         )
    #     #     finally:
    #     #         hf_hub_utils.enable_progress_bars()
    #     # else:
    #
	# ### [TODO] See if we need to do the multi-stage training here as the flows might be different.
	# ### And we would like to update the new dataloader
	# return inner_training_loop(
	#     args=args,
	#     resume_from_checkpoint=resume_from_checkpoint,
	#     trial=trial,
	#     ignore_keys_for_eval=ignore_keys_for_eval,
	# )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        ## recast the data index type since dist can only share tensor
        if self.is_ddp:
            inputs['data_index'] = inputs['data_index'].long()
        else:
            inputs['data_index'] = inputs['data_index'].long().detach().cpu().numpy()

        # calculate d_tokens first
        if 'd_tokens' in inputs:
            model.eval()
            demb, dtokemb = self.model.encoder(inputs['d_tokens'], inputs['d_mask'], return_multi_vectors=True)
            demb, dtokemb = demb.detach().cpu().numpy(), dtokemb.detach().cpu().numpy()
            self.train_dataset.update_spans(
                    data_indices=inputs['data_index'].detach().cpu().numpy().tolist(),
                    batch_d_tokens=inputs['d_tokens'].detach().cpu().numpy(),
                    batch_d_masks=inputs['d_mask'].long().detach().cpu().numpy(),
                    batch_token_embeds=dtokemb,
                    batch_doc_embeds=demb
            )
            model.train()

        outputs = model(**inputs)

        if self.state.global_step % 50 == 0:
            logger.info('\n===== Examples starts =====\n')
            for i in range(1):
                print('index', inputs['data_index'][i].item())
                for key in inputs.keys():
                    if '_tokens' in key:
                        print(f"{key:<12}: ", self.tokenizer.decode(inputs[key][i][:30], skip_special_tokens=False), '...')
            logger.info('===== Example ends =====\n')

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.state.global_step % 10 == 0:
            logger.info(f"loss: {outputs['loss'].item()} | acc: {outputs['acc']}")
            self.log({"loss": outputs['loss'].item(), "acc": outputs['acc'].item()})
            if outputs.get('logs', None):
                for k, v in outputs['logs'].items():
                    self.log({f"{k}": v.item()})

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, **kwargs):
        """ Discard the original argument of `state_dict`, since it's from entire wrapped model.  """
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}. The model checkpoint is an encoder for huggingface, not a wrapping model.")

        model = self.model.get_encoder()
        self.model.encoder.save_pretrained(
            output_dir, state_dict=model.state_dict(), safe_serialization=self.args.save_safetensors
        )
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))

    def log(self, logs) -> None:
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)
