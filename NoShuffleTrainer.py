# from transformers import Trainer
# from torch.utils.data import DataLoader, SequentialSampler
# import torch
# import torch.nn as nn
# from typing import Any

# class NoShuffleTrainer(Trainer):

#     def _get_train_sampler(self, dataset=None):
#         return SequentialSampler(dataset if dataset is not None else self.train_dataset)
       
#     def compute_loss(
#         self,
#         model: nn.Module,
#         inputs: dict[str, torch.Tensor | Any],
#         return_outputs: bool = False,
#         num_items_in_batch: torch.Tensor | None = None,
#     ):

#         pc = getattr(self.accelerator, "parallelism_config", None)
#         if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled:
#             return self._deepspeed_sp_compute_loss(model, inputs, return_outputs, pc)

#         if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
#             labels = inputs.pop("labels")
#         else:
#             labels = None
#         if self.model_accepts_loss_kwargs:
#             kwargs = {}
#             if num_items_in_batch is not None:
#                 kwargs["num_items_in_batch"] = num_items_in_batch
#             inputs = {**inputs, **kwargs}
#         outputs = model(**inputs)

#         # User-defined compute_loss function
#         if self.compute_loss_func is not None:
#             if labels is None:
#                 logger.warning(
#                     "Trainer: `compute_loss_func` is defined but `labels=None`. "
#                     "Your custom loss function will still be called with labels=None. "
#                 )
#             loss = self.compute_loss_func(
#                 outputs,
#                 labels,
#                 num_items_in_batch=num_items_in_batch,
#             )
#         # Default HF loss handling (label smoothing) if no custom loss function
#         elif labels is not None:
#             unwrapped_model = self.accelerator.unwrap_model(model)
#             model_name = (
#                 unwrapped_model.base_model.model._get_name()
#                 if _is_peft_model(unwrapped_model)
#                 else unwrapped_model._get_name()
#             )
#             if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#                 loss = self.label_smoother(outputs, labels, shift_labels=True)
#             else:
#                 loss = self.label_smoother(outputs, labels)
#         else:
#             if isinstance(outputs, dict) and "loss" not in outputs:
#                 raise ValueError(
#                     "The model did not return a loss from the inputs, only the following keys: "
#                     f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
#                 )
#             # We don't use .loss here since the model may return tuples instead of ModelOutput.
#             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

#         if (
#             self.args.average_tokens_across_devices
#             and (self.model_accepts_loss_kwargs or self.compute_loss_func)
#             and num_items_in_batch is not None
#         ):
#             loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

#         return (loss, outputs) if return_outputs else loss
