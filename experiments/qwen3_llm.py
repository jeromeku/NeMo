# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed, fp16_mixed

from typing import Optional
from nemo import lightning as nl
from megatron.core.distributed import DistributedDataParallelConfig

import lightning.pytorch as pl
import nemo_run as run
import torch
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.qwen3 import qwen3_model, qwen3_trainer
from nemo.collections.multimodal.parts.stable_diffusion.pipeline import pipeline
from nemo.utils.exp_manager import TimingCallback

NAME = "qwen3moe_debug"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Qwen3 30B-A3B model configuration.
    This is a MoE (Mixture of Experts) model with 128 experts.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Qwen3 30B-A3B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=qwen3_30b_a3b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return qwen3_model(version=NAME)


def get_qwen3_strategy(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    expert_parallelism: int = 1,
    account_for_embedding_in_pipeline_split: bool = False,
    account_for_loss_in_pipeline_split: bool = False,
    
):
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        expert_model_parallel_size=expert_parallelism,
        expert_tensor_parallel_size=1,
        account_for_embedding_in_pipeline_split=account_for_embedding_in_pipeline_split,
        account_for_loss_in_pipeline_split=account_for_loss_in_pipeline_split,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=
            DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,  # Not supported for custom FSDP for now, need to be set to False if using FSDP
            data_parallel_sharding_strategy="optim_grads_params",  # For custom FSDP only
        ),
    )
    return strategy

def get_trainer( 
    strategy: nl.MegatronStrategy, num_nodes: int = 1,
    num_gpus_per_node: int = 1,
    max_steps: int = 1168251,
    precision: str = "bf16-mixed",
    accumulate_grad_batches: int = 1,
    limit_test_batches: int = 32,
    limit_val_batches: int = 32,
    log_every_n_steps: int = 10,
    val_check_interval: int = 2000,
    callbacks: Optional[list[run.Config[any]]] = None,):
    
    precision_plugin = None
    if precision == "16-mixed":
        precision_plugin = fp16_mixed()
    elif precision == "bf16-mixed":
        precision_plugin = bf16_mixed()

    trainer = nl.Trainer(
        accelerator="gpu",
        callbacks=callbacks,
        devices=num_gpus_per_node,
        accumulate_grad_batches=accumulate_grad_batches,
        limit_test_batches=limit_test_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=log_every_n_steps,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=None,
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=val_check_interval,
    )
    return trainer

@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    # General
    dir: Optional[str] = None,
    name: str = "default",
    # Trainer
    tensor_parallelism: int = 4,  # Default for 30B-A3B model
    pipeline_parallelism: int = 2,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    expert_parallelism: Optional[int] = 4,
    sequence_parallelism: bool = True,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 300000,
    precision: str = "bf16-mixed",
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float = 1.0,
    limit_test_batches: int = 32,
    limit_val_batches: int = 32,
    log_every_n_steps: int = 10,
    val_check_interval: int = 500,
    # Data
    global_batch_size=32,
    micro_batch_size=2,
    seq_length=4096,
    # Optimizer
    warmup_steps=500,
    constant_steps=0,
    min_lr=3e-5,
    max_lr=3e-4,
    # Training function
    fn=pretrain,
) -> run.Partial:
    """
    Create a pre-training recipe for Qwen3 30B-A3B model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.
    This model uses Mixture of Experts (MoE) architecture with 128 experts.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_type (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps.
        precision (str): Precision configuration, one of fp32, 16-mixed or bf16-mixed.
        accumulate_grad_batches (int): Number of steps per gradient accumulation.
        gradient_clip_val (float): Value for gradient clipping.
        limit_test_batches (int): Limit the number of test batches.
        limit_val_batches (int): Limit the number of validation batches.
        log_every_n_steps (int): Log every n steps.
        val_check_interval (int): Run validation every N steps.
        global_batch_size (int): Global batch size.
        micro_batch_size (int): Micro batch size.
        seq_length (int): Sequence length.
        warmup_steps (int): Number of warmup steps.
        constant_steps (int): Number of constant steps.
        min_lr (float): Minimum learning rate.
        max_lr (float): Maximum learning rate.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for pre-training.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory qwen3_30b_a3b
            $ nemo llm pretrain --factory "qwen3_30b_a3b(num_nodes=1, name='my_qwen3_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="qwen3_pretrain", num_nodes=1)
            >>> print(recipe)

    Note:
        This recipe uses a mock dataset, look for the finetune examples to see how to change the dataset.
    """
    recipe = run.Partial(
        fn,
        model=model(),
        trainer=qwen3_trainer(
            tensor_parallelism=tensor_parallelism,
            pipeline_parallelism=pipeline_parallelism,
            pipeline_parallelism_type=pipeline_parallelism_type,
            virtual_pipeline_parallelism=virtual_pipeline_parallelism,
            context_parallelism=context_parallelism,
            sequence_parallelism=sequence_parallelism,
            expert_parallelism=expert_parallelism,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            max_steps=max_steps,
            precision=precision,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_test_batches=limit_test_batches,
            limit_val_batches=limit_val_batches,
            log_every_n_steps=log_every_n_steps,
            val_check_interval=val_check_interval,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(
            MockDataModule,
            seq_length=seq_length,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            tokenizer=run.Config(AutoTokenizer, "Qwen/Qwen3-30B-A3B"),
        ),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(
            precision=precision,
            warmup_steps=warmup_steps,
            constant_steps=constant_steps,
            min_lr=min_lr,
            max_lr=max_lr,
            clip_grad=gradient_clip_val,
        ),
        resume=default_resume(),
    )
    recipe.model.config.recompute_granularity = "full"
    recipe.model.config.recompute_method = "uniform"
    recipe.model.config.recompute_num_layers = 1
    return recipe

from contextlib import contextmanager
from torch.testing._internal.distributed.fake_pg import FakeStore, _create_fake_pg
import torch.distributed as dist

def setup_dist(rank=0, world_size=2):
    store = FakeStore()
    dist.init_process_group(backend="fake", rank=rank, world_size=world_size, store=store)
    # _create_fake_pg(store, 0, 4, None)

def teardown_dist():
    dist.destroy_process_group()
    

@contextmanager
def fake_pg_context(rank=0, world_size=2):
    setup_dist(rank, world_size)
    try:
        yield
    finally:
        teardown_dist()

if __name__ == "__main__":
    from nemo.collections.llm.gpt.model.hf_auto_model_for_causal_lm import HFAutoModelForCausalLM
    from nemo.collections.llm.gpt.model.qwen3 import Qwen3ConfigMoeDebug
    from nemo.collections.llm.gpt.model.qwen3 import Qwen3Model
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    with fake_pg_context():
        parallel_state.initialize_model_parallel(create_gloo_process_groups=False)
        model_parallel_cuda_manual_seed(0)
        model_config: Qwen3ConfigMoeDebug = Qwen3ConfigMoeDebug() #qwen3_model(version=NAME).config
        print(model_config)
        tokenizer = AutoTokenizer("Qwen/Qwen3-30B-A3B")
        model = model_config.configure_model(tokenizer=tokenizer)
        # print(model_config)
        # model = Qwen3Model(model_config)
        print(model)
    # strategy: nl.MegatronStrategy = get_qwen3_strategy()
    # recipe_config = dict(dir = None,
    # name= "default",
    # # Trainer
    # tensor_parallelism= 1,  # Default for 30B-A3B model
    # pipeline= 1,
    # pipeline_parallelism_type = None,
    # virtual_pipeline_parallelism = None,
    # expert_parallelism = 1,
    # num_nodes = 1,
    # num_gpus_per_node = 8,
    # precision = "bf16-mixed",
    # # Data
    # global_batch_size=32,
    # micro_batch_size=2,
    # seq_length=4096,
    # # Optimizer
    # warmup_steps=500,
    # constant_steps=0,
    # min_lr=3e-5,
    # max_lr=3e-4,
    # )
    # trainer = get_trainer(strategy, max_steps=5)
    # data_module = MockDataModule(
    #         seq_length=1024,
    #         global_batch_size=4,
    #         micro_batch_size=1,
    #         tokenizer=AutoTokenizer("Qwen/Qwen3-30B-A3B"),
    #     )
    # optim = distributed_fused_adam_with_cosine_annealing(
    #         precision="bf16-mixed",
    #         warmup_steps=2,
    #         constant_steps=1,
    #         min_lr=3e-5,
    #         max_lr=3e-4,
    #         clip_grad=1)
        
    # pretrain(model=model, data=data_module, optim=optim, trainer=trainer, resume=default_resume())

# @run.cli.factory(target=finetune, name=NAME)
# def finetune_recipe(
#     dir: Optional[str] = None,
#     name: str = "default",
#     num_nodes: int = 1,
#     num_gpus_per_node: int = 8,
#     peft_scheme: Optional[str] = "lora",
#     packed_sequence: bool = False,
# ) -> run.Partial:
#     """
#     Create a fine-tuning recipe for Qwen3 30B-A3B model.

#     This function sets up a complete configuration for fine-tuning, including
#     model, trainer, data, logging, optimization, and resumption settings.
#     The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.
#     This model uses Mixture of Experts (MoE) architecture with 128 experts.

#     Args:
#         dir (Optional[str]): Directory for saving logs and checkpoints.
#         name (str): Name of the fine-tuning run.
#         num_nodes (int): Number of compute nodes to use.
#         num_gpus_per_node (int): umber of GPUs per node.
#         peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning.
#             Allowed values: 'lora'/'dora'/'none'/None.
#         packed_sequence (Optional[bool]): Packing multiple training sequences into one long sequence for training
#             efficiency. Default sequence length is 2048.

#     Returns:
#         run.Partial: Partial configuration for fine-tuning.

#     Examples:
#         CLI usage:
#             $ nemo llm finetune --factory qwen3_30b_a3b

#         Python API usage:
#             >>> recipe = finetune_recipe(name="qwen3_30b_a3b_finetune", num_nodes=2)
#             >>> print(recipe)

#     Note:
#         This recipe uses the SQuAD dataset for fine-tuning.
#     """
#     recipe = default_finetune_recipe(
#         model(), "Qwen/Qwen3-30B-A3B", dir, name, num_nodes, num_gpus_per_node, packed_sequence
#     )
#     if peft_scheme is None or peft_scheme.lower() == "none":
#         recipe.trainer.strategy.tensor_model_parallel_size = 4
#         recipe.trainer.strategy.expert_model_parallel_size = 4
#         recipe.trainer.strategy.expert_tensor_parallel_size = 1
#         recipe.trainer.strategy.pipeline_model_parallel_size = 2
#         recipe.trainer.strategy.sequence_parallel = True
#         recipe.optim.config.lr = 5e-6
#     elif peft_scheme.lower() in ["lora", "dora"]:
#         recipe.trainer.strategy.tensor_model_parallel_size = 4
#         recipe.trainer.strategy.expert_model_parallel_size = 4
#         recipe.trainer.strategy.expert_tensor_parallel_size = 1
#         recipe.trainer.strategy.sequence_parallel = True
#         recipe.peft = run.Config(PEFT_STR2CLS[peft_scheme.lower()])
#         recipe.peft.target_modules = ["linear_qkv", "linear_proj"]
#         recipe.optim.config.lr = 1e-4
#     else:
#         raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")
#     return recipe

