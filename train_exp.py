import contextlib
import logging
import math
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import datasets
import torch
import transformers
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, InitProcessGroupKwargs, set_seed
from arguments import DatasetArguments, ModelArguments, MoshiTrainingArguments
from configuration_moshi import MoshiConfig
from data import MoshiDataCollator
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from modeling_moshi import MoshiForConditionalGeneration
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoTokenizer, HfArgumentParser
from transformers.optimization import get_scheduler
from utils import get_last_checkpoint, log_metric, rotate_checkpoints


logger = logging.getLogger(__name__)


def save_with_accelerate(accelerator, model, tokenizer, output_dir):
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.deepspeed_plugin.zero3_save_16bit_model = True
        accelerator.deepspeed_plugin.stage3_gather_16bit_weights_on_model_save = True

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    # check if state_dict is a dict has empty tensor
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        tokenizer.save_pretrained(output_dir)


def chunk_example(batch, max_length=2048):
    """
    batched=True 모드에서, batch는
    {
      'moshi_text_tokens': [[...], [...], ...],      # 여러 샘플
      'moshi_audio_tokens': [tensor(...), tensor(...), ...],
      'user_audio_tokens':  [tensor(...), tensor(...), ...],
    }
    형태가 됩니다.

    최종 반환은
    {
      'moshi_text_tokens': [...여러chunk...],
      'moshi_audio_tokens': [...여러chunk...],
      'user_audio_tokens':  [...여러chunk...],
    }
    형태가 되어야 합니다.
    """

    # 최종으로 쌓아 둘 리스트
    chunked_text_list = []
    chunked_moshi_audio_list = []
    chunked_user_audio_list = []
    chunked_user_text_list = []
    # batch 안에 있는 각 샘플들을 순회
    for text, moshi_audio, user_audio, user_text in zip(
        batch["moshi_text_tokens"],
        batch["moshi_audio_tokens"],
        batch["user_audio_tokens"],
        batch.get("user_text_tokens", None),
    ):
        # text: 길이가 (예) 7732인 1D 리스트 or 배열
        # moshi_audio: shape [8, 7732] (list of list or numpy array 등)
        # user_audio:  shape [8, 7732]

        # 원하는 max_length 단위로 잘라서 chunk들을 만들기
        total_length = len(text)
        for start in range(0, total_length, max_length):
            end = min(start + max_length, total_length)

            if isinstance(text, list):
                text = torch.tensor(text)
                user_text = torch.tensor(user_text)
                moshi_audio = torch.tensor(moshi_audio)
                user_audio = torch.tensor(user_audio)

            chunked_text_list.append(text[start:end])
            chunked_moshi_audio_list.append(moshi_audio[:, start:end])
            chunked_user_audio_list.append(user_audio[:, start:end])
            chunked_user_text_list.append(user_text[start:end])

    # 이제 각 key마다 리스트들을 합쳐서 반환
    return {
        "moshi_text_tokens": chunked_text_list,       # 길이가 batch_size * 몇개
        "moshi_audio_tokens": chunked_moshi_audio_list,
        "user_audio_tokens": chunked_user_audio_list,
        "user_text_tokens": chunked_user_text_list,
    }


def swap_moshi_user(examples):
    new_moshi_text_tokens = []
    new_user_text_tokens = []
    new_moshi_audio_tokens = []
    new_user_audio_tokens = []

    # examples 는 batched=True 일 경우, 
    # examples["moshi_text_tokens"], examples["user_text_tokens"] 등은 리스트(또는 array) 형태.
    for m_text, u_text, m_audio, u_audio in zip(
        examples["moshi_text_tokens"],
        examples["user_text_tokens"],
        examples["moshi_audio_tokens"],
        examples["user_audio_tokens"],
    ):
        # 1) 원본 샘플 (그대로)
        new_moshi_text_tokens.append(m_text)
        new_user_text_tokens.append(u_text)
        new_moshi_audio_tokens.append(m_audio)
        new_user_audio_tokens.append(u_audio)

        # 2) moshi 와 user 를 스왑한 샘플
        new_moshi_text_tokens.append(u_text)
        new_user_text_tokens.append(m_text)
        new_moshi_audio_tokens.append(u_audio)
        new_user_audio_tokens.append(m_audio)

    return {
        "moshi_text_tokens": new_moshi_text_tokens,
        "user_text_tokens": new_user_text_tokens,
        "moshi_audio_tokens": new_moshi_audio_tokens,
        "user_audio_tokens": new_user_audio_tokens,
    }


def main():
    parser = HfArgumentParser((DatasetArguments, ModelArguments, MoshiTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        dataset_args, model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        dataset_args, model_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        torch_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        torch_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        torch_dtype = torch.float32

    ####### A. Preparation
    kwargs_handlers = [
        InitProcessGroupKwargs(timeout=timedelta(minutes=120)),
        DistributedDataParallelKwargs(find_unused_parameters=False),
    ]

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        kwargs_handlers=kwargs_handlers,
    )

    accelerator.init_trackers(
        project_name=dataset_args.wandb_project,
        config={
            "learning_rate": training_args.learning_rate,
            "model_name": model_args.model_name,
            "dataset_name": dataset_args.dataset_name,
            "num_train_epochs": training_args.num_train_epochs,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "global_batch_size": training_args.per_device_train_batch_size * accelerator.num_processes,
            "mixed_precision": mixed_precision,
            "weight_decay": training_args.weight_decay,
            "adam_beta1": training_args.adam_beta1,
            "adam_beta2": training_args.adam_beta2,
        },
        init_kwargs={"wandb": {"name": dataset_args.wandb_run_name}} if dataset_args.wandb_run_name else {},
    )

    # Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARN)

    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # (1) 원본 dataset 로드
    dataset = load_dataset(dataset_args.dataset_name, split="train")
    logger.info(f"dataset length: {len(dataset)}")

    # (1-1) moshi <-> user 스왑 2배 augmentation
    if training_args.swap_moshi_user:
        dataset = dataset.map(
            swap_moshi_user,
            batched=True,
            num_proc=os.cpu_count() // 2,
            desc="swap_moshi_user",
        )
    logger.info(f"dataset length after swap_moshi_user: {len(dataset)}")

    # (1-2) chunk_example 을 활용해 최대 2048 길이로 잘라서 augmentation
    dataset = dataset.map(
        chunk_example,
        batched=True,
        num_proc=os.cpu_count()//2,
        fn_kwargs={'max_length': dataset_args.max_length},
        desc=f"chunk_example by {dataset_args.max_length}",
    )
    logger.info(f"dataset length after chunk_example: {len(dataset)}")

    config = MoshiConfig.from_pretrained(
        model_args.model_name,
        cache_dir=model_args.cache_dir,
        token=dataset_args.token,
        trust_remote_code=dataset_args.trust_remote_code,
    )

    model = MoshiForConditionalGeneration.from_pretrained(
        model_args.model_name,
        config=config,
        torch_dtype=torch_dtype,
        cache_dir=model_args.cache_dir,
        token=dataset_args.token,
        trust_remote_code=dataset_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
    )

    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    # Test all gather - used for warmout and avoiding timeout
    logger.debug(str(accelerator.process_index), main_process_only=False, in_order=True)
    test_tensor = torch.tensor([accelerator.process_index], device=accelerator.device)
    gathered_tensor = accelerator.gather(test_tensor)
    print("gathered_tensor", gathered_tensor)
    accelerator.wait_for_everyone()

    # load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name,
        cache_dir=model_args.cache_dir,
        token=dataset_args.token,
        trust_remote_code=dataset_args.trust_remote_code,
    )

    # (2) 필요한 ID 값
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name,
        cache_dir=model_args.cache_dir,
        token=dataset_args.token,
        trust_remote_code=dataset_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
    )

    # fix token embedding size if not matched
    # do not worry, model.config.vocab_size will also be changed accordingly
    if model.decoder.config.vocab_size != tokenizer.vocab_size:
        model.resize_token_embeddings(tokenizer.vocab_size)
        model.depth_decoder.resize_token_embeddings(tokenizer.vocab_size)

    if model_args.use_fast_tokenizer:
        logger.warning(
            "Disabling fast tokenizer warning: https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3231-L3235"
        )
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    text_pad_token_id = model.config.vocab_size
    audio_pad_token_id = model.config.audio_vocab_size
    text_align_pad_token_id = tokenizer.encode("<pad>")[0]

    # text loss reweigting
    text_loss_weight = torch.ones(model.config.vocab_size, dtype=torch_dtype).to(accelerator.device)
    text_loss_weight[text_align_pad_token_id] = training_args.text_align_pad_loss_weight

    # (3) DataCollator 생성 (모델과 pad_token_id 전달)
    collator = MoshiDataCollator(
        text_pad_token_id=text_pad_token_id,
        audio_pad_token_id=audio_pad_token_id,
        text_vocab_size=model.config.vocab_size,
        audio_vocab_size=model.config.audio_vocab_size,
        text_align_pad_token_id=text_align_pad_token_id,
    )

    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)

    if training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(dataset) // (train_batch_size * gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        logger.info("max_steps is given, it will override any value given in num_train_epochs")
        total_train_steps = int(training_args.max_steps)
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_epochs = sys.maxsize
        steps_per_epoch = total_train_steps

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not (n.startswith("decoder.lm_head") or n.startswith("decoder.model.embed_tokens"))
            ],
            "lr": training_args.learning_rate,
        },
        {
            "params": model.decoder.lm_head.parameters(),
            "lr": training_args.learning_rate * training_args.text_lr_multiplier,
        },
        {
            "params": model.decoder.model.embed_tokens.parameters(),
            "lr": training_args.learning_rate * training_args.text_lr_multiplier,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        weight_decay=training_args.weight_decay,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    model, optimizer = accelerator.prepare(model, optimizer)

    num_examples = total_train_steps * train_batch_size * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info("  Instantaneous batch size per device =" f" {per_device_train_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")

    # ======================== Training ================================
    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    continue_training = True
    epochs_trained = 0
    cur_step = 0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if accelerator.is_main_process:
        if training_args.push_to_hub:
            api = HfApi(token=training_args.hub_token)

            # Create repo (repo_name from args or inferred)
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Now save everything to be able to create a single processor later
    # make sure all processes wait until data is saved
    # only the main process saves them
    if accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        feature_extractor.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        # Find num steps and epoch from saved state string pattern
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")

        steps_trained_progress_bar.update(cur_step)

        for epoch in range(0, epochs_trained):
            with accelerator.local_main_process_first():
                dataset = dataset.shuffle(training_args.seed)

        if training_args.max_steps < 0:
            # we know exactly the number of steps per epoch, so can skip through the required number of batches
            resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
        else:
            # Currently we don't know how many steps we've taken in the current epoch
            # So we just shuffle the dataset one extra time and start from a fresh epoch
            # This is "good enough" for our purposes but not fully correct
            resume_step = None
            with accelerator.local_main_process_first():
                dataset = dataset.shuffle(training_args.seed)
    else:
        resume_step = None

    def train_step(
        batch,
        accelerator,
        num_items_text_in_batch,
        num_items_audio_semantic_in_batch,
        num_items_audio_accoustic_in_batch,
        gradient_accumulation_steps,
    ):
        # giving num_items_in_batch enables loss reduction by mean (https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L24)
        # also gives option to set loss reduction to depth decoder manually
        outputs = model(
            **batch,
            decoder_weight=text_loss_weight,
            decoder_num_items_in_batch=1,
            depth_decoder_loss_reduction="sum",
        )

        text_loss = outputs.loss
        audio_semantic_loss = outputs.depth_semantic_loss
        audio_accoustic_loss = outputs.depth_accoustic_loss

        text_loss = (text_loss * gradient_accumulation_steps * accelerator.num_processes) / num_items_text_in_batch
        audio_semantic_loss = (
            audio_semantic_loss * gradient_accumulation_steps * accelerator.num_processes
        ) / num_items_audio_semantic_in_batch
        audio_accoustic_loss = (
            audio_accoustic_loss * gradient_accumulation_steps * accelerator.num_processes
        ) / num_items_audio_accoustic_in_batch

        # make semantic audio codebook loss alpha_semantic times more weighted than remaining codebooks
        # (assume num of items for all codebooks are the same)
        alpha_semantic = training_args.alpha_semantic
        audio_loss = alpha_semantic * audio_semantic_loss + (model.config.num_codebooks - 1) * audio_accoustic_loss
        audio_loss /= alpha_semantic + (model.config.num_codebooks - 1)

        loss = text_loss + audio_loss

        metrics = {"loss": loss, "loss_text": text_loss, "loss_audio": audio_loss}

        return loss, metrics

    model.train()

    total_batched_samples = resume_step if resume_step is not None else 0

    for epoch in range(epochs_trained, num_epochs):
        with accelerator.local_main_process_first():
            dataset = dataset.shuffle(training_args.seed)

        # (4) DataLoader 생성
        train_dataloader = DataLoader(
            dataset,
            batch_size=per_device_train_batch_size,
            collate_fn=collator,
            shuffle=True,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            # Skip the first N batches in the dataloader when resuming from a checkpoint
            logger.info(f"  Skip first {resume_step} batches")
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None
            accelerator.wait_for_everyone()

        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        train_iterator = iter(train_dataloader)
        num_steps_in_epoch = len(train_dataloader)
        remainder = num_steps_in_epoch % gradient_accumulation_steps
        remainder = remainder if remainder != 0 else gradient_accumulation_steps
        total_updates = math.ceil(num_steps_in_epoch / gradient_accumulation_steps)

        update_step = -1
        for _ in range(total_updates):
            update_step += 1

            # preload the total batch per step
            batch_samples = []
            num_batches_in_step = gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
            for _ in range(num_batches_in_step):
                batch_samples += [next(train_iterator)]

            # get num items in batch - if different than BOS and than -100
            num_items_text_in_batch = sum(
                [
                    # -2 because of shifting (https://github.com/huggingface/transformers/blob/main/src/transformers/models/moshi/modeling_moshi.py#L1887) + huggingface implementation of CausalLMLoss (https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L32)
                    (batch["text_labels"].ne(-100)).sum() - 2
                    for batch in batch_samples
                ]
            )
            num_items_text_in_batch = accelerator.gather(num_items_text_in_batch).sum().item()
            num_items_audio_semantic_in_batch = sum(
                [
                    # -1 because of shifting
                    (batch["audio_labels"][:, :1].ne(audio_pad_token_id)).sum() - 1
                    for batch in batch_samples
                ]
            )
            num_items_audio_semantic_in_batch = accelerator.gather(num_items_audio_semantic_in_batch).sum().item()
            num_items_audio_accoustic_in_batch = sum(
                [
                    # -1 because of shifting
                    (batch["audio_labels"][:, 1:].ne(audio_pad_token_id)).sum() - 1
                    for batch in batch_samples
                ]
            )
            num_items_audio_accoustic_in_batch = accelerator.gather(num_items_audio_accoustic_in_batch).sum().item()

            for i, batch in enumerate(batch_samples):
                total_batched_samples += 1
                ctx = (
                    model.no_sync
                    if (i < len(batch_samples) - 1 and accelerator.num_processes > 1)
                    else contextlib.nullcontext
                )

                with ctx():
                    loss, train_metric = train_step(
                        batch,
                        accelerator,
                        num_items_text_in_batch,
                        num_items_audio_semantic_in_batch,
                        num_items_audio_accoustic_in_batch,
                        gradient_accumulation_steps,
                    )
                    accelerator.backward(loss)

            grad_norm = accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # The accelerator has performed an optimization step behind the scenes
            steps_trained_progress_bar.update(1)
            cur_step += 1

            if cur_step % training_args.logging_steps == 0:
                steps_trained_progress_bar.write(
                    f"Step... ({cur_step} / {total_train_steps} | Loss: {train_metric['loss']:.3f} | Text Loss: "
                    f" {train_metric['loss_text']:.3f} | Audio Loss: {train_metric['loss_audio']:.3f}"
                )
                train_metric["grad_norm"] = (
                    grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                )
                log_metric(
                    accelerator,
                    metrics=train_metric,
                    learning_rate=training_args.learning_rate,
                    train_time=train_time + time.time() - train_start,
                    step=cur_step,
                    epoch=epoch,
                    prefix="train",
                )

            # save checkpoint and weights after each save_steps and at the end of training
            if (cur_step % training_args.save_steps == 0) or cur_step == total_train_steps:
                intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                # https://github.com/huggingface/transformers/issues/27293#issuecomment-1872560074
                accelerator.save_state(output_dir=intermediate_dir, safe_serialization=False)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    rotate_checkpoints(
                        training_args.save_total_limit, output_dir=training_args.output_dir, logger=logger
                    )

                    if cur_step == total_train_steps:
                        # un-wrap student model for save
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(training_args.output_dir)

                    if training_args.push_to_hub:
                        api.upload_folder(
                            repo_id=repo_id,
                            folder_path=training_args.output_dir,
                            commit_message=f"Saving train state of step {cur_step}",
                            run_as_future=True,
                        )
                accelerator.wait_for_everyone()

            # break condition
            if cur_step == total_train_steps:
                continue_training = False
                break

        if not continue_training:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main()
