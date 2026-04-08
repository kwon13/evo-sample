
import os
import uuid
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from ray.experimental.tqdm_ray import tqdm
import pandas as pd
from collections import defaultdict
from typing import Any, Dict, List, Optional
import time
import subprocess
import copy
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER
from ..utils.logger import Tracker
from ..utils.py_functional import timer, extract_tag
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from .ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from rewards.code_reward import evaluate_single_code, run_one_case_subprocess
from rewards.extract_starter_code import starter_code_from_solution_text

from data.prompts import (
                            construct_questioner_msgs,
                            construct_verifier_msgs,
                            construct_solver_msgs
                        )


def _create_raw_prompt_ids(tokenizer, prompts: List[str]) -> np.ndarray:
    """Create raw_prompt_ids (no padding) for vLLM generation."""
    raw_ids_list = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        raw_ids_list.append(ids)
    return np.array(raw_ids_list, dtype=object)


def pad_batch_to_length(batch: DataProto, target_seq_len: int, target_resp_len: int, pad_token_id: int = 0) -> DataProto:
    """
    Pad tensors in batch to target lengths using LEFT padding.
    Separates tensors into two groups:
    - Sequence length group: input_ids, attention_mask, position_ids
    - Response length group: responses, old_log_probs, ref_log_probs, token_level_scores, 
                             token_level_rewards, response_mask, values
    
    Args:
        batch: DataProto to pad
        target_seq_len: Target sequence length for input_ids, attention_mask, position_ids
        target_resp_len: Target response length for responses and related tensors
        pad_token_id: Token ID to use for padding input_ids
        
    Returns:
        Padded DataProto (modified in place)
    """
    if batch.batch is None:
        return batch
    
    # Tensors that should match sequence length
    SEQ_LEN_KEYS = {'input_ids', 'attention_mask', 'position_ids'}
    # Tensors that should match response length  
    RESP_LEN_KEYS = {'responses', 'old_log_probs', 'ref_log_probs', 'token_level_scores', 
                     'token_level_rewards', 'response_mask', 'values', 'prompts'}
    
    for key in list(batch.batch.keys()):
        tensor = batch.batch[key]
        if tensor is None or len(tensor.shape) < 2:
            continue
        
        current_len = tensor.shape[1]
        batch_size = tensor.shape[0]
        
        # Determine target length based on key
        if key in SEQ_LEN_KEYS:
            target_len = target_seq_len
        elif key in RESP_LEN_KEYS:
            target_len = target_resp_len
        else:
            # Unknown key - skip
            continue
        
        if current_len >= target_len:
            continue
        
        pad_size = target_len - current_len
        
        # Create padding tensor based on key type
        if key == 'input_ids':
            padding = torch.full((batch_size, pad_size), pad_token_id, dtype=tensor.dtype, device=tensor.device)
        elif key == 'attention_mask':
            padding = torch.zeros((batch_size, pad_size), dtype=tensor.dtype, device=tensor.device)
        elif key == 'position_ids':
            padding = torch.zeros((batch_size, pad_size), dtype=tensor.dtype, device=tensor.device)
        else:
            # For other tensors, use 0
            padding = torch.zeros((batch_size, pad_size), dtype=tensor.dtype, device=tensor.device)
        
        # LEFT padding: [padding | original]
        batch.batch[key] = torch.cat([padding, tensor], dim=1)
    
    return batch





class EvolvingDataset(torch.utils.data.Dataset):
    """A simple dataset that wraps a list of records for DataLoader."""
    
    def __init__(self, records: List[Dict], tokenizer, max_prompt_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        row = copy.deepcopy(self.records[idx])
        
        # Build prompt from problem_description
        prompt = row.get('problem_description', '')
        
        # Tokenize
        model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.max_prompt_length)
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        
        # Pad to max_prompt_length
        pad_len = self.max_prompt_length - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype), input_ids])
            attention_mask = torch.cat([torch.zeros(pad_len, dtype=attention_mask.dtype), attention_mask])
        
        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
        
        row['input_ids'] = input_ids
        row['attention_mask'] = attention_mask
        row['position_ids'] = position_ids
        row['raw_prompt_ids'] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row['ground_truth'] = row.get('completion', '')
        
        return row


def evolving_collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for evolving dataset."""
    batch = {}
    keys = features[0].keys()
    
    for key in keys:
        values = [f[key] for f in features]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        elif isinstance(values[0], (list, np.ndarray)):
            batch[key] = np.array(values, dtype=object)
        else:
            batch[key] = np.array(values, dtype=object)
    
    return batch


class CodeRayPPOTrainer(RayPPOTrainer):
    """
    Evolutionary Trainer that cycles data across epochs.
    Each epoch, transformed problems become input for the next epoch.
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        processor,
        train_dataloader,
        val_dataloader,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        reward_fn,
        solver_reward_fn=None,
        val_reward_fn=None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        # Store solver reward function
        self.solver_reward_fn = solver_reward_fn
        
        # Ablation experiment switches
        ab_conf = getattr(config, 'ablation', None)
        if ab_conf is None:
             # Default True if config missing
             self.enable_questioner = True
             self.enable_validation = True
             self.enable_solver = True
        elif isinstance(ab_conf, dict):
             self.enable_questioner = ab_conf.get('enable_questioner', True)
             self.enable_validation = ab_conf.get('enable_validation', True)
             self.enable_solver = ab_conf.get('enable_solver', True)
        else:
             # Assume dataclass/object
             self.enable_questioner = getattr(ab_conf, 'enable_questioner', True)
             self.enable_validation = getattr(ab_conf, 'enable_validation', True)
             self.enable_solver = getattr(ab_conf, 'enable_solver', True)
        
        # Constraint: validation requires questioner
        if self.enable_validation and not self.enable_questioner:
            print("[WARNING] enable_validation=True but enable_questioner=False. Auto-disabling validation.")
            self.enable_validation = False
        
        # Log ablation mode
        mode_name = self._get_ablation_mode_name()
        print(f"[Ablation] Mode: {mode_name}")
        print(f"[Ablation]   enable_questioner={self.enable_questioner}")
        print(f"[Ablation]   enable_validation={self.enable_validation}")
        print(f"[Ablation]   enable_solver={self.enable_solver}")
    
    def _get_ablation_mode_name(self) -> str:
        """Get human-readable ablation mode name."""
        if not self.enable_questioner and self.enable_solver:
            return "Solver-only (Mode A)"
        elif self.enable_questioner and self.enable_validation and self.enable_solver:
            return "Full Pipeline (Mode B)"
        elif self.enable_questioner and not self.enable_validation and self.enable_solver:
            return "No Validation (Mode C)"
        elif self.enable_questioner and not self.enable_solver:
            return "Questioner-only (Mode D)"
        else:
            return "Custom"


    def _create_evolving_dataloader(self, records: List[Dict]) -> StatefulDataLoader:
        """Create a new dataloader from evolved records."""
        dataset = EvolvingDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length
        )
        
        if self.config.data.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.config.data.seed or 42)
            sampler = RandomSampler(data_source=dataset, generator=generator)
        else:
            sampler = SequentialSampler(data_source=dataset)
        
        return StatefulDataLoader(
            dataset=dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=sampler,
            num_workers=4,
            collate_fn=evolving_collate_fn,
            pin_memory=False,
            drop_last=True,
        )
    
    def _save_epoch_snapshot(self, records: List[Dict], epoch: int):
        """Save epoch data snapshot for evolution tracking."""
        try:
            snapshot_dir = os.path.join(self.config.trainer.default_local_dir, "epoch_snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)
            
            df = pd.DataFrame(records)
            snapshot_path = os.path.join(snapshot_dir, f"epoch_{epoch}.parquet")
            df.to_parquet(snapshot_path)
            print(f"[Epoch {epoch}] Saved snapshot with {len(records)} samples -> {snapshot_path}")
            
            # Log evolution stats
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "evolution/epoch": epoch,
                        "evolution/dataset_size": len(records),
                    }, step=self.global_steps)
            except Exception:
                pass
                
        except Exception as e:
            print(f"[Epoch {epoch}] Failed to save snapshot: {e}")

    def fit(self):
        """
        Evolutionary training loop with data cycling across epochs.
        """
        self.logger = Tracker(
            loggers=self.config.trainer.logger,
            config=self.config.to_dict(),
        )
        self.global_steps = 0
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # Load checkpoint before doing anything
        self._load_checkpoint()

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.val_only:
                return

        # Initialize current dataloader (will be rebuilt after each epoch)
        current_dataloader = self.train_dataloader
        
        for epoch in tqdm(range(self.config.trainer.total_epochs), desc="Epoch", position=0):
            epoch_log_data = []
            next_epoch_buffer = []  # Buffer for next epoch's data
            
            for batch_dict in tqdm(current_dataloader, desc="Running step", position=1):
                self.global_steps += 1
                self.global_step = self.global_steps
                if self.global_steps > self.training_steps:
                    break

                metrics, timing_raw = {}, {}
                batch = DataProto.from_single_dict(batch_dict)
                batch_tensors = batch.batch
                dataset_infos = batch.non_tensor_batch
                
                original_input_ids = batch_tensors['input_ids']
                original_texts = self.tokenizer.batch_decode(original_input_ids, skip_special_tokens=True)

                with timer("step", timing_raw):
                    # -----------------------------------------------------------
                    # Ablation Mode Check
                    # -----------------------------------------------------------
                    
                    # Mode A: Solver-only - Skip Questioner (Phase 1-3)
                    if not self.enable_questioner:
                        # Use seed data directly
                        final_prompts = list(dataset_infos['problem_description'])
                        final_entry_points = list(dataset_infos['entry_point'])
                        final_ground_truth_outputs = list(dataset_infos['outputs'])
                        final_solutions = list(dataset_infos['ground_truth'])
                        source_flags = [0] * len(final_prompts)  # All original
                        g_mask = [1] * len(final_prompts)
                        parsed_tags_list = [[] for _ in range(len(final_prompts))]
                        execution_valid_mask = [True] * len(final_prompts)
                        
                        # Create dummy questioner output for batch construction
                        questioner_prompts = []
                        for i in range(len(original_input_ids)):
                            row = {
                                'problem_description': dataset_infos['problem_description'][i],
                                'completion': dataset_infos['ground_truth'][i],
                                'entry_point': dataset_infos['entry_point'][i]
                            }
                            msgs = construct_questioner_msgs(row)
                            prompt_text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                            questioner_prompts.append(prompt_text)
                        
                        questioner_inputs = self.tokenizer(questioner_prompts, return_tensors='pt', padding=True, truncation=True)
                        questioner_inputs['position_ids'] = questioner_inputs['attention_mask'].long().cumsum(-1) - 1
                        questioner_inputs['position_ids'].masked_fill_(questioner_inputs['attention_mask'] == 0, 1)
                        questioner_inputs = {k: v.to(original_input_ids.device) for k, v in questioner_inputs.items()}
                        
                        questioner_batch = DataProto.from_single_dict(questioner_inputs)
                        questioner_batch.non_tensor_batch = {
                            **{k: v for k, v in dataset_infos.items()},
                            'raw_prompt_ids': _create_raw_prompt_ids(self.tokenizer, questioner_prompts),
                        }
                        questioner_batch.meta_info = {
                            'action': 'question',
                            'temperature': self.config.worker.rollout.temperature
                        }
                        
                        # Generate dummy responses (just copy input for logging consistency)
                        with timer("gen_question", timing_raw):
                            questioner_output = self.actor_rollout_wg.generate_sequences(questioner_batch)
                        
                        transformed_outputs = ["[SOLVER-ONLY MODE: No transformation]"] * len(original_input_ids)
                        transformed_problems = list(dataset_infos['problem_description'])
                        transformed_solutions = list(dataset_infos['ground_truth'])
                        transformed_entry_points = list(dataset_infos['entry_point'])
                        verifier_last_lines = ["[SKIPPED]"] * len(original_input_ids)
                        verifier_outputs = ["[SKIPPED]"] * len(original_input_ids)
                        generated_ground_truth_outputs = list(dataset_infos['outputs'])
                        
                        # Phase 5 will handle next_epoch_buffer creation for all modes
                    
                    # Skip Phase 1-3 if questioner is disabled (already set up in Mode A)
                    skip_phase_1_3 = not self.enable_questioner
                    
                    if not skip_phase_1_3:
                        # -----------------------------------------------------------
                        # Phase 1: Questioner (Transform Problem)
                        # -----------------------------------------------------------
                    
                        questioner_prompts = []
                        for i in range(len(original_input_ids)):
                            row = {
                                'problem_description': dataset_infos['problem_description'][i],
                                'completion': dataset_infos['ground_truth'][i],
                                'entry_point': dataset_infos['entry_point'][i]
                            }
                            msgs = construct_questioner_msgs(row)
                            prompt_text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                            questioner_prompts.append(prompt_text)

                        questioner_inputs = self.tokenizer(questioner_prompts, return_tensors='pt', padding=True, truncation=True)
                        questioner_inputs['position_ids'] = questioner_inputs['attention_mask'].long().cumsum(-1) - 1
                        questioner_inputs['position_ids'].masked_fill_(questioner_inputs['attention_mask'] == 0, 1)
                        questioner_inputs = {k: v.to(original_input_ids.device) for k, v in questioner_inputs.items()}
                        
                        questioner_batch = DataProto.from_single_dict(questioner_inputs)
                        questioner_batch.non_tensor_batch = {
                            **{k: v for k, v in dataset_infos.items()},
                            'raw_prompt_ids': _create_raw_prompt_ids(self.tokenizer, questioner_prompts),
                        }
                        questioner_batch.meta_info = {
                            'action': 'question',
                            'temperature': self.config.worker.rollout.temperature
                        }
                        
                        with timer("gen_question", timing_raw):
                            questioner_output = self.actor_rollout_wg.generate_sequences(questioner_batch)
                        
                        transformed_outputs = self.tokenizer.batch_decode(questioner_output.batch['responses'], skip_special_tokens=True)
                        
                        # Parse outputs
                        transformed_problems = []
                        transformed_solutions = []
                        transformed_entry_points = []
                        
                        for out_text in transformed_outputs:
                            t_prob = extract_tag(out_text, "transformed_question")
                            t_sol = extract_tag(out_text, "transformed_solution_code")
                            t_entry = extract_tag(out_text, "transformed_entry_point")
                            transformed_problems.append(t_prob)
                            transformed_solutions.append(t_sol)
                            transformed_entry_points.append(t_entry)

                        # -----------------------------------------------------------
                        # Phase 1.5: Execution Check (Generate Ground Truth)
                        # -----------------------------------------------------------
                        
                        execution_valid_mask = []
                        generated_ground_truth_outputs = []
                        
                        # Import unified test execution function
                        from rewards.code_reward import execute_test_cases
                        
                        with ThreadPoolExecutor(max_workers=32) as executor:
                            futures = []
                            timeout_sec = getattr(self.config.data, 'execution_timeout_sec', 5.0)
                            
                            for i in range(len(original_input_ids)):
                                # Use unified execute_test_cases function
                                futures.append(executor.submit(
                                    execute_test_cases,
                                    code=transformed_solutions[i],
                                    entry_point=transformed_entry_points[i],
                                    inputs=dataset_infos['inputs'][i],
                                    expected_outputs=dataset_infos['outputs'][i],  # Dummy, we want outputs
                                    prompt=dataset_infos.get('prompt', [""] * len(original_input_ids))[i],
                                    timeout_sec=timeout_sec,
                                    return_outputs=True  # Get new outputs for transformed problem
                                ))
                            
                            results = [f.result() for f in futures]
                            
                        for success, outputs in results:
                            execution_valid_mask.append(success)
                            generated_ground_truth_outputs.append(outputs)

                        # -----------------------------------------------------------
                        # Phase 2: Verifier / ConceptExpandGate (G_novel)
                        # Outputs: g_mask (gate), parsed_tags_list
                        # -----------------------------------------------------------
                        
                        verifier_prompts = []
                        verifier_outputs = []
                        verifier_last_lines = []
                        g_mask = []          # ConceptExpandGate: g ∈ {0,1}
                        parsed_tags_list = []  # New tags from verifier
                        
                        if self.enable_validation:
                            from rewards.code_reward import parse_tags_from_response
                            
                            for i in range(len(transformed_problems)):
                                t_prob = transformed_problems[i]
                                t_sol = transformed_solutions[i]
                                
                                # Get anchor tags for this sample
                                anchor_tags_raw = dataset_infos.get('tags', [[] for _ in range(len(transformed_problems))])[i]
                                if isinstance(anchor_tags_raw, (list, tuple)):
                                    anchor_tags_str = ", ".join(str(t) for t in anchor_tags_raw)
                                else:
                                    anchor_tags_str = str(anchor_tags_raw)
                                
                                if execution_valid_mask[i] and t_prob and t_sol:
                                    msgs = construct_verifier_msgs(t_prob, t_sol, anchor_tags=anchor_tags_str)
                                    prompt_text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                                    verifier_prompts.append(prompt_text)
                                else:
                                    verifier_prompts.append("SKIP")
                                
                            verifier_inputs = self.tokenizer(verifier_prompts, return_tensors='pt', padding=True, truncation=True)
                            verifier_inputs['position_ids'] = verifier_inputs['attention_mask'].long().cumsum(-1) - 1
                            verifier_inputs['position_ids'].masked_fill_(verifier_inputs['attention_mask'] == 0, 1)
                            verifier_inputs = {k: v.to(original_input_ids.device) for k, v in verifier_inputs.items()}
                            
                            verifier_batch = DataProto.from_single_dict(verifier_inputs)
                            verifier_batch.non_tensor_batch = {
                                **{k: v for k, v in dataset_infos.items()},
                                'raw_prompt_ids': _create_raw_prompt_ids(self.tokenizer, verifier_prompts),
                            }
                            verifier_temp = getattr(self.config.worker.rollout, 'verifier_temperature', 0.0)
                            verifier_batch.meta_info = {'action': 'verify', 'temperature': verifier_temp}
                            
                            with timer("gen_verify", timing_raw):
                                verifier_output = self.actor_rollout_wg.generate_sequences(verifier_batch)
                                
                            verifier_responses = self.tokenizer.batch_decode(verifier_output.batch['responses'], skip_special_tokens=True)
                            verifier_outputs = []
                            
                            for i, resp in enumerate(verifier_responses):
                                if verifier_prompts[i] == "SKIP":
                                    g_mask.append(0)
                                    parsed_tags_list.append([])
                                    verifier_last_lines.append("SKIP")
                                    verifier_outputs.append("SKIP")
                                    continue
                                verifier_outputs.append(resp)
                                
                                # Parse tags from response
                                new_tags = parse_tags_from_response(resp)
                                parsed_tags_list.append(new_tags)
                                
                                # Parse Yes/No (ConceptExpandGate result)
                                resp_lower = resp.lower().strip()
                                last_line = resp_lower.split('\n')[-1]
                                verifier_last_lines.append(resp.strip().split('\n')[-1])  # Original case for logging
                                
                                # Strategy 1: Check last line first (original logic)
                                if "yes" in last_line and "no" not in last_line:
                                    g_mask.append(1)
                                    continue
                                elif "no" in last_line and "yes" not in last_line:
                                    g_mask.append(0)
                                    continue
                                
                                # Strategy 2: Check for definitive Yes/No at start or end
                                if resp_lower.startswith("yes") or resp_lower.endswith("yes") or resp_lower.endswith("yes."):
                                    g_mask.append(1)
                                elif resp_lower.startswith("no") or resp_lower.endswith("no") or resp_lower.endswith("no."):
                                    g_mask.append(0)
                                # Strategy 3: Count occurrences
                                elif resp_lower.count("yes") > resp_lower.count("no"):
                                    g_mask.append(1)
                                else:
                                    g_mask.append(0)
                        else:
                            # Mode C: Skip validation
                            verifier_prompts = ["SKIP"] * len(transformed_problems)
                            verifier_outputs = ["SKIPPED (Mode C)"] * len(transformed_problems)
                            verifier_last_lines = ["SKIPPED (Mode C)"] * len(transformed_problems)
                            g_mask = [1 if success else 0 for success in execution_valid_mask]
                            parsed_tags_list = [[] for _ in range(len(transformed_problems))]
                             
                    # -----------------------------------------------------------
                    # Phase 3a: Select Solver Input (x_in)
                    # Algorithm line 10: x_in = (x if valid else a.problem)
                    # Anchor update deferred to Phase 5 (after solver)
                    # -----------------------------------------------------------
                    final_prompts = []
                    final_entry_points = []
                    final_ground_truth_outputs = []
                    final_solutions = []
                    source_flags = []  # 1 = transformed, 0 = original

                    for i in range(len(original_texts)):
                        if execution_valid_mask[i]:
                            # Use transformed problem as solver input
                            final_prompts.append(transformed_problems[i])
                            final_entry_points.append(transformed_entry_points[i])
                            final_ground_truth_outputs.append(generated_ground_truth_outputs[i])
                            final_solutions.append(transformed_solutions[i])
                            source_flags.append(1)
                        else:
                            # Revert to Original
                            final_prompts.append(dataset_infos['problem_description'][i])
                            final_entry_points.append(dataset_infos['entry_point'][i])
                            final_ground_truth_outputs.append(dataset_infos['outputs'][i])
                            final_solutions.append(dataset_infos['ground_truth'][i])
                            source_flags.append(0)

                    # -----------------------------------------------------------
                    # Phase 4: Solver (Solve) with G-Sampling
                    # -----------------------------------------------------------
                    num_samples = self.config.worker.rollout.solver_sampling_n
                    solver_temp = self.config.worker.rollout.solver_temperature

                    # Initialize Solver variables (default empty/dummy)
                    solver_prompts_chat = [""] * len(final_prompts)
                    best_solver_outputs = [""] * len(final_prompts)
                    pass_rates = [0.0] * len(final_prompts)
                    individual_solver_rewards = []
                    all_solutions = [[] for _ in range(len(final_prompts))]
                    all_sample_outputs = [] 
                    
                    if self.enable_solver:
                        solver_prompts_chat = []
                        for text, sol_code in zip(final_prompts, final_solutions):
                            try:
                                starter_code = starter_code_from_solution_text(sol_code)
                            except Exception:
                                starter_code = ""
                                
                            msgs = construct_solver_msgs(text, starter_code)
                            prompt_text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                            solver_prompts_chat.append(prompt_text)
    
                        solver_inputs = self.tokenizer(solver_prompts_chat, return_tensors='pt', padding=True, truncation=True)
                        solver_inputs['position_ids'] = solver_inputs['attention_mask'].long().cumsum(-1) - 1
                        solver_inputs['position_ids'].masked_fill_(solver_inputs['attention_mask'] == 0, 1)
                        solver_inputs = {k: v.to(original_input_ids.device) for k, v in solver_inputs.items()}
    
                        all_solutions = []
                        for _ in range(len(final_prompts)):
                            all_solutions.append([])
    
                        # Store ALL sample outputs for expanded solver batch
                        all_sample_outputs = []
    
                        with timer("gen_solution_sampling", timing_raw):
                            for sample_idx in range(num_samples):
                                solver_batch = DataProto.from_single_dict(solver_inputs)
                                solver_batch.non_tensor_batch = {
                                    **{k: v for k, v in dataset_infos.items()},
                                    'raw_prompt_ids': _create_raw_prompt_ids(self.tokenizer, solver_prompts_chat),
                                }
                                solver_batch.meta_info = {
                                    'action': 'solve',
                                    'temperature': solver_temp
                                }
                                
                                solver_output = self.actor_rollout_wg.generate_sequences(solver_batch)
                                decoded_responses = self.tokenizer.batch_decode(solver_output.batch['responses'], skip_special_tokens=True)
                                
                                for i, resp in enumerate(decoded_responses):
                                    all_solutions[i].append(resp)
                                
                                # Store all sample outputs for later concatenation
                                all_sample_outputs.append(solver_output)
                        
                        # -----------------------------------------------------------
                        # Phase 4.5: Evaluate Solutions & Compute Pass Rate
                        # -----------------------------------------------------------
                        pass_rates = []
                        eval_inputs = []
                        
                        for i, solutions in enumerate(all_solutions):
                            row = {
                                'entry_point': final_entry_points[i],
                                'inputs': dataset_infos['inputs'][i],
                                'outputs': final_ground_truth_outputs[i],
                                'prompt': dataset_infos['prompt'][i]
                            }
                            for sol in solutions:
                                eval_inputs.append((sol, row))
    
                        def _eval_wrapper(args):
                            return evaluate_single_code(args[0], args[1], timeout_sec=2.0)
    
                        with ThreadPoolExecutor(max_workers=32) as executor:
                            results = list(executor.map(_eval_wrapper, eval_inputs))
                        
                        curr = 0
                        for i in range(len(final_prompts)):
                            problem_results = results[curr : curr + num_samples]
                            curr += num_samples
                            passes = sum([1 for r in problem_results if r == 1])
                            pass_rate = passes / num_samples
                            pass_rates.append(pass_rate)
    
                        # Use individual granular rewards for all G samples
                        # results contains rewards in order: [problem0_sample0, problem0_sample1, ..., problem1_sample0, ...]
                        # Each reward is: -1 (format error), -0.5 (runtime error), 0 (wrong output), 1 (pass)
                        individual_solver_rewards = [float(r) for r in results]  # All N*G individual rewards
                        print(f"[DEBUG] Individual solver rewards: n={len(individual_solver_rewards)}, mean={np.mean(individual_solver_rewards):.4f}")
                        
                        # Logging selection
                        best_solver_outputs = []
                        for i, sols in enumerate(all_solutions):
                            # results for problem i: results[i*num_samples : (i+1)*num_samples]
                            problem_results = results[i * num_samples : (i + 1) * num_samples]
                            # Find first passing (reward == 1)
                            best_output = sols[0] if sols else ""  # default: first sample
                            for j, r in enumerate(problem_results):
                                if r == 1:  # Found a passing solution
                                    best_output = sols[j]
                                    break
                            best_solver_outputs.append(best_output)

                    # -----------------------------------------------------------
                    # Phase 5: Anchor Update (y_d, y, next_epoch_buffer)
                    # Algorithm lines 16, 25, 29-32
                    # -----------------------------------------------------------
                    frontier_L = getattr(self.config.algorithm, 'frontier_L', 0.1)
                    frontier_U = getattr(self.config.algorithm, 'frontier_U', 0.7)
                    dist_lambda = getattr(self.config.algorithm, 'dist_lambda', 0.3)
                    
                    y_d_list = []   # 1[p ∈ [L,U]]
                    y_list = []     # y_d ∧ g
                    
                    for i in range(len(original_texts)):
                        p = pass_rates[i]
                        g = g_mask[i] if i < len(g_mask) else 0
                        
                        # y_d = 1[p ∈ [L,U]]
                        y_d = 1 if (frontier_L <= p <= frontier_U) else 0
                        y = y_d & g  # y = y_d ∧ g
                        
                        y_d_list.append(y_d)
                        y_list.append(y)
                        
                        if y == 1 and execution_valid_mask[i]:
                            # Replace anchor with transformed problem + new tags
                            new_tags = parsed_tags_list[i] if i < len(parsed_tags_list) else []
                            evolved_row = {
                                'problem_description': transformed_problems[i],
                                'completion': transformed_solutions[i],
                                'entry_point': transformed_entry_points[i],
                                'inputs': dataset_infos['inputs'][i],
                                'outputs': generated_ground_truth_outputs[i],
                                'prompt': dataset_infos.get('prompt', [""] * len(original_input_ids))[i],
                                'task_id': dataset_infos.get('task_id', [f"evolved_{epoch}_{i}"] * len(original_input_ids))[i],
                                'question_id': dataset_infos.get('question_id', [0] * len(original_input_ids))[i],
                                'difficulty': dataset_infos.get('difficulty', ['medium'] * len(original_input_ids))[i],
                                'tags': new_tags if new_tags else dataset_infos.get('tags', [[]] * len(original_input_ids))[i],
                                'starter_code': dataset_infos.get('starter_code', [''] * len(original_input_ids))[i],
                            }
                        else:
                            # Keep original anchor unchanged
                            evolved_row = {
                                'problem_description': dataset_infos['problem_description'][i],
                                'completion': dataset_infos['ground_truth'][i],
                                'entry_point': dataset_infos['entry_point'][i],
                                'inputs': dataset_infos['inputs'][i],
                                'outputs': dataset_infos['outputs'][i],
                                'prompt': dataset_infos.get('prompt', [""] * len(original_input_ids))[i],
                                'task_id': dataset_infos.get('task_id', [f"original_{i}"] * len(original_input_ids))[i],
                                'question_id': dataset_infos.get('question_id', [0] * len(original_input_ids))[i],
                                'difficulty': dataset_infos.get('difficulty', ['medium'] * len(original_input_ids))[i],
                                'tags': dataset_infos.get('tags', [[]] * len(original_input_ids))[i],
                                'starter_code': dataset_infos.get('starter_code', [''] * len(original_input_ids))[i],
                            }
                        
                        next_epoch_buffer.append(evolved_row)

                    # Initialize PPO Batch using Questioner's rollout output
                    batch = questioner_output
                    
                    batch.non_tensor_batch['r_solve'] = np.array(pass_rates)
                    batch.non_tensor_batch['is_transformed'] = np.array(source_flags)
                    batch.non_tensor_batch['g_mask'] = np.array(g_mask)
                    batch.non_tensor_batch['y_d'] = np.array(y_d_list)
                    batch.non_tensor_batch['y'] = np.array(y_list)
                    
                    ground_truth_dicts = []
                    for i in range(len(pass_rates)):
                        ground_truth_dicts.append({
                            'problem_description': dataset_infos['problem_description'][i],
                            'completion': dataset_infos['ground_truth'][i],
                            'entry_point': dataset_infos['entry_point'][i],
                            # New algorithm fields
                            'valid': execution_valid_mask[i],  # code execution validity
                            'y_hat': g_mask[i],      # BucketVerify = verifier Yes/No
                            'g': g_mask[i],           # ConceptExpandGate (same as y_hat for now)
                            'y_d': y_d_list[i],       # 1[p ∈ [L,U]]
                            'y': y_list[i],           # y_d ∧ g
                            'p': pass_rates[i],       # solver pass rate
                            'L': frontier_L,
                            'U': frontier_U,
                            'dist_lambda': dist_lambda,
                        })
                    batch.non_tensor_batch['ground_truth'] = np.array(ground_truth_dicts, dtype=object)
                    
                    batch_log_data = {
                        "original_question": list(dataset_infos['problem_description']),
                        "raw_model_output": transformed_outputs,
                        "transformed_question": transformed_problems,
                        "transformed_solution": transformed_solutions,
                        "is_exec_valid": execution_valid_mask,
                        "verify_output": verifier_outputs,
                        "verifier_response": verifier_last_lines,
                        "g_mask": g_mask,
                        "y_d": y_d_list,
                        "y": y_list,
                        "selected_source": source_flags,
                        "final_prompt": final_prompts,
                        "solver_input": solver_prompts_chat,
                        "solver_output": best_solver_outputs,
                        "pass_rate": pass_rates
                    }
                    epoch_log_data.append(pd.DataFrame(batch_log_data))
                    
                    # W&B Logging
                    try:
                        import wandb
                        if wandb.run is not None:
                            sample_df = pd.DataFrame(batch_log_data)
                            sample_df = sample_df.drop(columns=["final_prompt"], errors="ignore")
                            for col in sample_df.columns:
                                if sample_df[col].dtype == object:
                                    sample_df[col] = sample_df[col].astype(str).str[:]
                            wandb_table = wandb.Table(dataframe=sample_df)
                            wandb.log({f"samples/step_{self.global_steps}": wandb_table}, step=self.global_steps)
                            
                            # Log evolution metrics
                            transform_success_rate = sum(source_flags) / len(source_flags) if source_flags else 0
                            anchor_update_rate = sum(y_list) / len(y_list) if y_list else 0
                            avg_pass_rate = np.mean(pass_rates) if pass_rates else 0
                            wandb.log({
                                "evolution/transform_success_rate": transform_success_rate,
                                "evolution/anchor_update_rate": anchor_update_rate,
                                "evolution/avg_pass_rate": avg_pass_rate,
                                "evolution/frontier_L": frontier_L,
                                "evolution/frontier_U": frontier_U,
                                "evolution/epoch": epoch,
                            }, step=self.global_steps)
                    except Exception as e:
                        pass

                    # -----------------------------------------------------------
                    # Phase 5: Dual PPO Updates (Questioner + Solver)
                    # -----------------------------------------------------------
                    
                    # === Step 1: Prepare both batches ===
                    
                    questioner_batch = None
                    if self.enable_questioner:
                        # Questioner batch
                        questioner_batch = questioner_output
                        questioner_batch.non_tensor_batch['uid'] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(questioner_batch.batch))], 
                            dtype=object
                        )
                        questioner_batch.meta_info["global_token_num"] = torch.sum(
                            questioner_batch.batch["attention_mask"], dim=-1
                        ).tolist()
                    
                    solver_batch = None
                    if self.enable_solver and all_sample_outputs:
                        # Solver batch - Expand to include ALL G samples per problem
                        # Concatenate all sample outputs: batch size becomes N * G
                        # Note: all_sample_outputs is a list of G DataProto objects, each with N samples
                        # We need to interleave them so that samples for same problem are adjacent
                        
                        n_problems = len(final_prompts)
                        
                        # Concatenate all sample outputs
                        # all_sample_outputs[sample_idx] contains all N problems for that sample
                        solver_batch_tmp = DataProto.concat(all_sample_outputs)  # Shape: [N*G, ...]
                        
                        # Reorder from [p0_all, p1_all, ...] to [p0_s0, p0_s1, ..., p1_s0, p1_s1, ...]
                        total_samples = n_problems * num_samples
                        reorder_indices = []
                        for p in range(n_problems):
                            for s in range(num_samples):
                                # Original index: sample s, problem p -> s * n_problems + p
                                reorder_indices.append(s * n_problems + p)
                        reorder_indices = torch.tensor(reorder_indices, device=solver_batch_tmp.batch['input_ids'].device)
                        
                        # Reorder all tensors in batch
                        for key in list(solver_batch_tmp.batch.keys()):
                            if solver_batch_tmp.batch[key] is not None:
                                solver_batch_tmp.batch[key] = solver_batch_tmp.batch[key][reorder_indices]
                        
                        print(f"[DEBUG] Solver batch expanded: n_problems={n_problems}, num_samples={num_samples}, total={total_samples}")
                        
                        # Expand metadata to match new batch size (N*G)
                        expanded_entry_points = []
                        expanded_inputs = []
                        expanded_outputs = []
                        expanded_prompts = []
                        expanded_raw_prompt_ids = []
                        
                        for p in range(n_problems):
                            for s in range(num_samples):
                                expanded_entry_points.append(final_entry_points[p])
                                expanded_inputs.append(dataset_infos['inputs'][p])
                                expanded_outputs.append(final_ground_truth_outputs[p])
                                expanded_prompts.append(dataset_infos.get('prompt', [''] * n_problems)[p])
                        
                        # Get raw_prompt_ids from first sample output
                        if all_sample_outputs:
                            first_raw_prompt_ids = all_sample_outputs[0].non_tensor_batch.get('raw_prompt_ids')
                            if first_raw_prompt_ids is not None:
                                for p in range(n_problems):
                                    for s in range(num_samples):
                                        expanded_raw_prompt_ids.append(first_raw_prompt_ids[p])
                        
                        new_non_tensor_batch = {
                            'uid': np.array([str(uuid.uuid4()) for _ in range(total_samples)], dtype=object),
                            'entry_point': np.array(expanded_entry_points, dtype=object),
                            'inputs': np.array(expanded_inputs, dtype=object),
                            'outputs': np.array(expanded_outputs, dtype=object),
                            'prompt': np.array(expanded_prompts, dtype=object),
                        }
                        if expanded_raw_prompt_ids:
                            new_non_tensor_batch['raw_prompt_ids'] = np.array(expanded_raw_prompt_ids, dtype=object)
                        solver_batch_tmp.non_tensor_batch = new_non_tensor_batch
                        
                        # Assign individual granular rewards to each sample
                        solver_ground_truth_dicts = []
                        for idx in range(total_samples):
                            # p = idx // num_samples  # Problem index
                            # Note: individual_solver_rewards is already 1D array of length total_samples
                            # aligned with the reordered batch (since we reordered results same way? 
                            # Wait, individual_solver_rewards was constructed from results list:
                            # results list was: [p0_s0...p0_sG-1, p1_s0...].
                            # This ALREADY MATCHES our reordered structure [p0_s0...].
                            # So we can just use idx.
                            solver_ground_truth_dicts.append({
                                'precomputed_solver_reward': individual_solver_rewards[idx],
                                'entry_point': expanded_entry_points[idx],
                                'inputs': expanded_inputs[idx],
                                'outputs': expanded_outputs[idx],
                                'prompt': expanded_prompts[idx],
                            })
                        solver_batch_tmp.non_tensor_batch['ground_truth'] = np.array(solver_ground_truth_dicts, dtype=object)
                        solver_batch_tmp.meta_info["global_token_num"] = torch.sum(
                            solver_batch_tmp.batch["attention_mask"], dim=-1
                        ).tolist()
                        
                        solver_batch = solver_batch_tmp
                    
                    # === Step 2: Compute all log_probs/values BEFORE any updates (FIX BUG #2) ===
                    
                    if questioner_batch:
                        # Questioner: rewards
                        with timer("questioner_reward", timing_raw):
                            questioner_reward_ref = self.reward_fn.compute_reward.remote(questioner_batch)
    
                        # Questioner: old_log_probs
                        with timer("questioner_old", timing_raw):
                            questioner_old_log_probs = self.actor_rollout_wg.compute_log_probs(questioner_batch)
                            questioner_batch = questioner_batch.union(questioner_old_log_probs)
                    
                    if solver_batch:
                        # Solver: rewards (using precomputed)
                        with timer("solver_reward", timing_raw):
                            solver_reward_ref = self.solver_reward_fn.compute_reward.remote(solver_batch)
                        
                        # Solver: old_log_probs (BEFORE actor update!)
                        with timer("solver_old", timing_raw):
                            solver_old_log_probs = self.actor_rollout_wg.compute_log_probs(solver_batch)
                            # Clear meta_info to avoid conflicts (temperature may differ)
                            solver_old_log_probs.meta_info = {}
                            solver_batch = solver_batch.union(solver_old_log_probs)
    
                    # Reference policy (if enabled) - for both batches
                    if self.use_reference_policy:
                        if questioner_batch:
                            with timer("questioner_ref", timing_raw):
                                questioner_ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(questioner_batch)
                                questioner_batch = questioner_batch.union(questioner_ref_log_probs)
                        
                        if solver_batch:
                            with timer("solver_ref", timing_raw):
                                solver_ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(solver_batch)
                                solver_ref_log_probs.meta_info = {}
                                solver_batch = solver_batch.union(solver_ref_log_probs)
    
                    # Critic values (if enabled) - for both batches
                    if self.use_critic:
                        if questioner_batch:
                            with timer("questioner_values", timing_raw):
                                questioner_values = self.critic_wg.compute_values(questioner_batch)
                                questioner_batch = questioner_batch.union(questioner_values)
                        
                        if solver_batch:
                            with timer("solver_values", timing_raw):
                                solver_values = self.critic_wg.compute_values(solver_batch)
                                solver_values.meta_info = {}
                                solver_batch = solver_batch.union(solver_values)
    
                    # Compute rewards for both batches (before combining)
                    if questioner_batch:
                        with timer("questioner_reward", timing_raw):
                            questioner_reward_tensor, questioner_reward_metrics = ray.get(questioner_reward_ref)
                            questioner_batch.batch["token_level_scores"] = questioner_reward_tensor
                            questioner_reward_metrics = {f"questioner_reward/{k}": v for k, v in reduce_metrics(questioner_reward_metrics).items()}
                            metrics.update(questioner_reward_metrics)
                            
                            if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                                questioner_batch, questioner_kl_metrics = apply_kl_penalty(questioner_batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                                metrics.update(questioner_kl_metrics)
                            else:
                                questioner_batch.batch["token_level_rewards"] = questioner_batch.batch["token_level_scores"]
                    
                    if solver_batch:
                        with timer("solver_reward", timing_raw):
                            solver_reward_tensor, solver_reward_metrics = ray.get(solver_reward_ref)
                            solver_batch.batch["token_level_scores"] = solver_reward_tensor
                            solver_reward_metrics = {f"solver_reward/{k}": v for k, v in reduce_metrics(solver_reward_metrics).items()}
                            metrics.update(solver_reward_metrics)
                            
                            if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                                solver_batch, solver_kl_metrics = apply_kl_penalty(solver_batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                                solver_kl_metrics = {f"solver_{k}": v for k, v in solver_kl_metrics.items()}
                                metrics.update(solver_kl_metrics)
                            else:
                                solver_batch.batch["token_level_rewards"] = solver_batch.batch["token_level_scores"]
                    
                    # === Step 3: Unified Batch Update ===
                    combined_batch = None
                    
                    # Case 1: Both batches available - Combine them
                    if questioner_batch is not None and solver_batch is not None:
                        # Pad both batches to same sequence length for concat compatibility
                        with timer("batch_concat", timing_raw):
                            # Get sequence lengths
                            q_seq_len = questioner_batch.batch['input_ids'].shape[1]
                            s_seq_len = solver_batch.batch['input_ids'].shape[1]
                            target_seq_len = max(q_seq_len, s_seq_len)
                            
                            # Get response lengths
                            q_resp_len = questioner_batch.batch['responses'].shape[1]
                            s_resp_len = solver_batch.batch['responses'].shape[1]
                            target_resp_len = max(q_resp_len, s_resp_len)
                            
                            print(f"[DEBUG] Before padding: q_seq_len={q_seq_len}, s_seq_len={s_seq_len}, target_seq_len={target_seq_len}")
                            print(f"[DEBUG] Before padding: q_resp_len={q_resp_len}, s_resp_len={s_resp_len}, target_resp_len={target_resp_len}")
                            
                            # Compute response_mask BEFORE clearing non_tensor_batch
                            questioner_batch.batch['response_mask'] = compute_response_mask(questioner_batch)
                            solver_batch.batch['response_mask'] = compute_response_mask(solver_batch)
                            
                            pad_token_id = self.tokenizer.pad_token_id or 0
                            questioner_batch = pad_batch_to_length(questioner_batch, target_seq_len, target_resp_len, pad_token_id)
                            solver_batch = pad_batch_to_length(solver_batch, target_seq_len, target_resp_len, pad_token_id)
                            
                            # DEBUG: Print tensor shapes after padding
                            # (Omitted debug prints for brevity, but could include if needed)
                            
                            # Ensure both batches have exactly the same keys
                            q_keys = set(questioner_batch.batch.keys())
                            s_keys = set(solver_batch.batch.keys())
                            common_keys = q_keys & s_keys
                            
                            # Remove keys that exist in only one batch
                            for k in list(q_keys - common_keys):
                                del questioner_batch.batch[k]
                            for k in list(s_keys - common_keys):
                                del solver_batch.batch[k]
                            
                            # Define key groups
                            SEQ_LEN_KEYS = {'input_ids', 'attention_mask', 'position_ids'}
                            RESP_LEN_KEYS = {'responses', 'old_log_probs', 'ref_log_probs', 'token_level_scores', 
                                             'token_level_rewards', 'response_mask', 'values', 'prompts'}
                            
                            # Force all 2D tensors to match (truncate if longer)
                            for k in common_keys:
                                qt = questioner_batch.batch[k]
                                st = solver_batch.batch[k]
                                # match length logic
                                if k in SEQ_LEN_KEYS: target_len = target_seq_len
                                elif k in RESP_LEN_KEYS: target_len = target_resp_len
                                else: target_len = max(qt.shape[1] if qt is not None and len(qt.shape) >= 2 else 0,
                                                       st.shape[1] if st is not None and len(st.shape) >= 2 else 0)
                                
                                if qt is not None and len(qt.shape) >= 2 and qt.shape[1] != target_len:
                                    if qt.shape[1] > target_len: questioner_batch.batch[k] = qt[:, -target_len:]
                                    else:
                                        pad_size = target_len - qt.shape[1]
                                        padding = torch.zeros((qt.shape[0], pad_size), dtype=qt.dtype, device=qt.device)
                                        questioner_batch.batch[k] = torch.cat([padding, qt], dim=1)
                                
                                if st is not None and len(st.shape) >= 2 and st.shape[1] != target_len:
                                    if st.shape[1] > target_len: solver_batch.batch[k] = st[:, -target_len:]
                                    else:
                                        pad_size = target_len - st.shape[1]
                                        padding = torch.zeros((st.shape[0], pad_size), dtype=st.dtype, device=st.device)
                                        solver_batch.batch[k] = torch.cat([padding, st], dim=1)
                            
                            # Clear non_tensor_batch and meta_info for clean concat
                            questioner_batch.non_tensor_batch = {}
                            solver_batch.non_tensor_batch = {}
                            questioner_batch.meta_info = {}
                            solver_batch.meta_info = {}
                            
                            combined_batch = DataProto.concat([questioner_batch, solver_batch])
                    
                    # Case 2: Only Questioner
                    elif questioner_batch is not None:
                        combined_batch = questioner_batch
                        combined_batch.batch['response_mask'] = compute_response_mask(combined_batch)
                        combined_batch.non_tensor_batch = {}
                        combined_batch.meta_info = {}
                        
                    # Case 3: Only Solver
                    elif solver_batch is not None:
                        combined_batch = solver_batch
                        combined_batch.batch['response_mask'] = compute_response_mask(combined_batch)
                        combined_batch.non_tensor_batch = {}
                        combined_batch.meta_info = {}
                        
                    # Case 4: None (Should not happen)
                    else:
                        print("[ERROR] Both questioner_batch and solver_batch are None!")
                        return {}

                        
                    # Correctly dedented logic for ALL cases
                    
                    # Ensure batch size is divisible by world_size for proper chunking
                    world_size = self.resource_pool_manager.get_num_gpus()
                    combined_batch, pad_size = pad_dataproto_to_divisor(combined_batch, world_size)
                    
                    # Generate uid for compute_advantage (required for GRPO/RLOO)
                    # Only keep uid to avoid chunk() issues with non-ndarray values
                    # uuid is already imported at top of file
                    combined_batch.non_tensor_batch = {
                        "uid": np.array([str(uuid.uuid4()) for _ in range(len(combined_batch))], dtype=object)
                    }
                    
                    # Set required meta_info for update_actor
                    combined_batch.meta_info = {
                        "temperature": self.config.worker.rollout.temperature,
                        "global_token_num": torch.sum(combined_batch.batch["attention_mask"], dim=-1).tolist(),
                    }
                    
                    print(f"[DEBUG] Combined batch: len={len(combined_batch)}, world_size={world_size}, pad_size={pad_size}")
                
                    # Compute advantage once on combined batch
                    with timer("combined_adv", timing_raw):
                        combined_batch = compute_advantage(
                            combined_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )
                    
                    # Update critic once with combined batch
                    if self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(combined_batch)
                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        critic_metrics = {f"critic/{k}": v for k, v in critic_metrics.items()}
                        metrics.update(critic_metrics)

                    # Update actor once with combined batch
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(combined_batch)
                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        actor_metrics = {f"actor/{k}": v for k, v in actor_metrics.items()}
                        metrics.update(actor_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # Collect metrics
                num_gpus = self.resource_pool_manager.get_num_gpus()
                metrics.update(compute_data_metrics(batch=combined_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=combined_batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=combined_batch, timing_raw=timing_raw, num_gpus=num_gpus))
                
                self.logger.log(data=metrics, step=self.global_steps)

            # -----------------------------------------------------------
            # End of Epoch: Save snapshot and rebuild dataloader
            # -----------------------------------------------------------
            
            # Save epoch rollout logs
            if epoch_log_data:
                try:
                    log_dir = os.path.join(self.config.trainer.default_local_dir, "rollout_logs")
                    os.makedirs(log_dir, exist_ok=True)
                    
                    df_epoch = pd.concat(epoch_log_data, ignore_index=True)
                    timestamp = int(time.time())
                    filename = f"rollout_epoch_{epoch}_{timestamp}.parquet"
                    df_epoch.to_parquet(os.path.join(log_dir, filename))
                    print(f"Saved epoch rollout log to {filename}")
                    
                    try:
                        import wandb
                        if wandb.run is not None:
                            sample_df = df_epoch
                            sample_df = sample_df.drop(columns=["final_prompt"], errors="ignore")
                            for col in sample_df.columns:
                                if sample_df[col].dtype == object:
                                    sample_df[col] = sample_df[col].astype(str).str[:]
                            wandb_table = wandb.Table(dataframe=sample_df)
                            wandb.log({f"samples/epoch_{epoch}": wandb_table}, step=self.global_steps)
                    except Exception as e:
                        print(f"Failed to log to W&B: {e}")
                        
                except Exception as e:
                    print(f"Failed to save epoch rollout log: {e}")
            
            # Save epoch snapshot for evolution tracking
            if next_epoch_buffer:
                self._save_epoch_snapshot(next_epoch_buffer, epoch)
                
                # Rebuild dataloader for next epoch (if not last epoch)
                if epoch < self.config.trainer.total_epochs - 1:
                    print(f"\n[Epoch {epoch}] Rebuilding dataloader with {len(next_epoch_buffer)} evolved samples for next epoch...")
                    current_dataloader = self._create_evolving_dataloader(next_epoch_buffer)
                    print(f"[Epoch {epoch}] New dataloader created with {len(current_dataloader)} batches\n")
            
            # Save checkpoint at end of each epoch (if enabled)
            if getattr(self.config.trainer, 'save_every_epoch', True):
                print(f"[Epoch {epoch}] Saving checkpoint...")
                self._save_checkpoint()
            
            # Run validation at end of each epoch (if enabled)
            if getattr(self.config.trainer, 'val_every_epoch', False) and self.val_reward_fn is not None:
                print(f"[Epoch {epoch}] Running validation...")
                val_metrics = self._validate()
                val_metrics['epoch'] = epoch
                self.logger.log(data=val_metrics, step=self.global_steps)

        # End of training validation
        if self.val_reward_fn is not None:
            if self.config.trainer.val_freq <= 0 or self.global_steps % self.config.trainer.val_freq != 0:
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_steps)
        
        self._save_checkpoint()

    def _validate(self) -> Dict[str, Any]:
        """
        Validate solver performance on original problems (no questioner).
        Measures raw solver pass rate as a clean baseline metric.
        """
        from collections import defaultdict
        from rewards.code_reward import evaluate_single_code
        from rewards.extract_starter_code import starter_code_from_solution_text
        
        metrics = defaultdict(list)
        sample_data = []
        
        for batch_dict in self.val_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            original_input_ids = batch.batch['input_ids']
            dataset_infos = batch.non_tensor_batch
            
            # === Solver only: solve original problems ===
            solver_prompts = []
            for i in range(len(original_input_ids)):
                prob = dataset_infos['problem_description'][i]
                try:
                    starter_code = starter_code_from_solution_text(dataset_infos['ground_truth'][i]) or ""
                except:
                    starter_code = ""
                msgs = construct_solver_msgs(prob, starter_code)
                prompt_text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                solver_prompts.append(prompt_text)
            
            solver_inputs = self.tokenizer(solver_prompts, return_tensors='pt', padding=True, truncation=True)
            solver_inputs['position_ids'] = solver_inputs['attention_mask'].long().cumsum(-1) - 1
            solver_inputs['position_ids'].masked_fill_(solver_inputs['attention_mask'] == 0, 1)
            solver_inputs = {k: v.to(original_input_ids.device) for k, v in solver_inputs.items()}
            
            solver_batch = DataProto.from_single_dict(solver_inputs)
            solver_batch.non_tensor_batch = {
                'raw_prompt_ids': _create_raw_prompt_ids(self.tokenizer, solver_prompts),
            }
            solver_batch.meta_info = {'action': 'solve'}
            solver_output = self.actor_rollout_wg.generate_sequences(solver_batch)
            
            solver_responses = self.tokenizer.batch_decode(solver_output.batch['responses'], skip_special_tokens=True)
            
            # === Evaluate ===
            pass_count = 0
            for i, resp in enumerate(solver_responses):
                row = {
                    'entry_point': dataset_infos['entry_point'][i],
                    'inputs': dataset_infos['inputs'][i],
                    'outputs': dataset_infos['outputs'][i],
                    'prompt': dataset_infos.get('prompt', [''])[i] if 'prompt' in dataset_infos else '',
                }
                score = evaluate_single_code(resp, row, timeout_sec=2.0)
                if score == 1:
                    pass_count += 1
                
                if len(sample_data) < 10:
                    sample_data.append({
                        'problem': dataset_infos['problem_description'][i],
                        'solver_response': resp[:500],
                        'score': score,
                    })
            
            metrics['val/solver_pass_rate'].append(pass_count / len(solver_responses))
        
        # Aggregate
        final_metrics = {}
        for k, v in metrics.items():
            final_metrics[k] = np.mean(v)
        
        print(f"\n[Validation] Solver pass rate: {final_metrics.get('val/solver_pass_rate', 0):.3f}")
        
        return final_metrics
