# /vllm/vllm/v1/core/sched/aimd_scheduler.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager


from typing import Any, Dict, List, Iterable
from collections import defaultdict

# [WT]virture mem 2026-01-06 00:39:06
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request
from .aimd_queue import AIMDRequestQueue
import bisect
import numpy as np
import time
import math
logger = init_logger(__name__)

class AIMDScheduler(Scheduler):
    """Scheduler 的子类：在 schedule/update_from_output 中注入 AIMD 逻辑。"""

    def __init__(self,
                 vllm_config: Any,
                 kv_cache_config: Any,
                 structured_output_manager: Any,
                 *args,
                 **kwargs) -> None:
        # [WT]virture mem 2026-01-06 00:21:04
        super().__init__(vllm_config, kv_cache_config, structured_output_manager, *args, **kwargs)
        self.max_model_len=self.vllm_config.model_config.max_model_len
        self.physical_mem=self.vllm_config.cache_config.num_gpu_blocks * self.vllm_config.cache_config.block_size
        self.test_model=self.vllm_config.scheduler_config.test_model
        if self.test_model != "":
            logger.info(f'[test]*** test_model: {self.test_model}')
            if self.test_model == "qwen":
                self.virtual_mem=self.physical_mem*4
            elif self.test_model == "llama":
                self.virtual_mem=int(self.physical_mem*1.5)
        # self.virtual_mem=self.max_num_scheduled_tokens if self.max_num_scheduled_tokens>self.max_model_len else self.max_model_len*2
        # self.request_length_record:dict[str,Any]={}
        self.waiting: AIMDRequestQueue = AIMDRequestQueue()
        self.waiting_times=0
        self.running_now=True
        self.max_preempt_count=0
        self.physical_mem_free=self.physical_mem
        # [WT]qwen 2026-01-25 17:19:21
        self.no_preempt_time=0
    def schedule(self) -> SchedulerOutput:
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule the RUNNING requests.
        req_index = 0
        
        # [WT]virture mem 2026-01-05 20:42:56
        # TODO:用tokens粒度判断还是block粒度？
        # self.running_now=self.last_running_tokens+len(self.running) < self.physical_mem
        # if self.running_now:
        #     pass
        # else:
        #     # running中的请求按照num_computed_tokens/predictor_output_len从高到底排序，优先抢占那些比例较低的请求
        #     # logger.info(f"[test]111 running_now=False, self.last_running_tokens: {self.last_running_tokens}, len(self.running): {len(self.running)}, physical_mem: {self.physical_mem}")
        #     self.running.sort(key=lambda r: (r.num_computed_tokens / max(1,(r.num_prompt_tokens+r.predictor_meta['predict_output_len']-r.num_computed_tokens))), reverse=True)
        #     # self.running_now=False
        
        # for i in range(len(self.running)):
        #     logger.info(f"[test]1-1 Running request {self.running[i].request_id}, h:{self.running[i].num_computed_tokens/max(1,(self.running[i].num_prompt_tokens+self.running[i].predictor_meta['predict_output_len']-self.running[i].num_computed_tokens)):.4f}, num_computed_tokens: {self.running[i].num_computed_tokens}, num_prompt_tokens: {self.running[i].num_prompt_tokens}, predict_output_len: {self.running[i].predictor_meta['predict_output_len']}, preempt_count: {self.running[i].preempt_count}")
        if self.physical_mem_free / self.physical_mem <= 0.1:
            # logger.info(f'[test]111 physical_mem_free: {self.physical_mem_free}, physical_mem: {self.physical_mem}, ratio: {self.physical_mem_free / self.physical_mem:.4f} <= 0.1') 
            # timestart=time.perf_counter()
            self.running.sort(key=lambda r: (r.num_computed_tokens / max(1,(r.num_prompt_tokens+r.predictor_meta['predict_output_len']-r.num_computed_tokens))), reverse=True)
            # timeend=time.perf_counter()
            # logger.info(f'[test]222 self.running.sort time cost: {timeend - timestart} s')
            # for i in range(len(self.running)):
            #     logger.info(f"[test]2-2 Running request {self.running[i].request_id}, h:{self.running[i].num_computed_tokens/max(1,(self.running[i].num_prompt_tokens+self.running[i].predictor_meta['predict_output_len']-self.running[i].num_computed_tokens)):.4f}, num_computed_tokens: {self.running[i].num_computed_tokens}, num_prompt_tokens: {self.running[i].num_prompt_tokens}, predict_output_len: {self.running[i].predictor_meta['predict_output_len']}, preempt_count: {self.running[i].preempt_count}")
            # logger.info(f'self.running.sort time:{timeend-timestart}')
        tmp_request_length_record=[]
        while req_index < len(self.running) and token_budget > 0:
            # 打印采样参数
            # logger.info(f"[test]sampling_params: {self.running[req_index].sampling_params} ")
            #TODO Check if we have reached the max wait tokens.
            request = self.running[req_index]
            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_compute_budget
                 ) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_compute_budget)
            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue
            # logger.info(f'[test] num_new_tokens:{num_new_tokens} for running request {request.request_id}')
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()
                        # [WT]virture mem 2026-01-05 20:53:15
                        self.running_now=False
                        preempted_req.preempt_count+=1
                        self.max_preempt_count=max(self.max_preempt_count,preempted_req.preempt_count)
                        timenow=time.time()
                        preempted_req.preempt_latency.append(timenow)
                        # logger.info(f'[test]222 Preempting request {preempted_req.request_id}, preempt_count: {preempted_req.preempt_count}, max_preempt_count: {self.max_preempt_count}')
                        # logger.info(f'[test]333 Preempting request:{preempted_req.request_id} arrival time: {preempted_req.arrival_time} preempted at: {timenow} preempted count:{preempted_req.preempt_count}, max_preempt_count: {self.max_preempt_count}')
                        self.physical_mem_free+=preempted_req.num_computed_tokens
                        self.no_preempt_time=0
                        # if self.waiting_times>0:
                        #     self.waiting_times = max(self.waiting_times,1 << (len(preempted_reqs)+self.max_preempt_count))
                        #     logger.info(f'[test]999 {self.waiting_times}')
                        # running_predict=False
                    self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            # [WT] 2026-01-23 02:54:14
            self.physical_mem_free-=num_new_tokens
            # [WT]virture mem 2026-01-05 20:26:39
            # TODO:后续有机会改成从历史记录读取request_length_record
            # if self.request_length_record.get(request.request_id,None) is not None:
            #     tmp_request_length_record.append(self.request_length_record[request.request_id])
            tmp_request_length_record.append((request.num_computed_tokens,max(1,request.num_prompt_tokens+request.predictor_meta['predict_output_len']-request.num_computed_tokens),request.predictor_meta['predict_output_len']))
            req_index += 1
            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        # [WT]virture mem 2026-01-06 00:53:54
        # skipped_waiting_requests = create_request_queue(self.policy)
        skipped_waiting_requests = AIMDRequestQueue()
        # Next, schedule the WAITING requests.
        # [WT]virture mem 2026-01-06 00:11:27
        if self.waiting_times==0:
            if not preempted_reqs:
                self.max_preempt_count=0
                tmp_request_length_record.sort(key=lambda x: -x[1])
                while self.waiting and token_budget > 0:
                    if len(self.running) == self.max_num_running_reqs:
                        break
                    request = self.waiting.peek_request()
                    # KVTransfer: skip request if still waiting for remote kvs.
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        is_ready = self._update_waiting_for_remote_kv(request)
                        if is_ready:
                            request.status = RequestStatus.WAITING
                        else:
                            logger.debug(
                                "%s is still in WAITING_FOR_REMOTE_KVS state.",
                                request.request_id)
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue
                    # [WT]virture mem 2026-01-05 22:18:34
                    # time_start=time.perf_counter()
                    bisect.insort(tmp_request_length_record, (request.num_computed_tokens,max(1,request.num_prompt_tokens+request.predictor_meta['predict_output_len']-request.num_computed_tokens),request.predictor_meta['predict_output_len']), key=lambda x: -x[1])
                    left_out_len_array = np.array([e[1] for e in tmp_request_length_record])
                    has_run_len_array = np.array([e[0] for e in tmp_request_length_record])
                    cum_run_len_array = np.cumsum(has_run_len_array)
                    size_array = np.arange(1, len(tmp_request_length_record) + 1, 1)
                    need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
                    # time_end=time.perf_counter()
                    # for i in range(len(tmp_request_length_record)):
                    #     print(f"[test]*** waiting_request_length_record idx:{i} request_length_record: {tmp_request_length_record[i]} ")
                    # logger.info(f"[test]444 len(tmp_request_length_record): {len(tmp_request_length_record)} bisect time cost: {time_end - time_start} s")
                    if self.no_preempt_time > 3:
                        self.virtual_mem=int(self.virtual_mem * 1.05)
                        self.no_preempt_time=0
                    if need_max_token_num > self.virtual_mem and self.no_preempt_time <= 3:
                        self.no_preempt_time+=1
                        # logger.info(f'[test]555 peek memory > virtual_mem: {need_max_token_num} > {self.virtual_mem}')
                        break
                    # if need_max_token_num > self.virtual_mem:
                    #     logger.info(f'[test]555 peek memory > virtual_mem: {need_max_token_num} > {self.virtual_mem}')
                    #     break
                    
                    # Skip request if the structured output request is still waiting
                    # for FSM compilation.
                    if request.status == RequestStatus.WAITING_FOR_FSM:
                        structured_output_req = request.structured_output_request
                        if structured_output_req and structured_output_req.grammar:
                            request.status = RequestStatus.WAITING
                        else:
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    # Check that adding the request still respects the max_loras
                    # constraint.
                    if (self.lora_config and request.lora_request and
                        (len(scheduled_loras) == self.lora_config.max_loras and
                        request.lora_request.lora_int_id not in scheduled_loras)):
                        # Scheduling would exceed max_loras, skip.
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_external_computed_tokens = 0
                    load_kv_async = False

                    # Get already-cached tokens.
                    if request.num_computed_tokens == 0:
                        # Get locally-cached tokens.
                        new_computed_blocks, num_new_local_computed_tokens = \
                            self.kv_cache_manager.get_computed_blocks(
                                request)

                        # Get externally-cached tokens if using a KVConnector.
                        if self.connector is not None:
                            num_external_computed_tokens, load_kv_async = (
                                self.connector.get_num_new_matched_tokens(
                                    request, num_new_local_computed_tokens))

                            if num_external_computed_tokens is None:
                                # The request cannot be scheduled because
                                # the KVConnector couldn't determine
                                # the number of matched tokens.
                                self.waiting.pop_request()
                                skipped_waiting_requests.prepend_request(request)
                                continue

                        # Total computed tokens (local + external).
                        num_computed_tokens = (num_new_local_computed_tokens +
                                            num_external_computed_tokens)
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    else:
                        new_computed_blocks = (
                            self.kv_cache_manager.create_empty_block_list())
                        num_new_local_computed_tokens = 0
                        num_computed_tokens = request.num_computed_tokens

                    encoder_inputs_to_schedule = None
                    new_encoder_compute_budget = encoder_compute_budget

                    # KVTransfer: loading remote KV, do not allocate for new work.
                    if load_kv_async:
                        assert num_external_computed_tokens > 0
                        num_new_tokens = 0
                    # Number of tokens to be scheduled.
                    else:
                        # We use `request.num_tokens` instead of
                        # `request.num_prompt_tokens` to consider the resumed
                        # requests, which have output tokens.
                        num_new_tokens = request.num_tokens - num_computed_tokens
                        if (0 < self.scheduler_config.long_prefill_token_threshold
                                < num_new_tokens):
                            num_new_tokens = (
                                self.scheduler_config.long_prefill_token_threshold)

                        # chunked prefill has to be enabled explicitly to allow
                        # pooling requests to be chunked
                        if not self.scheduler_config.chunked_prefill_enabled and \
                            num_new_tokens > token_budget:
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        num_new_tokens = min(num_new_tokens, token_budget)
                        assert num_new_tokens > 0

                        # Schedule encoder inputs.
                        if request.has_encoder_inputs:
                            (encoder_inputs_to_schedule, num_new_tokens,
                            new_encoder_compute_budget
                            ) = self._try_schedule_encoder_inputs(
                                request, num_computed_tokens, num_new_tokens,
                                encoder_compute_budget)
                            if num_new_tokens == 0:
                                # The request cannot be scheduled.
                                break

                    # Handles an edge case when P/D Disaggregation
                    # is used with Spec Decoding where an
                    # extra block gets allocated which
                    # creates a mismatch between the number
                    # of local and remote blocks.
                    effective_lookahead_tokens = (0 if request.num_computed_tokens
                                                == 0 else
                                                self.num_lookahead_tokens)

                    # Determine if we need to allocate cross-attention blocks.
                    if self.is_encoder_decoder and request.has_encoder_inputs:
                        # TODO(russellb): For Whisper, we know that the input is
                        # always padded to the maximum length. If we support other
                        # encoder-decoder models, this will need to be updated if we
                        # want to only allocate what is needed.
                        num_encoder_tokens =\
                            self.scheduler_config.max_num_encoder_input_tokens
                    else:
                        num_encoder_tokens = 0

                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens + num_external_computed_tokens,
                        num_new_local_computed_tokens,
                        new_computed_blocks,
                        num_lookahead_tokens=effective_lookahead_tokens,
                        delay_cache_blocks=load_kv_async,
                        num_encoder_tokens=num_encoder_tokens,
                    )

                    if new_blocks is None:
                        # The request cannot be scheduled.
                        break

                    # KVTransfer: the connector uses this info to determine
                    # if a load is needed. Note that
                    # This information is used to determine if a load is
                    # needed for this request.
                    if self.connector is not None:
                        self.connector.update_state_after_alloc(
                            request,
                            new_computed_blocks + new_blocks,
                            num_external_computed_tokens,
                        )

                    # Request was already popped from self.waiting
                    # unless it was re-added above due to new_blocks being None.
                    request = self.waiting.pop_request()
                    # if request is not None and request.predictor_meta:
                    #     logger.info(f'WAITING request {request.predictor_meta}')
                    if load_kv_async:
                        # If loading async, allocate memory and put request
                        # into the WAITING_FOR_REMOTE_KV state.
                        skipped_waiting_requests.prepend_request(request)
                        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                        continue

                    req_index += 1
                    self.running.append(request)
                    
                    if self.log_stats:
                        request.record_event(EngineCoreEventType.SCHEDULED,
                                            scheduled_timestamp)
                    if request.status == RequestStatus.WAITING:
                        scheduled_new_reqs.append(request)
                    elif request.status == RequestStatus.PREEMPTED:
                        scheduled_resumed_reqs.append(request)
                    else:
                        raise RuntimeError(
                            f"Invalid request status: {request.status}")

                    if self.lora_config and request.lora_request:
                        scheduled_loras.add(request.lora_request.lora_int_id)
                    req_to_new_blocks[request.request_id] = (
                        self.kv_cache_manager.get_blocks(request.request_id))
                    num_scheduled_tokens[request.request_id] = num_new_tokens
                    token_budget -= num_new_tokens
                    
                    # logger.info(f'[test]666 Scheduling waiting request {request.request_id} for {num_new_tokens} tokens.')
                    self.physical_mem_free-=num_new_tokens
                    request.status = RequestStatus.RUNNING
                    request.num_computed_tokens = num_computed_tokens
                    # Count the number of prefix cached tokens.
                    if request.num_cached_tokens < 0:
                        request.num_cached_tokens = num_computed_tokens
                    # Encoder-related.
                    if encoder_inputs_to_schedule:
                        scheduled_encoder_inputs[request.request_id] = (
                            encoder_inputs_to_schedule)
                        # Allocate the encoder cache.
                        for i in encoder_inputs_to_schedule:
                            self.encoder_cache_manager.allocate(request, i)
                        encoder_compute_budget = new_encoder_compute_budget
                
            # [WT]virture mem 2026-01-05 22:45:47
            else:
                if self.virtual_mem>=self.physical_mem:
                    # logger.info(f'[test]666 skip scheduling waiting requests due to virtual_mem > physical_mem: {self.virtual_mem} > {self.physical_mem}')
                    # self.virtual_mem=int(self.physical_mem * 0.9)
                    self.virtual_mem=self.physical_mem
                
                # 修改成2的len(preempted_reqs)幂次
                waiting_power=min(max((len(preempted_reqs),self.max_preempt_count)),10)
                self.waiting_times = max(self.waiting_times,waiting_power)
                self.no_preempt_time=0
                # self.waiting_times = max(self.waiting_times,1 << max(len(preempted_reqs),self.max_preempt_count))
                # logger.info(f'[test]777 have preempt waiting_times set to: {self.waiting_times}')
                # self.waiting_times = math.ceil(len(preempted_reqs) / 2)
        # [WT]virture mem 2026-01-06 00:13:11
        else:
            if self.waiting_times > 0:
                self.waiting_times-=1
                # logger.info(f'[test]888 skip schedule waiting waiting_times: {self.waiting_times}')
        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        scheduled_requests = (scheduled_new_reqs + scheduled_running_reqs +
                              scheduled_resumed_reqs)
        structured_output_request_ids, grammar_bitmask = (
            self.get_grammar_bitmask(scheduled_requests,
                                     scheduled_spec_decode_tokens))
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.
            get_freed_mm_hashes(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        return scheduler_output
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None
        kv_connector_stats = (kv_connector_output.kv_connector_stats
                              if kv_connector_output else None)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                request.num_computed_tokens -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids)

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len,
                                     pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                    # [WT]virture mem 2026-01-05 22:54:58
                    self.physical_mem_free+= request.num_computed_tokens
                    # logger.info(f'test]999 Stopping running request {request.request_id}, freeing up num_computed_tokens: {request.num_computed_tokens}, num_prompt_tokens: {request.num_prompt_tokens}, len(request._output_token_ids): {len(request._output_token_ids)}')
                    if self.waiting_times==0:
                        if self.test_model=="qwen":
                            self.virtual_mem=int(self.virtual_mem * 1.1)
                        elif self.test_model=="llama":
                            self.virtual_mem=int(self.virtual_mem * 1.01)

                        # if self.virtual_mem>self.physical_mem:
                        #     self.virtual_mem=self.physical_mem
                        # else:
                        #     self.virtual_mem=int(self.virtual_mem * 1.01)
                            # self.virtual_mem+=request.num_computed_tokens
                        # logger.info(f'[test]1212 virtual_mem: {self.virtual_mem}, physical_mem: {self.physical_mem}')
                    self.running_now=True
                    # if self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_role == 'kv_consumer':
                    #     if len(request.preempt_latency)>0:
                    #         timestop=time.time()
                            # logger.info(f'[test]1313 Request {request.request_id} finished. Total time: {timestop - request.arrival_time:.2f} seconds, arrival time: {request.arrival_time:.2f}, preempted time: {request.preempt_latency}, preempted count: {request.preempt_count}, stop time: {timestop}')
                else:
                    stopped_preempted_reqs.add(request)
            # Extract sample logprobs if needed.
            if request.sampling_params is not None \
                and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # checked above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None \
                or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    ))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if model_runner_output.kv_connector_output:
            self._update_from_kv_xfer_finished(
                model_runner_output.kv_connector_output)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()
        # [WT]prometheus 2026-01-02 21:14:52
        if self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_role == 'kv_consumer':
            self.running_tokens = 0
            self.running_predict_tokens = 0
            self.waiting_tokens = 0
            for req in self.running:
                self.running_tokens+= req.num_computed_tokens
                self.running_predict_tokens += max(1, req.num_prompt_tokens+req.predictor_meta['predict_output_len']-req.num_computed_tokens)
            for req in self.waiting:
                self.waiting_tokens =self.waiting_tokens+req.num_prompt_tokens+req.predictor_meta['predict_output_len']
        if (stats := self.make_stats(spec_decoding_stats,
                                     kv_connector_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    
