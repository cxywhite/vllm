# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import regex as re
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine)
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)
# [WT]delay kvcache transfer 2025-12-27 22:06:19
import zmq
import threading
import time
import re
from collections import deque
@dataclass
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor
# [WT] end
@dataclass
class ReqMeta:
    # Request Id
    request_id: str
    # Request block ids
    block_ids: torch.Tensor
    # Request num tokens
    num_tokens: int

    @staticmethod
    def make_meta(request_id: str, token_ids: list[int], block_ids: list[int],
                  block_size: int) -> "ReqMeta":
        block_ids_tensor = torch.tensor(block_ids)
        return ReqMeta(
            request_id=request_id,
            block_ids=block_ids_tensor,
            num_tokens=len(token_ids),
        )


@dataclass
class P2pNcclConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(request_id, token_ids, block_ids, block_size))

class P2pNcclConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        # logger.info(f'[WT]+++ P2pNcclConnector Init')
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer
        self.chunked_prefill: dict[str, Any] = {}

        self._rank = get_world_group().rank \
            if role == KVConnectorRole.WORKER else 0
        self._local_rank = get_world_group().local_rank \
            if role == KVConnectorRole.WORKER else 0

        self.p2p_nccl_engine = P2pNcclEngine(
            local_rank=self._local_rank,
            config=self.config,
            hostname="",
            port_offset=self._rank,
        ) if role == KVConnectorRole.WORKER else None
        # [WT]delay kvcache transfer 2025-12-27 21:47:07
        self.defer_kv_send = True
        # KV 缓存
        self.pending_layers: deque[SendQueueItem] = deque()
        #路由表：req_id -> decode_addr 每一项例如：{'cmpl-___prefill_addr_10.10.111.36:22010__db20a9c11814424abc9913fe7471112b-0': '10.10.111.36:22030'}
        self.routing_table = {}
        self.routing_lock = threading.Lock()
        # 启动 KV flush 线程
        self._start_kv_flush_thread()
        self.start_routing_listener("tcp://127.0.0.1:32324")
        # [WT] end
    # [WT]delay kvcache transfer 2025-12-27 21:48:52
    def start_routing_listener(self, zmq_addr: str):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(zmq_addr)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        def listen():
            while True:
                msg = sock.recv_string()
                # 格式：request_id|ip:port 例如：cmpl-___prefill_addr_10.10.111.36:22010__58241b5d91a941038a5e948b36b3f35c-0|10.10.111.36:22030
                req_id, decode_addr = msg.split("|", 1)
                with self.routing_lock:
                    self.routing_table[req_id] = decode_addr
                    logger.info(f"[WT]Received route message: {msg} p2p_nccl_connector.py")
        t = threading.Thread(target=listen, daemon=True)
        t.start()
    def _flush_ready_kv(self):
        if not self.pending_layers:
            return
        remaining = deque()
        while self.pending_layers:
            item = self.pending_layers.popleft()
            # 只经过prefill实例tensor_id 格式：req_id#layer_name 例如：tensor_id: cmpl-___prefill_addr_10.10.111.36:22010___decode_addr_10.10.111.36:22030_58241b5d91a941038a5e948b36b3f35c-0#model.layers.0.self_attn.attn
            req_id = item.tensor_id.split("#")[0]
            layer_name=item.tensor_id.split("#")[1]
            with self.routing_lock:
                # decode_addr 格式：ip:port 例如：10.10.111.36:22030
                decode_addr = self.routing_table.get(req_id)
            if decode_addr is None:
                remaining.append(item)
                continue
            else:
                logger.info(f'[WT]Flushing req_id: {req_id} to decode_addr: {decode_addr} p2p_nccl_connector.py')
                logger.info(f'[WT]routing_table: {self.routing_table} p2p_nccl_connector.py')
            ip, port = decode_addr.split(":")
            target = f"{ip}:{int(port) + self._rank}"
            separator = "__"
            last_index = req_id.rfind(separator)

            if last_index != -1:
                part1 = req_id[:last_index + len(separator)]  # 包含分隔符
                part2 = req_id[last_index + len(separator):]
            else:
                logger.error(f"[WT]Unexpected request_id format: {req_id} p2p_nccl_connector.py")
            # 经过proxy负载均衡后的 tensor_id 格式：cmpl- + ___prefill_addr_ + {prefill_ip:prefill_kv_port} + ___decode_addr_ + {decode_ip:decode_kv_port} + _ + uuid + -0 例如：tensor_id: cmpl-___prefill_addr_10.10.111.36:22010___decode_addr_10.10.111.36:22030_58241b5d91a941038a5e948b36b3f35c-0
            tensor_id= part1 +"_decode_addr_"+ decode_addr + "_" + part2+ "#" +layer_name
            logger.info(f'[WT] Sending tensor_id: {tensor_id} p2p_nccl_connector.py')
            self.p2p_nccl_engine.send_tensor(
                tensor_id=tensor_id,
                tensor=item.tensor,
                remote_address=target,
            )
        self.pending_layers = remaining
    def _start_kv_flush_thread(self):
        def loop():
            while True:
                self._flush_ready_kv()
                time.sleep(0.001)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
    # [WT] end
    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs: Any) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        # logger.info(f'[WT]+++ 5 P2pNcclConnector start_load_kv')
        # Only consumer/decode loads KV Cache
        if self.is_producer:
            return

        assert self.p2p_nccl_engine is not None

        attn_metadata = forward_context.attn_metadata
        # logger.info(f'[WT] Decode Instance attn_metadata {attn_metadata}')
        if attn_metadata is None:
            return

        def inject_kv_into_layer(
            layer: torch.Tensor,
            kv_cache: torch.Tensor,
            block_ids: torch.Tensor,
            request_id: str,
        ) -> None:
            """
            Inject KV cache data into a given attention layer tensor.

            This function updates `layer` in-place with values from `kv_cache`,
            handling different backend layouts:
              - MLA (Multi-Linear Attention) or FlashInfer: KV tensors are
                indexed along the first dimension.
              - FlashAttention: KV tensors are indexed along the second
                dimension.

            If the number of provided block IDs does not match the number of KV
            blocks, only the overlapping portion is updated, and a warning is
            logged.

            Args:
                layer (torch.Tensor): The attention layer KV tensor to update.
                kv_cache (torch.Tensor): The KV cache tensor to inject.
                block_ids (torch.Tensor): Indices of the blocks to update.
                request_id (str): Request identifier used for logging.

            Returns:
                None. The function modifies `layer` in-place.
            """
            # logger.info(f'[WT]+++ P2pNcclConnector start_load_kv inject_kv_into_layer')
            if (isinstance(attn_metadata, MLACommonMetadata)
                    or layer.shape[1] == 2):  # MLA or FlashInfer
                num_block = kv_cache.shape[0]
                self.check_tensors_except_dim(layer, kv_cache, 0)
                if len(block_ids) == num_block:
                    layer[block_ids, ...] = kv_cache
                else:
                    layer[block_ids[:num_block], ...] = kv_cache
                    logger.warning(
                        "🚧kv_cache does not match, block_ids:%d, "
                        "num_block:%d, request_id:%s", len(block_ids),
                        num_block, request_id)

            elif layer.shape[0] == 2:  # FlashAttention
                num_block = kv_cache.shape[1]
                self.check_tensors_except_dim(layer, kv_cache, 1)
                if len(block_ids) == num_block:
                    layer[:, block_ids, ...] = kv_cache
                else:
                    layer[:, block_ids[:num_block], ...] = kv_cache
                    logger.warning(
                        "🚧kv_cache does not match, block_ids:%d, "
                        "num_block:%d, request_id:%s", len(block_ids),
                        num_block, request_id)

        # Get the metadata
        metadata: KVConnectorMetadata = \
            self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadata)
        # logger.info(f'[WT] Decode Instance connector_metadata {metadata}')
        if metadata is None:
            return

        # Load the KV for each request each layer
        for request in metadata.requests:
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, False)
            remote_address = ip + ":" + str(port + self._rank)
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                # Only process layers that have kv_cache
                # attribute (attention layers) Skip non-attention
                # layers like FusedMoE
                kv_cache = getattr(layer, 'kv_cache', None)
                if kv_cache is None:
                    continue

                layer = kv_cache[forward_context.virtual_engine]

                kv_cache = self.p2p_nccl_engine.recv_tensor(
                    request.request_id + "#" + layer_name, remote_address)

                if kv_cache is None:
                    logger.warning("🚧kv_cache is None, %s", request.request_id)
                    continue

                inject_kv_into_layer(layer, kv_cache, request.block_ids,
                                     request.request_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        # logger.info(f'[WT]+++ 7 P2pNcclConnector wait_for_layer_load')
        return
    # [WT]delay kvcache transfer 2025-12-27 22:02:52
    # 修改原来save_kv_layer函数，仅缓存不发送
    def save_kv_layer(self,layer_name: str,kv_layer,attn_metadata,**kwargs,):
        if not self.is_producer:
            return
        def extract_kv_from_layer(
            layer: torch.Tensor,
            block_ids: torch.Tensor,
        ) -> torch.Tensor:
            """
            Extract KV cache slices from a given attention layer tensor.

            This function handles multiple backend layouts:
              - MLA (Multi-Linear Attention) or FlashInfer: KV tensors are
                indexed along the first dimension.
              - FlashAttention: KV tensors are indexed along the second
                dimension.

            Args:
                layer (torch.Tensor): The KV cache from the attention layer.
                block_ids (torch.Tensor): Indices of blocks to extract.

            Returns:
                torch.Tensor: A tensor containing the extracted KV slices.
                Returns None if the layout is unsupported.
            """
            # logger.info(f'[WT]+++ 12 P2pNcclConnector extract_kv_from_layer')
            if (isinstance(attn_metadata, MLACommonMetadata)
                    or layer.shape[1] == 2):  # MLA or FlashInfer
                return layer[block_ids, ...]
            if layer.shape[0] == 2:  # FlashAttention
                return layer[:, block_ids, ...]
            return None
        connector_metadata = self._get_connector_metadata()
        for request in connector_metadata.requests:
            kv_cache = extract_kv_from_layer(kv_layer, request.block_ids)
            # [WT][MOD]仅缓存，不发送
            self.pending_layers.append(
                SendQueueItem(request.request_id + "#" + layer_name,None,kv_cache)
            )
    # [WT] end


    # [WT]delay kvcache transfer 2025-12-27 22:04:22
    # 修改原来wait_for_save 函数，仅等待发送完成 
    def wait_for_save(self):
        # ================= 原始阻塞逻辑（废弃） =================
        # while self.pending_layers:
        #     self._try_send()
        # ======================================================
        # [WT][MOD]Prefill 阶段不阻塞
        if self.is_producer:
            if not self.defer_kv_send:
                assert self.p2p_nccl_engine is not None
                self.p2p_nccl_engine.wait_for_sent()
    # [WT] end
    
    def get_finished(
            self, finished_req_ids: set[str],
            **kwargs: Any) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        # logger.info(f'[WT]+++ P2pNcclConnector get_finished')
        assert self.p2p_nccl_engine is not None

        no_compile_layers = (
            self._vllm_config.compilation_config.static_forward_context)
        # logger.info(f'[WT]%%% P2pNcclConnector get_finished no_compile_layers:{no_compile_layers}')
        return self.p2p_nccl_engine.get_finished(finished_req_ids,
                                                 no_compile_layers)

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.is_producer:
            return 0, False

        num_external_tokens = (len(request.prompt_token_ids) - 1 -
                               num_computed_tokens)

        if num_external_tokens < 0:
            num_external_tokens = 0

        return num_external_tokens, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request, blocks.get_block_ids()[0])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        # logger.info(f'[WT]+++ 2 P2pNcclConnector build_connector_meta')
        meta = P2pNcclConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens)[new_req.req_id]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                # the request's prompt is chunked prefill
                if num_tokens < len(new_req.prompt_token_ids):
                    # 'CachedRequestData' has no attribute 'prompt_token_ids'
                    self.chunked_prefill[new_req.req_id] = (
                        new_req.block_ids[0], new_req.prompt_token_ids)
                    continue
                # the request's prompt is not chunked prefill
                meta.add_request(request_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size)
                continue
            if new_req.req_id in self._requests_need_load:
                meta.add_request(request_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size)
                self._requests_need_load.pop(new_req.req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = cached_reqs.resumed_from_preemption[i]

            if self.is_producer:
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens)[req_id]
                num_tokens = (num_scheduled_tokens + num_computed_tokens)
                assert req_id in self.chunked_prefill
                block_ids = new_block_ids[0]
                if not resumed_from_preemption:
                    block_ids = (self.chunked_prefill[req_id][0] + block_ids)
                prompt_token_ids = self.chunked_prefill[req_id][1]
                # the request's prompt is chunked prefill again
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[req_id] = (block_ids,
                                                    prompt_token_ids)
                    continue
                # the request's prompt is all prefilled finally
                meta.add_request(request_id=req_id,
                                 token_ids=prompt_token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size)
                self.chunked_prefill.pop(req_id, None)
                continue

            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not resumed_from_preemption:
                break
            if req_id in self._requests_need_load:
                request, _ = self._requests_need_load.pop(req_id)
                total_tokens = num_computed_tokens + 1
                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                block_ids = new_block_ids[0]

                meta.add_request(request_id=req_id,
                                 token_ids=token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size)

        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """

        self.chunked_prefill.pop(request.request_id, None)

        return False, None

    # ==============================
    # Static methods
    # ==============================

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
        # Regular expression to match the string hostname and integer port
        # logger.info(f'[WT]+++ 11 P2pNcclConnector parse_request_id')
        if is_prefill:
            pattern = r"___decode_addr_(.*):(\d+)"
        else:
            pattern = r"___prefill_addr_(.*):(\d+)___"

        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the ranks
            ip = match.group(1)
            port = int(match.group(2))

            return ip, port
        raise ValueError(
            f"Request id {request_id} does not contain hostname and port")

    @staticmethod
    def check_tensors_except_dim(tensor1, tensor2, dim):
        shape1 = tensor1.size()
        shape2 = tensor2.size()

        if len(shape1) != len(shape2) or not all(
                s1 == s2
                for i, (s1, s2) in enumerate(zip(shape1, shape2)) if i != dim):
            raise NotImplementedError(
                "Currently, only symmetric TP is supported. Asymmetric TP, PP,"
                "and others will be supported in future PRs.")