def _calculate_load(self, n_req, m_tok, kv_usage):
        load_model = self.load_model

        # Decode-side 3D load model (single concentrated reference block).
        #
        # Assumptions:
        # 1) Pure decode window: n_req = running decode requests, m_tok = running decode tokens.
        # 2) Peak hardware constants are used (peak_tflops, bandwidth_gbs), so outputs are
        #    pressure scores for online scheduling instead of strict hardware-utilization values.
        #
        # Symbols:
        # H = hidden_size, I = intermediate_size, L = num_hidden_layers,
        # V = vocab_size,
        # K = kv_dim, b = precision_bytes,
        # BW = bandwidth_gbs * 1e9 / 1000 (bytes/ms), TPOT in ms/token.
        #
        # 1) Compute pressure
        #    Original (no aggregated coeff symbols):
        #    Attention fixed part (QKV + O projections):
        #    attn_proj_params = H*(H + 2*K) + H*H
        #    说明: 单层注意力线性层参数，含 QKV 投影和 O 投影。
        #    per_req_attn_proj_flops = 2 * L * attn_proj_params
        #    说明: 每个请求每步都要做的注意力线性投影计算，2 表示乘加近似 2 FLOPs。
        #
        #    MLP part:
        #    mlp_params = 3 * H * I
        #    说明: 单层 MLP 近似参数规模（门控结构按 3HI 记）。
        #    per_req_mlp_flops = 2 * L * mlp_params
        #    说明: 每个请求每步都要做的 MLP 线性计算。
        #
        #    Output head part:
        #    per_req_vocab_flops = 2 * V * H
        #    说明: 输出 logits 线性层近似开销。
        #
        #    Attention context-length part:
        #    attn_ctx_flops = 4 * L * H * m_tok
        #    说明: 对历史 token 的 QK^T + AV 代价，随 m_tok 线性增长。
        #
        #    Combined original form:
        #    per_req_linear_flops = per_req_attn_proj_flops + per_req_mlp_flops + per_req_vocab_flops
        #    FLOPs = n_req * per_req_linear_flops + attn_ctx_flops
        #    说明: 总计算量拆成“请求数驱动固定成本 + 上下文长度驱动成本”。
        #
        #    Define aggregated coefficients used in code:
        #    layer_params = attn_proj_params + mlp_params = H*(H + 2*K) + H*H + 3*H*I
        #    说明: 与代码中的 layer_params 定义保持一致。
        #    linear_flops_coeff = 2 * (L*layer_params + V*H)
        #    说明: 把 Attention 固定部分 + MLP + 输出头聚合成 n_req 的统一系数。
        #    attn_flops_coeff = 4 * L * H
        #    说明: 把 Attention 长度相关部分聚合成 m_tok 的统一系数。
        #
        #    Coefficient form (implementation):
        #    FLOPs = n_req * linear_flops_coeff + m_tok * attn_flops_coeff
        #    说明: 与原始式等价，但在线计算只需两次乘法和一次加法。
        #    t_compute_ms = (FLOPs / 1e12) / peak_tflops * 1000
        #    说明: 先把 FLOPs 转 TFLOPs，再按峰值算理论耗时并换算为毫秒。
        #    Load_Compute = t_compute_ms / TPOT
        #    说明: 用计算耗时除以每 token 时间预算，得到无量纲计算压力。
        #
        # 2) Memory/working-set pressure
        #    Original form:
        #    kv_vram_bytes = 2 * L * m_tok * K * b
        #    说明: K/V 两路缓存、每层都保留、每元素按 b 字节计，且随 m_tok 线性增长。
        #    activation_vram_bytes = n_req * L * (2 * H + 2 * I) * b
        #    说明: 近似把 decode 中与请求并发相关的激活/中间态按层累加。
        #    total_dynamic_io_bytes = kv_vram_bytes + activation_vram_bytes
        #    说明: 将 token 相关和请求相关的动态内存工作集合并成总压力项。
        #
        #    Denominator and final pressure:
        #    memory_denominator = BW * TPOT - weights_vram
        #    说明: BW*TPOT 是单 token 时间窗内可供搬运预算，减去权重常驻占用后的有效余量。
        #    Load_Memory = total_dynamic_io_bytes / memory_denominator
        #    说明: 用动态工作集与可用预算做比值，得到无量纲内存压力。
        #
        # 3) Capacity pressure
        #    Load_Memory_Capacity = kv_usage
        #    说明: 直接采用运行时 KV cache 使用率，刻画容量接近上限的风险。
        #
        # Bottleneck score
        #    Load_Bottle = max(Load_Compute, Load_Memory, Load_Memory_Capacity)
        #    说明: 取三者最大值作为瓶颈负载，符合短板决定吞吐的调度直觉。
        h_dim = load_model.config.hidden_size
        i_dim = load_model.config.intermediate_size
        l_dim = load_model.config.num_hidden_layers

        bw_bytes_ms = (load_model.bandwidth_gbs * 1e9) / 1000

        kv_vram_bytes = (
            2 * l_dim * m_tok * load_model.kv_dim * load_model.precision_bytes
        )

        activation_vram_bytes = (
            n_req * l_dim * (2 * h_dim + 2 * i_dim) * load_model.precision_bytes
        )

        flops = (n_req * load_model.linear_flops_coeff + load_model.attn_flops_coeff * m_tok)
        t_compute = (flops / 1e12) / load_model.peak_tflops * 1000

        load_compute = t_compute / load_model.tpot

        memory_denominator = bw_bytes_ms * load_model.tpot - load_model.weights_vram
        total_dynamic_io_bytes = kv_vram_bytes + activation_vram_bytes

        load_memory = (
            total_dynamic_io_bytes / memory_denominator
            if memory_denominator > 0
            else float("inf")
        )

        load_memory_capacity = kv_usage

        return {
            "Load_Compute": load_compute,
            "Load_Memory": load_memory,
            "Load_Memory_Capacity": load_memory_capacity,
            "Load_Bottle": max(load_compute, load_memory, load_memory_capacity),
        }