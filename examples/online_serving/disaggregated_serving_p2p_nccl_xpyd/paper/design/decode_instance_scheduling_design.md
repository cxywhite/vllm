# Decode 实例内调度设计

## 一句话结论

Decode 实例内调度的核心，不是为每个 waiting 请求学习一个黑盒打分函数，而是保留预测器输出的 5 桶长度分布，用它推导请求准入后的短视界资源追加需求与三维负载变化，并在具体运行时的离散分配语义下用物理约束执行动态 admission control；对 vLLM 而言，这个离散分配单元具体体现为 KV blocks。

---

## Decode 实例内调度设计逻辑

### §1 Decode 实例内调度的核心决策不是“当前能不能塞下”，而是“现在准入后未来会不会越过瓶颈”

**首句**：Decode 实例内调度的核心问题，不是当前时刻是否还能临时容纳一个 waiting 请求，而是把它放入 running 后，实例在未来短视界内是否仍处在可接受的资源工作区间内。

**展开**：
- 现有 motivation 已经说明，decode goodput 存在稳定的倒 U 型并发规律，实例内并发并不是越高越好。
- 这意味着 waiting 请求的准入决策不能只看“当前还剩多少空间”，而必须看“这个请求进来以后，会不会在接下来的若干 decode step 中把系统推过 compute、memory bandwidth 或 KV capacity 的瓶颈点”。
- 因而，decode 实例内调度的本质是一个未来风险判断问题：在资源利用率提升和未来 preemption 风险之间，决定 waiting 请求是否应立即进入 running batch。

### §2 既然调度器关心未来风险，它需要的就不是点预测，而是能表达右尾不确定性的长度分布

**首句**：既然最优并发点依赖于请求未来输出长度风险，那么调度器真正需要的不是一个单点长度估计，而是能表达右尾和长尾概率的分布信息。

**展开**：
- 如果调度器只知道一个请求的均值长度，它至多只能知道“平均来看它不短”，却无法知道它是否有较大概率落入长尾区。
- 但 decode admission 真正关心的是：这个请求在未来若干步内是否仍然活跃、是否会继续新增 KV 相关资源、以及它与现有 running 请求叠加后是否会把实例推向更高的 preemption 风险区。
- 因此，从调度问题本身出发，最自然的上游输入不是点预测，而是能直接表达右尾风险的长度分布。

### §3 对当前 predictor 而言，5 桶概率分布应被直接保留，而不是过早压缩为均值和方差

**首句**：当预测器已经输出 5 个长度区间的概率分布时，最合理的做法是保留这 5 桶分布作为上游风险表示，而不是一开始就把它压缩为均值和方差。

**展开**：
- 若预测器输出

$$
([0,a], [a,b], [b,c], [c,d], [d,L_{max}]) \mapsto (p_1,p_2,p_3,p_4,p_5),
$$

它已经明确给出了请求落入短输出、中等输出和长尾输出区间的概率质量。
- 从这 5 桶分布当然可以进一步计算近似均值和方差，但这只是一个有损压缩；一旦只保留 $(\mu, \sigma^2)$，很多对调度最重要的右尾信息就会被抹平。
- 因此，5 桶分布应作为 predictor 和 scheduler 之间的基础接口保留下来；均值和方差可以作为辅助特征存在，但不应成为调度器的唯一输入。

### §4 在采用离散资源追加分配的系统中，长度分布不能直接接到 full-horizon 峰值显存预留上，而必须先转换成短视界追加风险

**首句**：只要底层运行时不是一次性为请求未来全部输出预留资源，而是采用某种离散分配单元按需追加资源，长度分布就不应直接用于最终峰值显存预留，而应先被映射成未来短视界内的资源追加风险。

**展开**：
- Past-Future 风格的方法之所以在你的 vLLM 复现中容易过于保守，关键不在于它用了长度预测，而在于它把长度预测直接接到了 full-horizon peak memory 这个过于超前的决策变量上。
- 更一般地说，只要系统采用的是“资源用到哪里就追加到哪里”的分配策略，调度器真正应关心的就不是“这个请求最终会不会很长”，而是“它在未来若干步内会不会和现有 running 请求一起快速消耗当前仍可追加的资源单元”。
- 在 vLLM 中，这个离散资源单元具体就是 KV block；在其他运行时里，它也可以是 page、chunk、slab，或任何等价的最小追加分配粒度。
- 这就要求我们把 5 桶长度分布重新映射成更贴近运行时语义的短视界风险量，而不是直接把它理解成一个最终总长度上界。

### §5 这一步映射的核心不是总长度，而是未来 $H$ 步内的新增资源单元需求

**首句**：对 decode 实例，最相关的未来风险量不是请求结束时的总长度，而是它在未来 $H$ 个 decode step 内大概率会新增多少离散资源单元；对 vLLM 而言，这些资源单元就是 KV blocks。

**展开**：
- 对一个请求来说，调度器最需要知道的不是它最终总共会生成多少 token，而是它在当前状态下，距离下一个分配边界还有多远，以及在未来 $H$ 步内大概率会跨过多少个分配边界。
- 因而，更贴近调度的风险量应写成

$$
\Delta Q_H = \text{未来 } H \text{ 步内新增离散资源单元需求}.
$$

- 与最终总占用相比，$\Delta Q_H$ 更符合按需追加分配的运行时路径，因为它直接决定近期是否容易触发 allocate failure、eviction 或 preemption。
- 在 vLLM 中，可直接把 $Q$ 实例化为 block，因此原先的 block-aware 写法只是这套更一般设计的一个具体实现。

### §6 从 5 桶分布到短视界追加风险，最自然的做法是先估计“未来每一步请求是否仍然活跃”

**首句**：从 5 桶分布构造短视界追加风险时，最自然的中间量不是一个综合分数，而是请求在未来每一个 step 上仍然活跃的概率；而在实现上，这个量不必逐步显式滚动模拟，也可以写成闭式近似。

**展开**：
- 设请求 $i$ 的未来剩余输出长度分布由 5 个 bucket 给出，bucket $k$ 的代表剩余长度为 $\bar{l}_k$，对应概率为 $p_{i,k}$。
- 那么在未来第 $h$ 个 decode step 上，请求仍然活跃的近似概率可以写成：

$$
A_i(h) = \sum_{k=1}^{5} p_{i,k} \cdot \mathbf{1}[\bar{l}_k \ge h], \quad h = 1,\dots,H.
$$

- 基于这个活跃概率序列，请求在未来 $H$ 步内预计新增的 token 数可以写成：

$$
\hat{y}_i^{H} = \sum_{h=1}^{H} A_i(h).
$$

- 由于上式可以交换求和次序，因此它也可写成一个更容易实现的闭式形式：

$$
\hat{y}_i^{H} = \sum_{k=1}^{5} p_{i,k} \cdot \min(H, \bar{l}_k).
$$

- 进一步，若当前请求距离下一个资源分配边界还剩 $d_i$ 个 token、单个离散分配单元可容纳的 token 数为 $S_{alloc}$，则其在未来 $H$ 步内的新增资源单元需求可近似写成：

$$
\hat{\Delta Q}_i^{H} = \left\lceil \frac{\max(0, \hat{y}_i^{H} - d_i)}{S_{alloc}} \right\rceil.
$$

- 这条映射路径的优点是：它直接把 5 桶分布转换成运行时最关心的“未来资源追加需求”，而不需要引入额外的黑盒权重。
- 更重要的是，running 请求的动态更新并不要求显式维护整个未来轨迹；每次调度时只需要基于“当前剩余分布 + 当前边界距离 $d_i$”重新计算一次即可。

### §7 为了减少超参数，bucket 代表值和短视界窗口都应尽量绑定到有物理意义的量

**首句**：如果本文偏系统论文风格，那么算法中保留的少数参数也应尽量来自物理含义或离线统计，而不是自由调参；其中最需要解释清楚的就是短视界窗口 $H$。

**展开**：
- bucket 的代表值 $\bar{l}_k$ 最好优先使用离线统计得到的 bucket 条件均值；如果暂时没有足够统计量，再退化为 bucket 中点，而最后一个长尾 bucket 则可以使用更保守的右端点或经验条件均值。
- 短视界窗口 $H$ 不应被解释成一个随意调出来的超参数，而更适合作为一个物理窗口。对本文而言，一个自然定义是：

$$
H = \min(H_{alloc}, H_{budget}),
$$

其中：
- $H_{alloc}$ 表示一个最小资源追加窗口。若系统以离散分配单元按需追加资源，则 $H_{alloc}$ 取“跨过一个分配单元所需的 decode step 数”；对 vLLM 而言，可直接取一个 block 对应的 token 容量 $S_{blk}$。
- $H_{budget}$ 表示一个等待预算窗口，可写成

$$
H_{budget} = \left\lfloor \frac{T_{budget} - T_{wait}}{TPOT_{target}} \right\rfloor,
$$

即请求在不违反等待侧预算前，大致还允许观察多少个 decode step。
- 这样定义后，$H$ 既不会超过一个自然的资源追加窗口，也不会超过请求自身等待预算允许的局部未来。
- 若论文中希望更简洁，也可以先采用更保守、无额外调参的单窗口版本：$H = H_{alloc}$，把它作为默认实现，而将预算截断作为可选增强。
- 更重要的是，第一版设计里最好不要同时让 $H$ 和风险保守度都随状态动态变化；否则方法会显得过于灵活、难以解释。更稳妥的做法是：把 $H$ 固定为物理窗口，把“空闲时更激进、繁忙时更保守”的适配全部交给风险估计部分完成。
- 这样做以后，算法中真正需要解释的量就会收敛为少数几个有物理意义的变量，而不是一串难以解释的打分权重。

### §8 三维负载较空闲时偏激进、较饱和时偏保守，是一个合理且自然的设计方向

**首句**：让调度器在三维负载较空闲时更激进、在三维负载较饱和时更保守，是一个合理的设计方向，而且可以直接用当前瓶颈负载驱动，而不需要额外引入黑盒超参数。

**展开**：
- 设当前实例的瓶颈负载为

$$
L_{bottle} = \max(L_C, L_M, L_K),
$$

相应的瓶颈 slack 为

$$
S_{bottle} = 1 - L_{bottle}.
$$

- 这里需要特别说明：根据现有 modeling 定义，$L_C$ 和 $L_M$ 本来就可能大于 1，因为它们表示的是“相对 TPOT 预算”或“相对有效带宽预算”的压力值，而不是严格限制在 $[0,1]$ 内的利用率。
- 因此，$L=1$ 更准确地说表示“达到预算前沿”或“达到目标工作点”，而 $L>1$ 则表示该维度已经进入过载区，意味着若仍维持当前状态，系统更容易出现 TPOT 违反、带宽拥塞或后续 preemption/排队放大。
- 当 $L_{bottle}$ 较小、即实例仍处在明显利用不足区间时，调度器可以更偏激进，因为此时 admission 的主要目标是提升并行度和资源利用率。
- 当 $L_{bottle}$ 接近 1、即实例已接近 compute、memory bandwidth 或 KV capacity 的瓶颈边界时，调度器则应更偏保守，因为此时一个额外请求更容易把系统推向 preemption 或 allocation failure 区域。
- 当 $L_{bottle} > 1$ 时，实例已经处在过载区。对 admission 而言，这并不表示模型失效，而表示策略应进入最保守状态：除非等待预算触发 override，否则不应再主动扩大并发。
- 这里的关键是：这种“激进/保守”切换完全可以由现有三维负载值直接驱动，而不需要再训练一个策略或引入额外打分权重。

### §9 更系统化的做法不是改变 H，而是用瓶颈负载决定分布解释得有多保守

**首句**：如果希望在空闲时偏激进、在繁忙时偏保守，一个更适合系统论文的做法不是动态改变 $H$，而是保持 $H$ 为固定物理窗口，并用当前瓶颈负载决定 5 桶分布被解释得有多保守。

**展开**：
- 设第 $k$ 个 bucket 的经验条件均值为 $\bar{l}_k^{mean}$，右端点或保守代表值为 $\bar{l}_k^{upper}$。
- 由于 $L_{bottle}$ 可能大于 1，真正进入插值的应是一个截断后的保守度系数，而不是原始负载值本身。定义：

$$
\rho = \min(1, L_{bottle}).
$$

- 那么在当前状态下，可定义一个由瓶颈负载驱动的 bucket 代表值：

$$
	ilde{l}_{i,k} = (1 - \rho) \cdot \bar{l}_k^{mean} + \rho \cdot \bar{l}_k^{upper}.
$$

- 当实例较空闲时，$L_{bottle}$ 较小，$\rho$ 也较小，因此 $\tilde{l}_{i,k}$ 更接近条件均值，风险估计更激进。
- 当实例较饱和时，$L_{bottle}$ 接近 1，$\rho$ 接近 1，因此 $\tilde{l}_{i,k}$ 更接近上界代表值，风险估计更保守。
- 当 $L_{bottle} > 1$ 时，$\rho$ 被截断到 1，算法自动进入“完全保守解释”状态，而不会继续做不合理的线性外推。
- 这样一来，算法仍然只依赖具有物理含义的量：当前瓶颈负载本身，而不需要再引入“风险温度”“保守系数”之类的新超参数。

### §10 Decode admission 的状态表示应直接围绕三维负载、可追加资源余量和等待预算展开

**首句**：在方法设计上，decode admission 的状态不应写成抽象打分向量，而应直接写成当前三维负载、可追加资源余量、短视界追加风险和等待预算这些有物理意义的量。

**展开**：
- 设当前实例状态为

$$
s_t = (B_{free}, R_t, W_t, T_{wait}, T_{budget}, L_C, L_M, L_K),
$$

其中 $B_{free}$ 是当前 free resource units 数，$R_t$ 和 $W_t$ 分别是 running 与 waiting 队列状态，$T_{wait}$ 与 $T_{budget}$ 则分别表示请求当前等待时间和等待预算，$L_C, L_M, L_K$ 是当前三维负载。
- 对每个请求 $i$，还需要维护当前位置相关的局部量，例如当前已占用资源单元数、已生成长度以及距离下一个分配边界的 token 距离 $d_i$；对 vLLM 而言，这个分配边界具体就是下一个 KV block 边界。
- 这些量要么来自现有三维负载模型，要么来自运行时调度器可直接观测的状态，因此比一个综合黑盒分数更适合系统论文叙述。

### §11 如果面向系统论文，最终准入规则最好直接写成“物理约束是否满足”

**首句**：对 queue head 请求做 decode admission 时，最终规则最适合直接写成“准入后三维负载与短视界资源追加需求是否仍在预算内”，而不是写成带多组权重的综合打分函数。

**展开**：
- 对 queue head 请求 $j$，设将其放入 running 后，未来短视界内的三维负载预测值为 $\hat{L}_C^{+}(j), \hat{L}_M^{+}(j), \hat{L}_K^{+}(j)$，对应的短视界新增资源单元需求保守估计为 $\hat{\Delta Q}_H^{+}(j)$。
- 则一个自然的准入规则可以写成：

$$
\text{Admit}(j) \iff
\hat{L}_C^{+}(j) \le 1,\;
\hat{L}_M^{+}(j) \le 1,\;
\hat{L}_K^{+}(j) \le 1,
$$

并同时满足：

$$
\hat{\Delta Q}_H^{+}(j) \le B_{free}.
$$

- 这里 $L_C, L_M, L_K \le 1$ 表示准入后不超过各自归一化后的资源预算，$\hat{\Delta Q}_H^{+}(j) \le B_{free}$ 则表示短期追加需求不超过当前可用余量。
- 这类判据的优势在于：每一个阈值都可以被解释为资源预算，而不是来自训练或网格搜索的抽象超参数。
- 这也意味着：$L>1$ 不是非法值，而是“已经越过预算前沿”的状态标志。策略把它当作 admission 的停止信号，而不是把它当作建模异常。
- 在这个框架下，“空闲时更激进、繁忙时更保守”并不需要额外改写准入条件本身，而可以通过前面的 $\tilde{l}_{i,k}$ 自然体现出来：同样一个 waiting 请求，在低负载状态下会得到更乐观的 $\hat{\Delta Q}_H^{+}(j)$，而在高负载状态下会得到更保守的 $\hat{\Delta Q}_H^{+}(j)$。

### §12 等待代价也应写成 SLA/TTFT 相关的预算规则，而不是额外年龄权重

**首句**：为了避免算法沦为调参式打分，等待代价最好直接绑定到 TTFT 或等待侧 SLA 预算，而不是引入额外的年龄权重系数。

**展开**：
- 在大多数在线 serving 场景下，请求延迟的约束最终都可以落到 TTFT 或等待预算上，因此防饥饿规则完全可以直接写成与等待预算相关的 override 条件。
- 例如，当 $T_{wait}(j)$ 明显小于 $T_{budget}(j)$ 时，调度器应优先遵守资源约束；而当 $T_{wait}(j)$ 已逼近 $T_{budget}(j)$ 时，调度器可以适度放宽保守性，优先避免长期排队。
- 这样以后，算法中的阈值就不是独立调出来的“年龄系数”，而是服务目标本身定义好的等待预算。

### §13 因而，第一版方法最合理的定位不是“保守调度 + 预测补偿”，而是“激进调度 + 物理约束刹车”

**首句**：综合前面的设计，第一版 decode admission 更合理的定位不是站在保守一侧再用预测往回拉，而是在具体运行时的激进调度基础上用物理约束做风险刹车；对 vLLM 而言，这个基础就是其原生激进调度。

**展开**：
- 这种定位和你当前的复现实验更一致，因为它承认原生激进调度已经很好地利用了底层按需追加分配，而我们真正要补的不是“更强保守性”，而是“在即将越过短期瓶颈时及时踩刹车”。
- 因而，这个方法的核心不是一个激进与保守之间的固定折中点，而是一个基于当前资源工作区间的动态 admission rule。
- 用系统论文的语言来说，这更像一个 allocation-quantum-aware、horizon-limited、physical-constraint-guided admission control，而不是一个黑盒 scoring policy；在 vLLM 中，它可被具体实例化为 block-aware admission。

---

## 算法设计伪代码

### §14 算法的执行流程可以直接写成“分布预测、风险映射、物理约束判定”三步

**首句**：如果把方法写成伪代码，最自然的结构就是先保留 5 桶分布，再把它映射为短视界资源追加风险，最后用三维负载与 free resources 做准入判定。

**展开**：

```text
Algorithm 1: Decode Intra-instance Admission with 5-Bucket Prediction

# Inputs:
#   j               : queue head waiting request
#   R               : current running request set
#   B_free          : current free allocation units in the instance
#   kv_usage        : current KV usage ratio
#   S_alloc         : number of tokens per allocation unit
#                     (for vLLM, this is the number of tokens per KV block)
#   H               : short-horizon window in decode steps
#   L_C, L_M, L_K   : current 3D loads of the instance
#   T_wait(j)       : current waiting time of request j
#   T_budget(j)     : waiting-side SLA/TTFT budget of request j
#   P_i             : 5-bucket remaining-length distribution of request i
#   d_i             : tokens remaining before request i reaches next allocation boundary
#   calc_load()     : existing 3D load model that returns L_C, L_M, L_K, L_bottle
#   total_capacity(): total number of allocation units in the instance

# Output:
#   admit or defer for queue head j

function SHOULD_ADMIT(j, R, B_free, kv_usage):

    R_plus <- R union {j}
    total_added_tokens <- 0
    total_added_units <- 0
    n_req_hat_plus <- 0
    L_bottle <- max(L_C, L_M, L_K)
    rho <- min(1, L_bottle)

    # Step 1. Convert each request's 5-bucket distribution into short-horizon activity.
    for each request i in R_plus:

        # The bucket representative becomes more conservative as the instance
        # approaches the budget frontier. Since L_bottle can be > 1 in the 3D
        # load model, we clip it into rho in [0, 1] before using it as the
        # conservatism level.

        # Closed-form approximation of tokens added within the next H steps.
        y_hat_i_H <- 0
        active_prob_i_H <- 0
        for each bucket k in {1,2,3,4,5}:
            l_tilde_i_k <- (1 - rho) * bucket_mean(i, k) + rho * bucket_upper(i, k)
            y_hat_i_H <- y_hat_i_H + p_i[k] * min(H, l_tilde_i_k)
            if l_tilde_i_k >= H:
                active_prob_i_H <- active_prob_i_H + p_i[k]

        # Estimated new allocation-unit demand within the next H steps.
        delta_Q_hat_i_H <- ceil(max(0, y_hat_i_H - d_i) / S_alloc)

        total_added_tokens <- total_added_tokens + y_hat_i_H
        total_added_units <- total_added_units + delta_Q_hat_i_H
        n_req_hat_plus <- n_req_hat_plus + active_prob_i_H

    # Step 2. Build the post-admission short-horizon system state.
    m_tok_hat_plus <- current_running_tokens(R) + total_added_tokens
    kv_usage_hat_plus <- kv_usage + total_added_units / total_capacity()

    # Step 3. Predict the 3D load after admitting j.
    load_hat_plus <- calc_load(n_req_hat_plus, m_tok_hat_plus, kv_usage_hat_plus)

    # Step 4. Check physical resource constraints first.
    if load_hat_plus.L_C <= 1 and
       load_hat_plus.L_M <= 1 and
       load_hat_plus.L_K <= 1 and
       total_added_units <= B_free:
        return ADMIT

    # Step 5. If the request is close to violating waiting-side SLA, relax conservatism.
    # This override is still tied to a physical service budget, not a learned score.
    if T_wait(j) >= T_budget(j):
        if immediate_step_feasible(j, B_free):
            return ADMIT

    return DEFER
```

### §15 这段伪代码的关键不在于复杂性，而在于每一步都能对应到具体的系统含义

**首句**：这份伪代码最重要的特点，不是它多复杂，而是它把 predictor、风险映射和 admission control 三层职责清楚分开，并且每一步都对应到明确的系统含义，同时没有要求对每个 running 请求显式展开未来每一步的完整滚动模拟。

**展开**：
- 第一层是 predictor，只负责输出 5 桶长度分布，不直接输出准入决策。
- 第二层是风险映射，只负责把长度分布变成未来 $H$ 步内的 token 增长和资源单元增长估计；其中保守程度直接由当前 $L_{bottle}$ 调节，因此空闲时更激进、繁忙时更保守这一行为是算法自然涌现出来的，而不是额外加的打分项。
- 第三层是 admission control，只根据准入后的三维负载与可追加资源余量判定是否允许 queue head 进入 running。
- 这样的分层让方法更容易实现，也更容易在论文里解释为什么它不是一个“用预测替代调度”的黑盒系统。

---

## 符号说明

### §16 为了保证方法部分可读，核心符号应集中给出并保持和 motivation 一致

**首句**：为了让方法部分更容易阅读，本文在这里集中列出 decode 实例内调度设计中最关键的一组符号，并尽量与现有 motivation 和三维负载模型保持一致。

**展开**：

| 符号 | 含义 |
| --- | --- |
| $R_t$ | 当前时刻实例中的 running 请求集合 |
| $W_t$ | 当前时刻实例中的 waiting 队列 |
| $j$ | 当前 queue head waiting 请求 |
| $B_{free}$ | 当前实例中可用的 free resource units 数；对 vLLM 而言即 free KV blocks 数 |
| $S_{alloc}$ | 单个离散资源分配单元可容纳的 token 数；对 vLLM 而言即单个 KV block 可容纳的 token 数 |
| $H$ | 短视界观察窗口，以 decode step 为单位 |
| $L_{bottle}$ | 当前实例的瓶颈负载，定义为 $\max(L_C, L_M, L_K)$ |
| $S_{bottle}$ | 当前实例的瓶颈 slack，定义为 $1 - L_{bottle}$ |
| $\rho$ | 由瓶颈负载导出的保守度系数，定义为 $\min(1, L_{bottle})$ |
| $p_{i,k}$ | 请求 $i$ 的第 $k$ 个长度 bucket 概率 |
| $\bar{l}_k^{mean}$ | 第 $k$ 个 bucket 的经验条件均值 |
| $\bar{l}_k^{upper}$ | 第 $k$ 个 bucket 的上界代表值或保守代表值 |
| $\tilde{l}_{i,k}$ | 在当前瓶颈负载下，请求 $i$ 的第 $k$ 个 bucket 的动态代表长度 |
| $A_i(h)$ | 请求 $i$ 在未来第 $h$ 个 step 上仍然活跃的近似概率 |
| $\hat{y}_i^{H}$ | 请求 $i$ 在未来 $H$ 步内预计新增的 token 数 |
| $d_i$ | 请求 $i$ 距离下一个资源分配边界还差的 token 数；对 vLLM 而言即下一个 block 边界 |
| $\hat{\Delta Q}_i^{H}$ | 请求 $i$ 在未来 $H$ 步内预计新增的资源单元数 |
| $\hat{\Delta Q}_H^{+}(j)$ | 将请求 $j$ 准入后，实例在未来 $H$ 步内的新增资源单元需求估计 |
| $H_{alloc}$ | 一个最小资源追加窗口；对 vLLM 而言可取一个 KV block 对应的 token 数 |
| $H_{budget}$ | 由等待预算推导出的局部未来窗口，近似为 $(T_{budget}-T_{wait}) / TPOT_{target}$ |
| $L_C$ | 当前或预测的 compute pressure |
| $L_M$ | 当前或预测的 memory/bandwidth pressure |
| $L_K$ | 当前或预测的 KV capacity pressure |
| $\hat{L}_C^{+}(j), \hat{L}_M^{+}(j), \hat{L}_K^{+}(j)$ | 请求 $j$ 准入后，实例在未来短视界内的三维负载预测值 |
| $T_{wait}(j)$ | 请求 $j$ 当前已经等待的时间 |
| $T_{budget}(j)$ | 请求 $j$ 的等待预算或 TTFT 预算 |

### §17 如果需要一句最短的方法概括，这个设计可以压缩成“分布保留、风险映射、物理约束准入”

**首句**：如果把整套 decode 实例内调度设计再压缩成一句最短概括，那么它可以表述为“保留 5 桶分布，用它估计短视界资源追加风险，再用三维负载与 free resources 做物理约束准入；在 vLLM 中，这一设计具体实例化为 block-aware admission”。

**展开**：
- 预测层不做过早压缩，保留 5 桶长度分布作为风险原语。
- 风险层不直接追求 final peak memory，而是把长度分布映射为未来 $H$ 步内的 token 增长和资源单元增长，并由当前 $L_{bottle}$ 自适应地调节激进/保守程度。
- 调度层不引入过多抽象超参数，而是用 $L_C, L_M, L_K, B_{free}$ 和等待预算来决定 queue head 是否准入。
- 这样得到的方法既和现有 motivation 完整衔接，也符合系统论文对可解释性和低超参数设计的偏好。
