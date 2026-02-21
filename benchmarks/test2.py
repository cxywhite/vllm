if args.dataset_name == "sharegpt":
        default_max_pos = 2048
        max_pos = default_max_pos
        if AutoConfig is not None and tokenizer_id is not None:
            try:
                config = AutoConfig.from_pretrained(tokenizer_id, trust_remote_code=True)
                max_pos = getattr(config, 'max_position_embeddings', max_pos)
                print(f'tokenizer_id={tokenizer_id}')
                print(f"加载模型配置成功，模型声明 max_position_embeddings={max_pos}")
            except Exception as e:
                print(f"读取 AutoConfig 失败，使用默认 max_pos={max_pos}: {e}")
                max_pos = default_max_pos
        max_pos = min(int(max_pos), 131072)
        print(f"最终使用 max_pos={max_pos}")

        # max_embedding_pos 可由 args.max_embedding_pos 覆盖
        max_embedding_pos = getattr(args, 'max_embedding_pos', max_pos)
        max_embedding_pos = min(int(max_embedding_pos), 131072)

        # 全局 maxtokenscustom（fallback）
        maxtokenscustom_global = getattr(args, 'maxtokenscustom', None) or 1024

        # ------------------ 2) tokenize_len helper（优先使用传入 tokenizer，再回退 AutoTokenizer，再回退字符长度） ------------------
        def tokenize_len(text: str) -> int:
            if text is None:
                return 0
            text = str(text)
            # 优先使用已提供的 tokenizer
            try:
                if tokenizer is not None:
                    maybe = tokenizer(text, add_special_tokens=True)
                    if isinstance(maybe, dict) and 'input_ids' in maybe:
                        return len(maybe['input_ids'])
                    if hasattr(tokenizer, 'encode'):
                        ids = tokenizer.encode(text, add_special_tokens=True)
                        return len(ids)
            except Exception:
                pass
            # 回退到 AutoTokenizer（如果可用）
            try:
                if AutoTokenizer is None:
                    raise RuntimeError("transformers.AutoTokenizer 未可用")
                if tokenizer_id is None:
                    raise RuntimeError("没有提供 tokenizer_id 用于回退 AutoTokenizer")
                # logger.info(f"加载 tokenizer_id={tokenizer_id} 进行分词长度计算")
                local_tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True, trust_remote_code=True)
                maybe = local_tok(text, add_special_tokens=True)
                return len(maybe['input_ids'])
            except Exception:
                # 最后回退到字符数
                return len(text)

        # ------------------ 3) 加载 HF 本地数据集 ------------------
        ds_path = getattr(args, 'dataset_path', None)
        if not ds_path:
            raise RuntimeError("args.dataset_path 未设置")

        # 自动处理：如果路径是目录，尝试寻找 .json 文件
        target_file = ds_path
        if os.path.isdir(ds_path):
            print(f"检测到目录: {ds_path}，正在查找 json 文件...")
            # 优先找 cleaned 版本，其次找任意 json
            priority_files = ["ShareGPT_V4.3_unfiltered_cleaned_split.json", "ShareGPT_2023.05.04v0_Wasteland_Edition.json"]
            found = False
            for fname in priority_files:
                fpath = os.path.join(ds_path, fname)
                if os.path.exists(fpath):
                    target_file = fpath
                    found = True
                    break
            if not found:
                # 没找到优先文件，随便找一个 json
                candidates = [f for f in os.listdir(ds_path) if f.endswith('.json')]
                if candidates:
                    target_file = os.path.join(ds_path, candidates[0])
        
        print(f"正在加载 ShareGPT 文件: {target_file}")
        
        try:
            # 使用 'json' builder 加载，而不是 load_from_disk
            ds = load_dataset("json", data_files=target_file, split="train")
            ds = ds.select(range(3))
            print(f"数据集加载成功，原始样本数: {len(ds)}")
            print(ds)
        except Exception as e:
            raise RuntimeError(f"加载 JSON 失败 (请检查路径是否正确): {e}") from e

        # ------------------ 4) 解析 ShareGPT 格式并转为 DataFrame ------------------
        # ShareGPT 结构嵌套: {'conversations': [{'from': 'human', 'value': '...'}, ...]}
        # 我们需要提取 human 的输入，并补充 Step 8 需要的采样参数
        
        # 获取默认采样参数 (防止后续 KeyError)
        d_temp = getattr(args, 'temperature', 0.7)
        d_top_p = getattr(args, 'top_p', 0.9)
        d_top_k = getattr(args, 'top_k', -1)
        d_rep_pen = getattr(args, 'repetition_penalty', 1.0)

        parsed_rows = []
        print("正在解析 ShareGPT 对话格式...")
        
        for record in ds:
            convs = record.get('conversations', [])
            if not convs:
                continue
            
            # 提取逻辑：获取第一个 human/user 的发言作为 prompt
            prompt_text = ""
            for turn in convs:
                if turn.get('from') in ['human', 'user']:
                    prompt_text = turn.get('value', '')
                    break
            
            # 如果是空对话则跳过
            if not prompt_text.strip():
                continue

            parsed_rows.append({
                "id": record.get('id', None),
                "prompt": prompt_text,
                # 必须补充这些列，否则后面的 Step 8 会报错
                "temperature": d_temp,
                "topp": d_top_p,
                "topk": d_top_k,
                "repetition_penalty": d_rep_pen
            })

        df = pd.DataFrame(parsed_rows)
        valid_count = len(df) 
        # ------------------ 5) 计算 prompt_tokens（优先使用 input_length 字段）并过滤过长（阈值 8192） ------------------
        # if 'input_length' in df.columns:
        #     df['prompt_tokens'] = df['input_length'].apply(_to_int_safe)
        # else:
        #     df['prompt_tokens'] = df['prompt'].apply(lambda x: tokenize_len(x))
        df['prompt_tokens'] = df['prompt'].apply(lambda x: tokenize_len(x))

        # 采用原代码常数阈值（保持行为）：8192
        filtered_df = df[df['prompt_tokens'] < 8192].reset_index(drop=True)
        filtered_count = len(filtered_df)
        print(f"过滤长 prompt: {valid_count} -> {filtered_count} 条记录 (阈值=8192)")
        if filtered_count == 0:
            raise ValueError("过滤后没有可用的 prompt：所有 prompt 超过模型的最大长度限制。")

        # ------------------ 6) 抽样（按 args.num_prompts） ------------------
        num_prompts = getattr(args, 'num_prompts', None)
        if num_prompts is None:
            # 若没有提供，使用全部
            sample_size = filtered_count
            print(f"args.num_prompts 未设置，使用全部 {sample_size} 条样本")
        else:
            if filtered_count < int(num_prompts):
                print(f"警告: 过滤后只有 {filtered_count} 条记录，少于请求的 {num_prompts} 条；将使用所有 {filtered_count} 条")
                sample_size = filtered_count
            else:
                sample_size = int(num_prompts)
        sample_df = filtered_df.sample(n=sample_size, random_state=getattr(args, 'seed', 0)).reset_index(drop=True)
        print(f"抽样完成: {sample_size} 个样本 (从 {filtered_count} 条过滤后的样本中抽取)")

        # ------------------ 7) 计算 custom_max_tokens（保留 qwen 公式: 32*1024 - prompt_len） ------------------
        # LLaMA3
        sample_df['custom_max_tokens'] = sample_df['prompt_tokens'].apply(
            lambda input_len: max(1, min(8*1024 - int(input_len), int(maxtokenscustom_global)))
        )
        # Qwen
        # sample_df['custom_max_tokens'] = sample_df['prompt_tokens'].apply(
        #     lambda input_len: max(1, min(32*1024 - int(input_len), int(maxtokenscustom_global)))
        # )
        # print(sample_df['temperature'][:5])
        # print(sample_df['topp'][:5])

        # ------------------ 8) 构建 SampleRequest 列表 ------------------
        input_requests: List[SampleRequest] = []
        for idx, row in sample_df.iterrows():
            # base_id 优先用 row['id']，否则用 row index
            base_id = None
            if 'id' in row and row['id'] is not None:
                try:
                    base_id = str(_to_int_safe(row['id']))
                except Exception:
                    base_id = str(row['id'])
            if not base_id:
                base_id = f"row{idx}"

            prompt_text = str(row['prompt'])
            prompt_len = int(row['prompt_tokens'])

            # expected_output_len 优先使用 output_length 字段（如果存在），否则使用 custom_max_tokens
            expected_output_len = None
            # if 'output_length' in row and row['output_length'] is not None:
            #     expected_output_len = _to_int_safe(row['output_length'])
            # else:
            #     expected_output_len = int(row['custom_max_tokens'])
            expected_output_len = int(row['custom_max_tokens'])
            # optional sampling_params（如果你希望把采样参数传入 SampleRequest，可在这里解注并使用 args）
            sampling_params = None
            # if you want sampling params, uncomment:
            # print(round(row['temperature'], 2))
            input_requests.append(
                SampleRequest(
                    req_id=f"{base_id}",
                    prompt=prompt_text,
                    prompt_len=prompt_len,
                    expected_output_len=int(expected_output_len),
                    temperature=round(row['temperature'], 2),
                    top_p=round(row['topp'], 2),
                    top_k=int(row['topk']),
                    repetition_penalty=round(row['repetition_penalty'], 2),
                )
            )

        print(f"构建 input_requests 完成：样本 {len(sample_df)} 条，总请求数 = {len(input_requests)}")
        if args.ablation_p2d and args.baseline_csv_path:

            # 读取CSV文件
            test_df = pd.read_csv(args.baseline_csv_path)

            # 1. 创建req_id到expected_output_len的映射字典（关键优化）
            req_id_to_output_len = {}
            for idx, row in test_df.iterrows():
                # 确保req_id作为字符串键，保持与input_requests中req_id类型一致
                req_id_str = str(row['req_id'])
                req_id_to_output_len[req_id_str] = int(row['output_tokens'])  # 或者 row['expected_output_len'] 根据实际列名

            # 2. 单次遍历input_requests，通过字典快速查找
            total_num = 0
            for i, request in enumerate(input_requests):
                req_id_str = str(request.req_id)
                
                if req_id_str in req_id_to_output_len:
                    request.expected_output_len = req_id_to_output_len[req_id_str]
                    total_num += 1
                else:
                    # 可选：处理找不到匹配的情况
                    print(f"Warning: req_id {req_id_str} not found in baseline CSV")

            # 3. 验证匹配数量
            assert total_num == len(input_requests), f"匹配baseline数据条数{total_num}不等于input_requests长度{len(input_requests)}"            
            print(f'已根据baseline CSV文件更新 {total_num} 条请求的 expected_output_len。')