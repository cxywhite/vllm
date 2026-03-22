import pandas as pd
dataset_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/mysharegpt/processed_output/qwen-mysharegpt-updated.csv'
save_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/Loadbalance/dataset/qwen-mysharegpt-loadbalance_1p3d_4.csv'
df = pd.read_csv(dataset_path)
# 打乱随机选择1000条数据
df1 = df[(df['output_tokens'] >512) & (df['output_tokens'] <= 10000)].sort_values(by='output_tokens')
df2 = df[(df['output_tokens'] > 10000) & (df['output_tokens'] <= 20000)].sort_values(by='output_tokens')
df3 = df[df['output_tokens'] > 25000].sort_values(by='output_tokens')
# 从df1中随机选取500条数据，df2中随机选取400条数据，df3中随机选取100条数据
df1 = df1.sample(n=897, random_state=42)
df2 = df2.sample(n=100, random_state=42)
df3 = df3.sample(n=3, random_state=42)
df_final = pd.concat([df1, df2, df3], ignore_index=True)
# df1_1选取df1中300条数据,选取df2中33条数据，选取df3中1条数据
df1_1 = df1.iloc[0:320]
df2_1 = df2.iloc[0:13]
df3_1 = df3.iloc[0:1]
df_final_1 = pd.concat([df1_1, df2_1, df3_1], ignore_index=True)
# 打乱df_final_1
df_final_1 = df_final_1.sample(frac=1, random_state=42).reset_index(drop=True)
# df2_2选取df1中250条数据,选取df2中82条数据，选取df3中1条数据
df1_2 = df1.iloc[320:620]
df2_2 = df2.iloc[13:45]
df3_2 = df3.iloc[1:2]
df_final_2 = pd.concat([df1_2, df2_2, df3_2], ignore_index=True)
# 打乱df_final_2
df_final_2 = df_final_2.sample(frac=1, random_state=42).reset_index(drop=True)
# df3_3选取df1中150条数据,选取df2中182条数据，选取df3中1条数据
df1_3 = df1.iloc[620:897]
df2_3 = df2.iloc[45:100]
df3_3 = df3.iloc[2:3]
df_final_3 = pd.concat([df1_3, df2_3, df3_3], ignore_index=True)
# 打乱df_final_3
df_final_3 = df_final_3.sample(frac=1, random_state=42).reset_index(drop=True)
# 打印df_final_1的output_tokens的描述性统计信息
print(df_final_1[['prompt_len','output_tokens']].describe())
# 打印df_final_2的output_tokens的描述性统计信息
print(df_final_2[['prompt_len','output_tokens']].describe())
# 打印df_final_3的output_tokens的描述性统计信息
print(df_final_3[['prompt_len','output_tokens']].describe())
# 将df_final_1到df_final_3合并成一个新的dataframe，保证df_final_1的数据都在0,3,6,9,12的行中，df_final_2的数据都在1,4,7,10,13的行中，df_final_3的数据都在2,5,8,11,14的行中
df_final = pd.DataFrame()
for i in range(333):
    df_final = pd.concat([df_final, df_final_1.iloc[i:i+1], df_final_2.iloc[i:i+1], df_final_3.iloc[i:i+1]], ignore_index=True)
df_final = pd.concat([df_final, df_final_1.iloc[333:]], ignore_index=True)
print(len(df_final))
# 打印df_final的前6行
print(df_final.head(6))
print(len(df_final))
# 打印df_final的前6行
# 删除total_tokens列
# df_final = df_final.drop(columns=['total_tokens'])
# 将df_final保存为csv文件
df_final.to_csv(save_path, index=False)

# df_final['total_tokens'] = df_final['prompt_len'] + df_final['output_tokens']
# print(df_final[['prompt_len','output_tokens','total_tokens']].describe())
# # 删除total_tokens列
# df = df_final.drop(columns=['total_tokens'])
# print(len(df))
# print(df.head(6))
# # 保存
# df.to_csv(save_path, index=False)

# select llama mysharegpt
# import pandas as pd
# dataset_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/mysharegpt/processed_output/llama-mysharegpt.csv'
# save_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/Loadbalance/dataset/llama-mysharegpt-loadbalance_2.csv'
# df=pd.read_csv(dataset_path)
# # 过滤出output_tokens在0-3000之间的数据
# df1 = df[(df['output_tokens'] >0) & (df['output_tokens'] <= 3000)]
# df2 = df[(df['output_tokens'] > 3000) & (df['output_tokens'] <= 5000)]
# df3 = df[df['output_tokens'] > 5000]
# # 从df1中随机选取333条数据，df2中随机选取333条数据，df3中随机选取334条数据
# df1 = df1.sample(n=500, random_state=42)
# df2 = df2.sample(n=400, random_state=42)
# df3 = df3.sample(n=100, random_state=42)
# # 将df1到df3合并成一个新的dataframe然后打散
# df_final = pd.concat([df1, df2, df3], ignore_index=True)
# df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
# print(len(df_final))
# # 打印df_final的前6行
# print(df_final.head(6))
# # 打印df_final的prompt_len和output_tokens的描述性统计信息
# print(df_final[['prompt_len', 'output_tokens']].describe())
# # 将df_final保存为csv文件
# df_final.to_csv(save_path, index=False)

# select lmsys 2short1long
# import pandas as pd
# dataset_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/lmsys-chat/processed_output/llama-lmsys-chat.csv'
# save_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/Loadbalance/dataset/llama-lmsys-chat-loadbalance_1p3d_3.csv'
# df = pd.read_csv(dataset_path)
# # df['total_tokens'] = df['prompt_len'] + df['output_tokens']
# # 按照total_tokens从小到大排序
# df = df.sort_values(by='output_tokens')
# df1 = df.iloc[int(len(df)*0.87):int(len(df)*0.87)+333]
# # 在df中去除df1中的数据，得到df
# df = df.drop(df1.index)
# # 根据df长度选90分位数长度的后334条数据作为df2
# df2 = df.iloc[int(len(df)*0.87):int(len(df)*0.87)+333]
# # 在df中去除df2中的数据，得到df
# df = df.drop(df2.index)
# # 根据df长度选90分位数的333条数据作为df3
# df3 = df.iloc[int(len(df)*0.9):int(len(df)*0.9)+334]
# # 在df中去除df3中的数据，得到df
# df = df.drop(df3.index)
# # 将df1到df3合并成一个新的dataframe，保证df1的数据都在0,3,6,9,12的行中，df2的数据都在1,4,7,10,13的行中，df3的数据都在2,5,8,11,14的行中
# df_final = pd.DataFrame()
# for i in range(333):
#     df_final = pd.concat([df_final, df1.iloc[i:i+1], df2.iloc[i:i+1], df3.iloc[i:i+1]], ignore_index=True)
# df_final = pd.concat([df_final, df3.iloc[333:]], ignore_index=True)
# print(len(df_final))
# # 打印df_final的前6行
# print(df_final['output_tokens'].head(6))
# # 删除total_tokens列
# # df_final = df_final.drop(columns=['total_tokens'])
# # 将df_final保存为csv文件
# df_final.to_csv(save_path, index=False)

# select lmsys 2short1long
# import pandas as pd
# dataset_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/lmsys-chat/processed_output/llama-lmsys-chat.csv'
# save_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/Loadbalance/dataset/llama-lmsys-chat-loadbalance_1p3d_2.csv'
# df = pd.read_csv(dataset_path)
# # df['total_tokens'] = df['prompt_len'] + df['output_tokens']
# # 按照total_tokens从小到大排序
# df = df.sort_values(by='output_tokens')
# df1 = df.iloc[int(len(df)*0.87):int(len(df)*0.87)+333]
# # 在df中去除df1中的数据，得到df
# df = df.drop(df1.index)
# # 根据df长度选90分位数长度的后334条数据作为df2
# df2 = df.iloc[int(len(df)*0.9):int(len(df)*0.9)+333]
# # 在df中去除df2中的数据，得到df
# df = df.drop(df2.index)
# # 根据df长度选90分位数的333条数据作为df3
# df3 = df.iloc[int(len(df)*0.9):int(len(df)*0.9)+334]
# # 在df中去除df3中的数据，得到df
# df = df.drop(df3.index)
# # 将df1到df3合并成一个新的dataframe，保证df1的数据都在0,3,6,9,12的行中，df2的数据都在1,4,7,10,13的行中，df3的数据都在2,5,8,11,14的行中
# df_final = pd.DataFrame()
# for i in range(333):
#     df_final = pd.concat([df_final, df1.iloc[i:i+1], df2.iloc[i:i+1], df3.iloc[i:i+1]], ignore_index=True)
# df_final = pd.concat([df_final, df3.iloc[333:]], ignore_index=True)
# print(len(df_final))
# # 打印df_final的前6行
# print(df_final['output_tokens'].head(6))
# # 删除total_tokens列
# # df_final = df_final.drop(columns=['total_tokens'])
# # 将df_final保存为csv文件
# df_final.to_csv(save_path, index=False)

# import pandas as pd
# dataset_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/lmsys-chat/processed_output/llama-lmsys-chat.csv'
# save_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/Loadbalance/dataset/llama-lmsys-chat-loadbalance_1p3d_2.csv'
# df = pd.read_csv(dataset_path)
# df['total_tokens'] = df['prompt_len'] + df['output_tokens']
# # 按照total_tokens从小到大排序
# df = df.sort_values(by='total_tokens')
# df1 = df.iloc[int(len(df)*0.8):int(len(df)*0.8)+333]
# # 在df中去除df1中的数据，得到df
# df = df.drop(df1.index)
# # 根据df长度选90分位数长度的后334条数据作为df2
# df2 = df.iloc[int(len(df)*0.9):int(len(df)*0.9)+333]
# # 在df中去除df2中的数据，得到df
# df = df.drop(df2.index)
# # 根据df长度选90分位数的333条数据作为df3
# df3 = df.iloc[int(len(df)*0.9):int(len(df)*0.9)+334]
# # 在df中去除df3中的数据，得到df
# df = df.drop(df3.index)
# # 将df1到df3合并成一个新的dataframe，保证df1的数据都在0,3,6,9,12的行中，df2的数据都在1,4,7,10,13的行中，df3的数据都在2,5,8,11,14的行中
# df_final = pd.DataFrame()
# for i in range(333):
#     df_final = pd.concat([df_final, df1.iloc[i:i+1], df2.iloc[i:i+1], df3.iloc[i:i+1]], ignore_index=True)
# df_final = pd.concat([df_final, df3.iloc[333:]], ignore_index=True)
# print(len(df_final))
# # 打印df_final的前6行
# print(df_final.head(6))
# # 删除total_tokens列
# df_final = df_final.drop(columns=['total_tokens'])
# # 将df_final保存为csv文件
# df_final.to_csv(save_path, index=False)

# import pandas as pd
# dataset_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/lmsys-chat/processed_output/llama-lmsys-chat.csv'
# save_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/Loadbalance/dataset/llama-lmsys-chat-loadbalance.csv'
# df = pd.read_csv(dataset_path)
# # 选取prompt_len+output_tokens最小的前833条数据作为df1
# df['total_tokens'] = df['prompt_len'] + df['output_tokens']
# df1 = df.nsmallest(833, 'total_tokens')
# # 在df中去除df1中的数据，得到df
# df = df.drop(df1.index)
# # 选取prompt_len+output_tokens最小的前833条数据作为df2
# df2 = df.nsmallest(833, 'total_tokens')
# # 在df中去除df2中的数据，得到df
# df = df.drop(df2.index)
# # 选取prompt_len+output_tokens最小的前833条数据作为df3
# df3 = df.nsmallest(833, 'total_tokens')
# # 在df中去除df3中的数据，得到df
# df = df.drop(df3.index)
# # 选取prompt_len+output_tokens最小的前833条数据作为df4
# df4 = df.nsmallest(833, 'total_tokens')
# # 在df中去除df4中的数据，得到df
# df = df.drop(df4.index)
# # 选取prompt_len+output_tokens最小的前833条数据作为df5
# df5 = df.nsmallest(833, 'total_tokens')
# # 在df中去除df5中的数据，得到df
# df = df.drop(df5.index)
# # 选取prompt_len+output_tokens最大的前835条数据作为df6
# df6 = df.nlargest(835, 'total_tokens')
# # 在df中去除df6中的数据，得到df
# df = df.drop(df6.index)
# # 选取prompt_len+output_tokens最大的前833条数据作为df7
# df7 = df.nlargest(833, 'total_tokens')
# df2 = df7
# # 在df中去除df7中的数据，得到df
# df = df.drop(df7.index)
# # 选取prompt_len+output_tokens最大的前833条数据作为df8
# df8 = df.nlargest(833, 'total_tokens')
# df3 = df8
# # 在df中去除df8中的数据，得到df
# df = df.drop(df8.index)
# # 将df1到df6合并成一个新的dataframe，保证df1的数据都在0,5,10,15,20的行中，df2的数据都在1,6,11,16,21的行中，df3的数据都在2,7,12,17,22的行中，df4的数据都在3,8,13,18,23的行中，df5的数据都在4,9,14,19,24的行中，df6的数据都在5,10,15,20,25的行中
# df_final = pd.DataFrame()
# for i in range(833):
#     df_final = pd.concat([df_final, df1.iloc[i:i+1], df2.iloc[i:i+1], df3.iloc[i:i+1], df4.iloc[i:i+1], df5.iloc[i:i+1], df6.iloc[i:i+1]], ignore_index=True)
# df_final = pd.concat([df_final, df6.iloc[833:]], ignore_index=True)
# print(len(df_final))
# # 打印df_final的前12行
# print(df_final.head(12))
# # 删除total_tokens列
# df_final = df_final.drop(columns=['total_tokens'])
# # 将df_final保存为csv文件
# df_final.to_csv(save_path, index=False)

