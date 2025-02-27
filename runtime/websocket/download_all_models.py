from modelscope.hub.snapshot_download import snapshot_download

# 指定下载路径
model_dir = "/root/autodl-tmp/models"

# 模型名称列表
models = [
    "damo/speech_fsmn_vad_zh-cn-16k-common-onnx",  # VAD 模型
    "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx",  # 离线 ASR 模型
    "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx",  # 在线 ASR 模型
    "damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx",  # 标点模型
    "damo/speech_ngram_lm_zh-cn-ai-wesp-fst",  # 语言模型
    "thuduj12/fst_itn_zh"  # 逆文本归一化模型
]

# 下载所有模型
for model_name in models:
    print(f"Downloading model: {model_name}")
    snapshot_download(model_name, cache_dir=model_dir)
    print(f"Model {model_name} downloaded successfully.")

print("All models downloaded to:", model_dir)
