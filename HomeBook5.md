# 构建环境lmdeploy
studio-conda -t lmdeploy -o pytorch-2.1.2

# 进入环境lmdeploy
conda activate lmdeploy

# 安装版本lmdeploy
pip install lmdeploy[all]==0.3.0

# InternStudio开发机上下载模型
ls /root/share/new_models/Shanghai_AI_Laboratory/

# 由OpenXLab平台下载模型
返回HOME
cd ~

# 安装git-lfs组件
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt update
sudo apt install git-lfs   
sudo git lfs install  --system

# 下载InternLM2-Chat-1.8B模型
git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b.git

# 文档统一
mv /root/internlm2-chat-1.8b /root/internlm2-chat-1_8b

# 使用Transformer库运行模型

# 在终端中输入如下指令，新建pipeline_transformer.py。
touch /root/pipeline_transformer.py

# 将以下内容复制粘贴进入pipeline_transformer.py。

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/root/internlm2-chat-1_8b", trust_remote_code=True)

Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("/root/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()

inp = "hello"
print("[INPUT]", inp)
response, history = model.chat(tokenizer, inp, history=[])
print("[OUTPUT]", response)

inp = "please provide three suggestions about time management"
print("[INPUT]", inp)
response, history = model.chat(tokenizer, inp, history=history)
print("[OUTPUT]", response)

# 按Ctrl+S键保存（Mac用户按Command+S）。

# 激活conda环境。
conda activate lmdeploy

# 运行python代码：
python /root/pipeline_transformer.py

# 使用LMDeploy与模型对话
conda activate lmdeploy

# LMDeploy模型量化(lite)
# 运行下载的1.8B模型
lmdeploy chat /root/internlm2-chat-1_8b

# 使用W4A16量化
# 安装一个依赖库
pip install einops==0.7.0

# 仅需执行一条命令，就可以完成模型量化工作。

lmdeploy lite auto_awq \
   /root/internlm2-chat-1_8b \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 1024 \
  --w-bits 4 \
  --w-group-size 128 \
  --work-dir /root/internlm2-chat-1_8b-4bit

# W4A16，将KV Cache比例再次调为0.4
lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq --cache-max-entry-count 0.4

# 启动API服务器

lmdeploy serve api_server \
    /root/internlm2-chat-1_8b \
    --model-format hf \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1


