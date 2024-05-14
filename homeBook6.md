# Lagent & AgentLego 智能体应用搭建

# 创建开发机和 conda 环境

mkdir -p /root/agent

studio-conda -t agent -o pytorch-2.1.2

# 安装 Lagent 和 AgentLego

cd /root/agent

conda activate agent

git clone https://gitee.com/internlm/lagent.git

cd lagent && git checkout 581d9fb && pip install -e . && cd ..

git clone https://gitee.com/internlm/agentlego.git

cd agentlego && git checkout 7769e0d && pip install -e . && cd ..


conda activate agent

pip install lmdeploy==0.3.0


准备 Tutorial

cd /root/agent

git clone -b camp2 https://gitee.com/internlm/Tutorial.git

