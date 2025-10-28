# 使用官方的Ubuntu 22.04作为基础镜像
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive

# 更新软件源并安装基础工具（Python, pip, Git）
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# [关键步骤1] 将 requirements.txt 文件从主机复制到镜像的 /app 目录中
# 注意：这个 COPY 指令必须在 RUN pip install 之前
COPY requirements.txt .

# [关键步骤2] 读取 requirements.txt 文件，安装所有Python库
RUN pip install -r requirements.txt

# 当容器启动时，默认执行bash命令
CMD ["bash"]