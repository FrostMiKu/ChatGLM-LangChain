# ChatGLM-LangChain

基于向量匹配实现的本地知识库问答的 ChatGLM 模型，**支持流式推理**

## Setup

1. 将本地的 markdown 文档丢入 content 文件夹中
2. 安装依赖环境
```bash
pip install -r requirements.txt
```
3. 启动程序
```bash
streamlit run ui.py --browser.gatherUsageStats False
```

## Screenshot

基于本地数据驱动的效果
![](screenshot/1.png)

![](screenshot/2.png)

更多 ACGN AIGC 项目欢迎访问我们的 [GitHub Organization](https://github.com/vtuber-plan)