import streamlit as st

from langchain.memory import ConversationBufferMemory
from utils import qa_agent


st.title("📑 AI智能PDF问答工具")

with st.sidebar:
    api_key = st.text_input("请输入Qwen AI API密钥：", type="password")
    st.markdown("[获取Qwen API密钥](https://bailian.console.aliyun.com/#/home)")

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

question = st.text_input("对PDF的内容进行提问")


ask_button = st.button("提问")

if question and not api_key:
    st.info("请输入你的Qwen API密钥")

if ask_button and question and api_key:
    with st.spinner("AI正在思考中，请稍等..."):

        response = qa_agent(api_key, st.session_state["memory"], question)
    st.write("### 答案")
    st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()