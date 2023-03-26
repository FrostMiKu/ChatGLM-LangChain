from transformers import AutoModel, AutoTokenizer
import streamlit as st

MAX_CONTEXT = 720

st.set_page_config(
    page_title="Chat Page",
    page_icon=":robot:",
    menu_items={"about": '''
                Author: FrostMiKu

                Model: ChatGLM-6B-INT4
                '''}
)


@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "silver/chatglm-6b-int4-slim", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "silver/chatglm-6b-int4-slim", trust_remote_code=True).half().cuda()
    model = model.eval()
    return tokenizer, model


if 'first_run' not in st.session_state:
    st.session_state.first_run = True
if 'history' not in st.session_state:
    st.session_state.history = []
if 'ctx' not in st.session_state:
    st.session_state.ctx = []

st.title("# Hello, there👋")
ctx_dom = st.empty()
question_dom = st.markdown(
    "> 为了能够顺利的运行在显存仅有 6GB 的 RTX 2060 Ti 上\\\n本模型被限制了上下文能力，当前最大 Token 长度：{}".format(MAX_CONTEXT))
md_dom = st.empty()
st.write("")

tokenizer, model = get_model()


def display_ctx(history=None):
    if history != None:
        text = ""
        for q, a in history:
            text += ":face_with_cowboy_hat:\n{}\n\n---\n{}\n\n---\n".format(
                q, a)
            ctx_dom.markdown(text)


def check_ctx_len(history):
    total = 0
    for q, a in history:
        total = total + len(q) + len(a)
    return total <= (MAX_CONTEXT + 10)


def predict(input, history=None):
    if history is None:
        history = []
    while not check_ctx_len(history):
        print("Free Context!")
        history.pop(0)

    for response, history in model.stream_chat(tokenizer, input, history, max_length=1024, top_p=0.8,
                                               temperature=0.9):
        md_dom.markdown(response)

    st.session_state.history.append(history[-1])
    return history


with st.form("form", True):
    # create a prompt text for the text generation
    prompt_text = st.text_area(label=":thinking_face: 聊点什么？",
                               height=100,
                               max_chars=MAX_CONTEXT,
                               placeholder="支持使用 Markdown 格式书写")
    col1, col2 = st.columns([1, 1])
    with col1:
        btn_send = st.form_submit_button(
            "发送", use_container_width=True, type="primary")
    with col2:
        btn_clear = st.form_submit_button("清除历史记录", use_container_width=True)

    if btn_send and prompt_text != "":
        display_ctx(st.session_state.history)
        question_dom.markdown(
            ":face_with_cowboy_hat:\n{}\n\n---\n".format(prompt_text))
        st.session_state.ctx = predict(prompt_text, st.session_state.ctx)
        if st.session_state.first_run:
            st.session_state.first_run = False
            st.balloons()

    if btn_clear:
        ctx_dom.empty()
        st.session_state.history = []
        st.session_state.ctx = []
