import os
from pathlib import Path
import streamlit as st

from medrag.config.settings import settings
from medrag.llm import get_llm_client

# LLM client (used for case summarisation)
deepseek_client = get_llm_client("deepseek")


# ---------------------------------------------------------------------------
# Cached initialisation (singleton)
# ---------------------------------------------------------------------------

@st.cache_resource
def _init_chat_service():
    """Load MedicalChatService.  Cached so the NER model loads once."""
    from medrag.service.chat_service import MedicalChatService

    try:
        return {"service": MedicalChatService(), "error": None}
    except Exception as exc:
        return {"service": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Case-file processing
# ---------------------------------------------------------------------------

def process_case_file(uploaded_file, usname: str) -> str:
    """Save → parse → clean & desensitize → summarise → return summary."""
    from medrag.data.case_parser import parse_case_file
    from medrag.data.text_cleaner import clean_medical_text, desensitize_medical_text
    from medrag.data.case_summary import summarize_case

    user_dir = Path("user_uploads") / usname
    user_dir.mkdir(parents=True, exist_ok=True)

    dest = user_dir / uploaded_file.name
    with open(dest, "wb") as fh:
        fh.write(uploaded_file.getbuffer())

    raw = parse_case_file(str(dest))
    cleaned = clean_medical_text(raw)
    safe = desensitize_medical_text(cleaned)
    return summarize_case(safe, deepseek_client)


# ---------------------------------------------------------------------------
# Main Streamlit UI
# ---------------------------------------------------------------------------

def main(is_admin: bool, usname: str):
    st.title("医疗智能问答机器人")

    # -- Sidebar -------------------------------------------------------------
    with st.sidebar:
        col1, col2 = st.columns([0.6, 0.6])
        with col1:
            st.image(os.path.join("img", "logo.jpg"), use_column_width=True)

        st.caption(
            f"""<p align="left">欢迎您，{'管理员' if is_admin else '用户'}{usname}！当前版本：2.0</p>""",
            unsafe_allow_html=True,
        )

        # Multi-window chat
        if "chat_windows" not in st.session_state:
            st.session_state.chat_windows = [[]]
            st.session_state.messages = [[]]

        if st.button("新建对话窗口"):
            st.session_state.chat_windows.append([])
            st.session_state.messages.append([])

        window_options = [f"对话窗口 {i + 1}" for i in range(len(st.session_state.chat_windows))]
        selected_window = st.selectbox("请选择对话窗口:", window_options)
        active_idx = int(selected_window.split()[1]) - 1

        # Case upload
        st.divider()
        st.subheader("病例上传（可选）")
        case_key = f"case_summary_{active_idx}"

        uploaded = st.file_uploader(
            "上传病例文件",
            type=["txt", "pdf", "docx"],
            key=f"case_upload_{active_idx}",
            help="支持 TXT、PDF、DOCX；上传后会自动脱敏处理",
        )

        if uploaded is not None:
            with st.spinner("正在解析和脱敏病例…"):
                try:
                    st.session_state[case_key] = process_case_file(uploaded, usname)
                    st.success("病例处理完成")
                except Exception as exc:
                    st.error(f"病例处理失败：{exc}")

        if st.session_state.get(case_key):
            if st.button("清除已上传的病例"):
                st.session_state[case_key] = None
                st.experimental_rerun()

        # Admin-only debug toggles
        show_route = show_kg = show_toyhom = False
        if is_admin:
            st.divider()
            show_route = st.checkbox("显示检索路由信息")
            show_kg = st.checkbox("显示知识图谱结果")
            show_toyhom = st.checkbox("显示相似问答结果")
            if st.button("修改知识图谱"):
                st.markdown(
                    "[点击这里修改知识图谱](http://127.0.0.1:7474/)",
                    unsafe_allow_html=True,
                )

        if st.button("返回登录"):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.experimental_rerun()

    # -- Init service --------------------------------------------------------
    svc = _init_chat_service()
    if svc["error"]:
        st.error(f"系统初始化失败：{svc['error']}")
        st.info("请检查 Neo4j / Milvus 连接以及模型文件是否就位，然后刷新页面重试。")
        return

    service = svc["service"]

    # -- History -------------------------------------------------------------
    current_messages = st.session_state.messages[active_idx]

    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if show_route and msg.get("route"):
                    with st.expander("检索路由信息"):
                        st.json(msg["route"])
                if show_kg and msg.get("kg_results"):
                    with st.expander("知识图谱结果"):
                        for r in msg["kg_results"][:5]:
                            name = r.get("name") or r.get("名称", "")
                            desc = r.get("desc") or r.get("疾病简介", "")
                            st.markdown(f"**{name}**")
                            if desc:
                                st.caption(str(desc)[:300])
                            st.divider()
                if show_toyhom and msg.get("toyhom_results"):
                    with st.expander("相似问答结果"):
                        for r in msg["toyhom_results"][:5]:
                            st.markdown(f"**Q:** {r.get('question', '')}")
                            ans = r.get("answer", "")
                            if ans:
                                st.caption(str(ans)[:300])
                            st.divider()

    # -- Chat input ----------------------------------------------------------
    if query := st.chat_input("请输入您的医疗问题…", key=f"chat_input_{active_idx}"):
        current_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("正在检索和分析…"):
                try:
                    result = service.chat(
                        query,
                        user_case_summary=st.session_state.get(case_key),
                    )
                except Exception as exc:
                    result = {
                        "answer": f"抱歉，处理您的问题时出现错误：{exc}",
                        "route": {},
                        "kg_results": [],
                        "toyhom_results": [],
                        "reranked_results": [],
                        "risk_info": {},
                    }

            st.markdown(result["answer"])

            # Evidence expanders (always visible for transparency)
            if result.get("route"):
                with st.expander("检索路由信息"):
                    st.json(result["route"])
            if result.get("kg_results"):
                with st.expander("知识图谱结果"):
                    for r in result["kg_results"][:5]:
                        name = r.get("name") or r.get("名称", "")
                        desc = r.get("desc") or r.get("疾病简介", "")
                        st.markdown(f"**{name}**")
                        if desc:
                            st.caption(str(desc)[:300])
                        st.divider()
            if result.get("toyhom_results"):
                with st.expander("相似问答结果"):
                    for r in result["toyhom_results"][:5]:
                        st.markdown(f"**Q:** {r.get('question', '')}")
                        ans = r.get("answer", "")
                        if ans:
                            st.caption(str(ans)[:300])
                        st.divider()

            current_messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "route": result.get("route", {}),
                    "kg_results": result.get("kg_results", []),
                    "toyhom_results": result.get("toyhom_results", []),
                }
            )

    st.session_state.messages[active_idx] = current_messages
