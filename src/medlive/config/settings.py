"""统一读取 LiveRAG Agent 的运行配置。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv

RagToolMode = Literal["auto", "never"]
_MASKED_SECRET_MARKER = "*****"


def load_environment() -> None:
    """按本地优先级加载环境变量。"""

    load_dotenv(".env.local", override=True)
    load_dotenv()


def _str_env(name: str, default: str = "") -> str:
    """读取字符串环境变量。"""

    return os.getenv(name, default).strip()


def _int_env(name: str, default: int) -> int:
    """读取整数环境变量。"""

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    """读取浮点数环境变量。"""

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _bool_env(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def mask_secret(value: str, *, prefix_chars: int = 2, suffix_chars: int = 8) -> str:
    """把密钥转换成前端可展示的掩码值。"""

    clean = value.strip()
    if not clean:
        return ""
    if len(clean) <= prefix_chars + suffix_chars:
        short_suffix = min(2, max(len(clean) - prefix_chars, 0))
        return f"{clean[:prefix_chars]}{_MASKED_SECRET_MARKER}{clean[-short_suffix:] if short_suffix else ''}"
    return f"{clean[:prefix_chars]}{_MASKED_SECRET_MARKER}{clean[-suffix_chars:]}"


def is_masked_secret(value: Any) -> bool:
    """判断前端提交值是否是后端返回的密钥掩码。"""

    return isinstance(value, str) and _MASKED_SECRET_MARKER in value


load_environment()


def _rag_tool_mode_env() -> RagToolMode:
    """读取 RAG 工具调用模式。"""

    value = _str_env("LIGHTRAG_TOOL_MODE", "auto")
    return cast(RagToolMode, value if value in {"auto", "never"} else "auto")


_RAG_RUNTIME_FIELDS = {
    "enabled",
    "base_url",
    "api_key",
    "query_mode",
    "timeout_ms",
    "top_k",
    "chunk_top_k",
    "context_max_chars",
    "cache_ttl_s",
    "enable_rerank",
    "rag_tool_mode",
}

_MODEL_RUNTIME_FIELDS = {
    "voice": {
        "stt": {"provider", "model", "app_id", "access_token"},
        "llm": {"model", "base_url", "api_key"},
        "tts": {"provider", "model", "voice", "api_key"},
    }
}

_STT_PROVIDER_OPTIONS = [
    {
        "provider": "volcengine_bigmodel",
        "label": "火山引擎 BigModel STT",
        "description": "当前 LiveRAG 已适配的实时语音识别 provider。",
        "models": [{"id": "bigmodel", "label": "bigmodel", "verified": True}],
        "default_model": "bigmodel",
        "config_fields": [
            {"key": "app_id", "label": "App ID", "type": "secret", "required": True},
            {"key": "access_token", "label": "Access Token", "type": "secret", "required": True},
        ],
    },
    {
        "provider": "stepfun",
        "label": "StepFun StepAudio 2.5 ASR",
        "description": "通过 Step Plan HTTP+SSE 接口，在 LiveKit VAD 切句后准实时识别。",
        "models": [
            {
                "id": "stepaudio-2.5-asr",
                "label": "stepaudio-2.5-asr",
                "verified": False,
            }
        ],
        "default_model": "stepaudio-2.5-asr",
        "config_fields": [
            {
                "key": "access_token",
                "label": "Step Plan API Key",
                "type": "secret",
                "required": True,
            }
        ],
    },
]


def _verified_voice_options(
    *voice_ids: str,
    metadata: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """生成已经通过真实合成验证的 voice 选项。"""

    metadata = metadata or {}
    options: list[dict[str, Any]] = []
    for voice_id in voice_ids:
        meta = metadata.get(voice_id, {})
        name = meta.get("name", "").strip()
        label = meta.get("label", "").strip()
        if not label:
            label = f"{name}（{voice_id}）" if name else voice_id
        option: dict[str, Any] = {"id": voice_id, "label": label, "verified": True}
        for key in ("name", "description", "language", "description_source"):
            value = meta.get(key, "").strip()
            if value:
                option[key] = value
        options.append(option)
    return options


_MINIMAX_VOICE_METADATA = {
    "socialmedia_female_2_v1": {
        "name": "社媒女声 2",
        "description": "社交媒体场景女声，适合短视频、直播和轻量口播。",
        "description_source": "derived",
    },
    "socialmedia_female_1_v1": {
        "name": "社媒女声 1",
        "description": "社交媒体场景女声，适合自然、亲切的实时语音对话。",
        "description_source": "derived",
    },
    "voice_agent_Female_Phone_4": {
        "name": "电话 Agent 女声 4",
        "description": "电话/客服场景女声，适合实时语音助手和通话交互。",
        "description_source": "derived",
    },
    "voice_agent_Male_Phone_1": {
        "name": "电话 Agent 男声 1",
        "description": "电话/客服场景男声，适合实时语音助手和通话交互。",
        "description_source": "derived",
    },
    "voice_agent_Male_Phone_2": {
        "name": "电话 Agent 男声 2",
        "description": "电话/客服场景男声，适合更沉稳的通话交互。",
        "description_source": "derived",
    },
    "English_StressedLady": {
        "name": "Stressed Lady",
        "description": "英语女声，紧张、急促的表达风格。",
        "language": "English",
        "description_source": "derived",
    },
    "English_SentimentalLady": {
        "name": "Sentimental Lady",
        "description": "英语女声，情绪化、感性的表达风格。",
        "language": "English",
        "description_source": "derived",
    },
    "English_radiant_girl": {
        "name": "Radiant Girl",
        "description": "英语女声，明亮、年轻、有活力。",
        "language": "English",
        "description_source": "derived",
    },
    "English_WiseScholar": {
        "name": "Wise Scholar",
        "description": "英语男声，智慧学者风格，适合解释和知识类内容。",
        "language": "English",
        "description_source": "derived",
    },
    "English_Persuasive_Man": {
        "name": "Persuasive Man",
        "description": "英语男声，说服力强，适合介绍和观点表达。",
        "language": "English",
        "description_source": "derived",
    },
    "English_Explanatory_Man": {
        "name": "Explanatory Man",
        "description": "英语男声，解释型表达，适合教学和说明。",
        "language": "English",
        "description_source": "derived",
    },
    "English_Insightful_Speaker": {
        "name": "Insightful Speaker",
        "description": "英语说话人，沉稳、有洞察力，适合知识型对话。",
        "language": "English",
        "description_source": "derived",
    },
    "japanese_male_social_media_1_v2": {
        "name": "日语社媒男声",
        "description": "日语男声，适合社交媒体和轻量口播。",
        "language": "Japanese",
        "description_source": "derived",
    },
    "japanese_female_social_media_1_v2": {
        "name": "日语社媒女声",
        "description": "日语女声，适合社交媒体和轻量口播。",
        "language": "Japanese",
        "description_source": "derived",
    },
    "French_CasualMan": {
        "name": "Casual Man",
        "description": "一位悠闲放松的中年男性法语声音。",
        "language": "French",
        "description_source": "official",
    },
    "French_Female Journalist": {
        "name": "Female Journalist",
        "description": "法语女声，新闻记者风格，适合播报和说明。",
        "language": "French",
        "description_source": "derived",
    },
    "Spanish_Narrator": {
        "name": "Narrator",
        "description": "一位适合叙述的中年女性叙述者声音，西班牙语。",
        "language": "Spanish",
        "description_source": "official",
    },
    "Spanish_WiseScholar": {
        "name": "Wise Scholar",
        "description": "一位亲切健谈的青年男性智慧学者声音，西班牙语。",
        "language": "Spanish",
        "description_source": "official",
    },
    "Spanish_ThoughtfulMan": {
        "name": "Thoughtful Man",
        "description": "一位冷静、体贴的青年男性声音，西班牙语。",
        "language": "Spanish",
        "description_source": "official",
    },
    "Arabic_CalmWoman": {
        "name": "Calm Woman",
        "description": "一位宁静的青年女性阿拉伯语声音。",
        "language": "Arabic",
        "description_source": "official",
    },
    "Arabic_FriendlyGuy": {
        "name": "Friendly Guy",
        "description": "一位沉稳友好的青年男性阿拉伯语声音。",
        "language": "Arabic",
        "description_source": "official",
    },
    "Portuguese_ThoughtfulLady": {
        "name": "Thoughtful Lady",
        "description": "一位忧虑体贴的中年女士声音，葡萄牙语。",
        "language": "Portuguese",
        "description_source": "official",
    },
    "German_PlayfulMan": {
        "name": "Playful Man",
        "description": "一位活泼的青年男性德语声音。",
        "language": "German",
        "description_source": "official",
    },
    "German_SweetLady": {
        "name": "Sweet Lady",
        "description": "一位灵动甜美的青年女性德语声音。",
        "language": "German",
        "description_source": "official",
    },
    "moss_audio_7c7e7ae2-7356-11f0-9540-7ef9b4b62566": {
        "name": "MiniMax 自定义音色 1",
        "description": "当前账号可用的 MiniMax moss_audio 音色，已通过真实合成验证。",
        "description_source": "derived",
    },
    "moss_audio_b118f320-78c0-11f0-bbeb-26e8167c4779": {
        "name": "MiniMax 自定义音色 2",
        "description": "当前账号可用的 MiniMax moss_audio 音色，已通过真实合成验证。",
        "description_source": "derived",
    },
    "moss_audio_84f32de9-2363-11f0-b7ab-d255fae1f27b": {
        "name": "MiniMax 自定义音色 3",
        "description": "当前账号可用的 MiniMax moss_audio 音色，已通过真实合成验证。",
        "description_source": "derived",
    },
    "moss_audio_82ebf67c-78c8-11f0-8e8e-36b92fbb4f95": {
        "name": "MiniMax 自定义音色 4",
        "description": "当前账号可用的 MiniMax moss_audio 音色，已通过真实合成验证。",
        "description_source": "derived",
    },
}


_DASHSCOPE_VOICE_METADATA = {
    "Cherry": {
        "name": "芊悦",
        "description": "阳光积极、亲切自然小姐姐（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Serena": {
        "name": "苏瑶",
        "description": "温柔小姐姐（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Ethan": {
        "name": "晨煦",
        "description": "标准普通话，带部分北方口音。阳光、温暖、活力、朝气（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Chelsie": {
        "name": "千雪",
        "description": "二次元虚拟女友（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Momo": {
        "name": "茉兔",
        "description": "撒娇搞怪，逗你开心（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Vivian": {
        "name": "十三",
        "description": "拽拽的、可爱的小暴躁（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Moon": {
        "name": "月白",
        "description": "率性帅气的月白（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Maia": {
        "name": "四月",
        "description": "知性与温柔的碰撞（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Kai": {
        "name": "凯",
        "description": "耳朵的一场 SPA（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Nofish": {
        "name": "不吃鱼",
        "description": "不会翘舌音的设计师（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Bella": {
        "name": "萌宝",
        "description": "喝酒不打醉拳的小萝莉（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Jennifer": {
        "name": "詹妮弗",
        "description": "品牌级、电影质感般美语女声（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Ryan": {
        "name": "甜茶",
        "description": "节奏拉满，戏感炸裂，真实与张力共舞（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Katerina": {
        "name": "卡捷琳娜",
        "description": "御姐音色，韵律回味十足（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Aiden": {
        "name": "艾登",
        "description": "精通厨艺的美语大男孩（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Eldric Sage": {
        "name": "沧明子",
        "description": "沉稳睿智的老者，沧桑如松却心明如镜（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Mia": {
        "name": "乖小妹",
        "description": "温顺如春水，乖巧如初雪（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Mochi": {
        "name": "沙小弥",
        "description": "聪明伶俐的小大人，童真未泯却早慧如禅（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Bellona": {
        "name": "燕铮莺",
        "description": "声音洪亮，吐字清晰，人物鲜活，听得人热血沸腾（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Vincent": {
        "name": "田叔",
        "description": "一口独特的沙哑烟嗓，道尽千军万马与江湖豪情（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Bunny": {
        "name": "萌小姬",
        "description": "“萌属性”爆棚的小萝莉（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Neil": {
        "name": "阿闻",
        "description": "平直语调、字正腔圆，专业新闻主持人风格（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Elias": {
        "name": "墨讲师",
        "description": "保持学科严谨性，适合把复杂知识讲清楚（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Arthur": {
        "name": "徐大爷",
        "description": "质朴沧桑、不疾不徐的乡土叙事男声（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Nini": {
        "name": "邻家妹妹",
        "description": "软糯甜美的邻家妹妹声线（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Seren": {
        "name": "小婉",
        "description": "温和舒缓的助眠声线（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Pip": {
        "name": "顽屁小孩",
        "description": "调皮捣蛋、充满童真的男孩声（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Stella": {
        "name": "少女阿月",
        "description": "甜美迷糊的少女音，也能表达正义感和张力（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Bodega": {
        "name": "博德加",
        "description": "热情的西班牙大叔（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Sonrisa": {
        "name": "索尼莎",
        "description": "热情开朗的拉美大姐（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Alek": {
        "name": "阿列克",
        "description": "战斗民族的冷与毛呢大衣下的暖（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Dolce": {
        "name": "多尔切",
        "description": "慵懒的意大利大叔（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Sohee": {
        "name": "素熙",
        "description": "温柔开朗，情绪丰富的韩国欧尼（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Ono Anna": {
        "name": "小野杏",
        "description": "鬼灵精怪的青梅竹马（女性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Lenn": {
        "name": "莱恩",
        "description": "理性底色里带一点叛逆的德国青年（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Emilien": {
        "name": "埃米尔安",
        "description": "浪漫的法国大哥哥（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Andre": {
        "name": "安德雷",
        "description": "磁性、自然舒服、沉稳的男声（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Radio Gol": {
        "name": "拉迪奥·戈尔",
        "description": "足球诗人和足球解说风格（男性）",
        "language": "中文（普通话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Jada": {
        "name": "上海-阿珍",
        "description": "风风火火的沪上阿姐（女性）",
        "language": "中文（上海话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Dylan": {
        "name": "北京-晓东",
        "description": "北京胡同里长大的少年（男性）",
        "language": "中文（北京话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Li": {
        "name": "南京-老李",
        "description": "耐心的瑜伽老师（男性）",
        "language": "中文（南京话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Marcus": {
        "name": "陕西-秦川",
        "description": "面宽话短、心实声沉的老陕味道（男性）",
        "language": "中文（陕西话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Roy": {
        "name": "闽南-阿杰",
        "description": "诙谐直爽、市井活泼的台湾哥仔形象（男性）",
        "language": "中文（闽南语）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Peter": {
        "name": "天津-李彼得",
        "description": "天津相声，专业捧哏（男性）",
        "language": "中文（天津话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Sunny": {
        "name": "四川-晴儿",
        "description": "甜到心里的川妹子（女性）",
        "language": "中文（四川话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Eric": {
        "name": "四川-程川",
        "description": "跳脱市井的四川成都男子（男性）",
        "language": "中文（四川话）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Rocky": {
        "name": "粤语-阿强",
        "description": "幽默风趣的阿强，在线陪聊（男性）",
        "language": "中文（粤语）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
    "Kiki": {
        "name": "粤语-阿清",
        "description": "甜美的港妹闺蜜（女性）",
        "language": "中文（粤语）、英语、法语、德语、俄语、意大利语、西班牙语、葡萄牙语、日语、韩语",
        "description_source": "official",
    },
}


_TTS_PROVIDER_OPTIONS = [
    {
        "provider": "minimax",
        "label": "MiniMax TTS",
        "description": "当前线上默认 TTS provider，已在 LiveRAG 实时链路验证。",
        "models": [{"id": "speech-02-turbo", "label": "speech-02-turbo", "verified": True}],
        "voices": _verified_voice_options(
            "socialmedia_female_2_v1",
            "socialmedia_female_1_v1",
            "voice_agent_Female_Phone_4",
            "voice_agent_Male_Phone_1",
            "voice_agent_Male_Phone_2",
            "English_StressedLady",
            "English_SentimentalLady",
            "English_radiant_girl",
            "English_WiseScholar",
            "English_Persuasive_Man",
            "English_Explanatory_Man",
            "English_Insightful_Speaker",
            "japanese_male_social_media_1_v2",
            "japanese_female_social_media_1_v2",
            "French_CasualMan",
            "French_Female Journalist",
            "Spanish_Narrator",
            "Spanish_WiseScholar",
            "Spanish_ThoughtfulMan",
            "Arabic_CalmWoman",
            "Arabic_FriendlyGuy",
            "Portuguese_ThoughtfulLady",
            "German_PlayfulMan",
            "German_SweetLady",
            "moss_audio_7c7e7ae2-7356-11f0-9540-7ef9b4b62566",
            "moss_audio_b118f320-78c0-11f0-bbeb-26e8167c4779",
            "moss_audio_84f32de9-2363-11f0-b7ab-d255fae1f27b",
            "moss_audio_82ebf67c-78c8-11f0-8e8e-36b92fbb4f95",
            metadata=_MINIMAX_VOICE_METADATA,
        ),
        "default_model": "speech-02-turbo",
        "default_voice": "socialmedia_female_1_v1",
        "config_fields": [
            {"key": "api_key", "label": "API Key", "type": "secret", "required": True}
        ],
    },
    {
        "provider": "dashscope_realtime",
        "label": "阿里 DashScope Qwen Realtime TTS",
        "description": "已适配 qwen3-tts 实时 WebSocket 链路，固定使用后端内置 endpoint。",
        "models": [
            {
                "id": "qwen3-tts-flash-realtime",
                "label": "qwen3-tts-flash-realtime",
                "verified": True,
            },
            {
                "id": "qwen3-tts-instruct-flash-realtime",
                "label": "qwen3-tts-instruct-flash-realtime",
                "verified": True,
            },
            {"id": "qwen-tts-realtime", "label": "qwen-tts-realtime", "verified": True},
        ],
        "voices": _verified_voice_options(
            "Cherry",
            "Serena",
            "Ethan",
            "Chelsie",
            "Momo",
            "Vivian",
            "Moon",
            "Maia",
            "Kai",
            "Nofish",
            "Bella",
            "Jennifer",
            "Ryan",
            "Katerina",
            "Aiden",
            "Eldric Sage",
            "Mia",
            "Mochi",
            "Bellona",
            "Vincent",
            "Bunny",
            "Neil",
            "Elias",
            "Arthur",
            "Nini",
            "Seren",
            "Pip",
            "Stella",
            "Bodega",
            "Sonrisa",
            "Alek",
            "Dolce",
            "Sohee",
            "Ono Anna",
            "Lenn",
            "Emilien",
            "Andre",
            "Radio Gol",
            "Jada",
            "Dylan",
            "Li",
            "Marcus",
            "Roy",
            "Peter",
            "Sunny",
            "Eric",
            "Rocky",
            "Kiki",
            metadata=_DASHSCOPE_VOICE_METADATA,
        ),
        "default_model": "qwen3-tts-flash-realtime",
        "default_voice": "Cherry",
        "config_fields": [
            {
                "key": "api_key",
                "label": "DashScope API Key",
                "type": "secret",
                "required": False,
                "description": "留空时后端默认复用 DASHSCOPE_API_KEY。",
            }
        ],
    },
    {
        "provider": "stepfun",
        "label": "StepFun StepAudio 2.5 TTS",
        "description": "已按 Step Plan WebSocket PCM 协议适配，待真实凭据语音验收。",
        "models": [
            {
                "id": "stepaudio-2.5-tts",
                "label": "stepaudio-2.5-tts",
                "verified": False,
            }
        ],
        "voices": [
            {
                "id": "cixingnansheng",
                "label": "磁性男声（cixingnansheng）",
                "verified": False,
            }
        ],
        "default_model": "stepaudio-2.5-tts",
        "default_voice": "cixingnansheng",
        "config_fields": [
            {
                "key": "api_key",
                "label": "Step Plan API Key",
                "type": "secret",
                "required": True,
            }
        ],
    },
]

_CONTEXT_MODEL_RUNTIME_FIELDS = {
    "model",
    "base_url",
    "api_key",
    "temperature",
    "max_tokens",
    "max_session_chars",
    "history_reference_limit",
    "timeout_ms",
}


def _runtime_rag_config_path(user_data_dir: Path | None = None) -> Path:
    """返回运行时 RAG 配置文件路径。"""

    root = user_data_dir or Path(_str_env("LIVERAG_USER_DATA_DIR", "~/.LiveRAG")).expanduser()
    return root / "rag" / "config.json"


def runtime_model_config_path(user_data_dir: Path | None = None) -> Path:
    """返回运行时语音模型配置文件路径。"""

    root = user_data_dir or Path(_str_env("LIVERAG_USER_DATA_DIR", "~/.LiveRAG")).expanduser()
    return root / "model" / "config.json"


def runtime_context_model_config_path(user_data_dir: Path | None = None) -> Path:
    """返回运行时上下文模型配置文件路径。"""

    root = user_data_dir or Path(_str_env("LIVERAG_USER_DATA_DIR", "~/.LiveRAG")).expanduser()
    return root / "model" / "context_config.json"


def _read_runtime_rag_overrides(user_data_dir: Path | None = None) -> dict[str, Any]:
    """读取前端 API 写入的 RAG 配置覆盖项。"""

    path = _runtime_rag_config_path(user_data_dir)
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if raw.endswith("\\n"):
            raw = raw[:-2].rstrip()
        payload = json.loads(raw)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    overrides = {key: value for key, value in payload.items() if key in _RAG_RUNTIME_FIELDS}
    if overrides.get("rag_tool_mode") not in {None, "auto", "never"}:
        overrides["rag_tool_mode"] = "auto"
    return overrides


def read_runtime_model_config(user_data_dir: Path | None = None) -> dict[str, Any]:
    """读取前端 API 写入的语音模型配置覆盖项。"""

    path = runtime_model_config_path(user_data_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return _filter_runtime_model_config(payload)


def write_runtime_model_config(config: dict[str, Any], user_data_dir: Path | None = None) -> None:
    """写入运行时语音模型配置。"""

    path = runtime_model_config_path(user_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = voice_config_for_storage(_filter_runtime_model_config(config))
    path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_runtime_context_model_config(user_data_dir: Path | None = None) -> dict[str, Any]:
    """读取前端 API 写入的上下文模型配置覆盖项。"""

    path = runtime_context_model_config_path(user_data_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return _filter_runtime_context_model_config(payload)


def write_runtime_context_model_config(
    config: dict[str, Any], user_data_dir: Path | None = None
) -> None:
    """写入运行时上下文模型配置。"""

    path = runtime_context_model_config_path(user_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = _filter_runtime_context_model_config(config)
    path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _filter_runtime_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    """只保留模型运行时配置支持的字段。"""

    voice_payload = payload.get("voice")
    if not isinstance(voice_payload, dict):
        return {}

    voice: dict[str, Any] = {}
    for section, allowed_fields in _MODEL_RUNTIME_FIELDS["voice"].items():
        section_payload = voice_payload.get(section)
        if not isinstance(section_payload, dict):
            continue
        values = {
            key: value
            for key, value in section_payload.items()
            if key in allowed_fields and isinstance(value, str)
        }
        if values:
            voice[section] = values
    return {"voice": voice} if voice else {}


def _filter_runtime_context_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    """只保留上下文模型运行时配置支持的字段。"""

    return {
        key: value
        for key, value in payload.items()
        if key in _CONTEXT_MODEL_RUNTIME_FIELDS and isinstance(value, (str, int, float))
    }


@dataclass(frozen=True)
class RagClientSettings:
    """语音链路访问 RAG 服务的配置。"""

    enabled: bool = _bool_env("LIGHTRAG_ENABLED", True)
    base_url: str = _str_env("LIGHTRAG_BASE_URL", "http://127.0.0.1:9721").rstrip("/")
    api_key: str = _str_env("LIGHTRAG_API_KEY", _str_env("KB_SERVICE_API_KEY", ""))
    query_mode: str = _str_env("LIGHTRAG_QUERY_MODE", _str_env("LIGHTRAG_VOICE_MODE", "naive"))
    timeout_ms: int = _int_env("LIGHTRAG_TIMEOUT_MS", 900)
    top_k: int = _int_env("LIGHTRAG_TOP_K", _int_env("LIGHTRAG_VOICE_TOP_K", 4))
    chunk_top_k: int = _int_env("LIGHTRAG_CHUNK_TOP_K", _int_env("LIGHTRAG_VOICE_CHUNK_TOP_K", 4))
    context_max_chars: int = _int_env(
        "LIGHTRAG_CONTEXT_MAX_CHARS",
        _int_env("LIGHTRAG_VOICE_CONTEXT_MAX_CHARS", 1800),
    )
    cache_ttl_s: float = _float_env("LIGHTRAG_CACHE_TTL_S", 45.0)
    enable_rerank: bool = _bool_env("LIGHTRAG_VOICE_ENABLE_RERANK", False)
    rag_tool_mode: RagToolMode = field(default_factory=_rag_tool_mode_env)

    def __post_init__(self) -> None:
        """校验 RAG 工具调用模式。"""

        if self.rag_tool_mode not in {"auto", "never"}:
            raise ValueError("rag_tool_mode must be one of: auto, never")


@dataclass(frozen=True)
class VoiceSettings:
    """实时语音模型配置。"""

    livekit_url: str = ""
    stt_provider: str = "volcengine_bigmodel"
    stt_app_id: str = ""
    stt_access_token: str = ""
    stt_model: str = "bigmodel"
    stt_base_url: str = ""
    llm_model: str = "qwen-flash"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = ""
    tts_provider: str = "minimax"
    tts_model: str = "speech-02-turbo"
    tts_voice: str = "socialmedia_female_1_v1"
    tts_api_key: str = ""
    tts_base_url: str = "https://api.minimax.chat"


@dataclass(frozen=True)
class ContextModelSettings:
    """通话历史压缩和知识库概览生成使用的模型配置。"""

    model: str = "qwen-max"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    max_tokens: int = 2000
    max_session_chars: int = 16000
    history_reference_limit: int = 8
    timeout_ms: int = 15000
    temperature: float = 0.0


@dataclass(frozen=True)
class ApiSettings:
    """前端管理 API 的内部运行配置。"""

    rag_gateway_timeout_ms: int = 10000
    rag_gateway_upload_timeout_ms: int = 30000
    rag_ready_timeout_ms: int = 15000
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_public_url: str = ""
    voice_session_token_ttl_s: int = 3600
    voice_session_created_ttl_s: int = 120
    voice_session_idle_ttl_s: int = 300
    voice_session_ending_ttl_s: int = 120
    voice_session_cleanup_interval_s: int = 60


def _env_voice_settings() -> VoiceSettings:
    """读取环境变量中的语音模型默认配置。"""

    stt_provider = _str_env("VOICE_STT_PROVIDER", "volcengine_bigmodel").lower()
    tts_provider = _canonical_tts_provider(_str_env("VOICE_TTS_PROVIDER", "minimax"))
    stepfun_key = _str_env("STEPFUN_API_KEY", "")
    llm_base_url = _str_env(
        "VOICE_LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")
    llm_api_key_fallback = (
        stepfun_key
        if "api.stepfun.com/step_plan/" in f"{llm_base_url}/"
        else _str_env("DASHSCOPE_API_KEY", "")
    )
    return VoiceSettings(
        livekit_url=_str_env("LIVEKIT_URL", ""),
        stt_provider=stt_provider,
        stt_app_id=("" if stt_provider == "stepfun" else _str_env("VOLCENGINE_STT_APP_ID", "")),
        stt_access_token=(
            stepfun_key
            if stt_provider == "stepfun"
            else _str_env("VOLCENGINE_STT_ACCESS_TOKEN", "")
        ),
        stt_model=_str_env(
            "VOICE_STT_MODEL",
            (
                _str_env("STEPFUN_ASR_MODEL", "stepaudio-2.5-asr")
                if stt_provider == "stepfun"
                else _str_env("VOLCENGINE_BIGMODEL_STT_MODEL", "bigmodel")
            ),
        ),
        stt_base_url=(
            _str_env(
                "STEPFUN_BASE_URL",
                "https://api.stepfun.com/step_plan/v1",
            ).rstrip("/")
            if stt_provider == "stepfun"
            else ""
        ),
        llm_model=_str_env("VOICE_LLM_MODEL", "qwen-flash"),
        llm_base_url=llm_base_url,
        llm_api_key=_str_env("VOICE_LLM_API_KEY", llm_api_key_fallback),
        tts_provider=tts_provider,
        tts_model=_str_env("VOICE_TTS_MODEL") or _default_tts_model(tts_provider),
        tts_voice=_str_env("VOICE_TTS_VOICE") or _default_tts_voice(tts_provider),
        tts_api_key=_str_env("VOICE_TTS_API_KEY") or _default_tts_api_key(tts_provider),
        tts_base_url=_str_env("VOICE_TTS_BASE_URL") or _default_tts_base_url(tts_provider),
    )


def _override_str(config: dict[str, Any], section: str, key: str, fallback: str) -> str:
    """读取运行时覆盖值，缺失或空值时使用默认值。"""

    value = config.get("voice", {}).get(section, {}).get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def load_voice_settings(user_data_dir: Path | None = None) -> VoiceSettings:
    """按环境默认值和运行时配置覆盖项生成语音模型配置。"""

    base = _env_voice_settings()
    config = read_runtime_model_config(user_data_dir)
    stt_provider = _override_str(config, "stt", "provider", base.stt_provider).lower()
    stt_provider_changed = stt_provider != base.stt_provider
    stt_model_fallback = (
        _default_stt_model(stt_provider) if stt_provider_changed else base.stt_model
    )
    stt_token_fallback = (
        _default_stt_api_key(stt_provider) if stt_provider_changed else base.stt_access_token
    )
    tts_provider = _canonical_tts_provider(
        _override_str(config, "tts", "provider", base.tts_provider)
    )
    tts_api_key_fallback = _voice_tts_default(
        base.tts_api_key,
        tts_provider,
        "api_key",
        source_provider=base.tts_provider,
    )
    return VoiceSettings(
        livekit_url=base.livekit_url,
        stt_provider=stt_provider,
        stt_app_id=(
            _override_str(config, "stt", "app_id", base.stt_app_id)
            if stt_provider == "volcengine_bigmodel"
            else ""
        ),
        stt_access_token=_override_str(config, "stt", "access_token", stt_token_fallback),
        stt_model=_override_str(config, "stt", "model", stt_model_fallback),
        stt_base_url=_default_stt_base_url(stt_provider),
        llm_model=_override_str(config, "llm", "model", base.llm_model),
        llm_base_url=_override_str(config, "llm", "base_url", base.llm_base_url).rstrip("/"),
        llm_api_key=_override_str(config, "llm", "api_key", base.llm_api_key),
        tts_provider=tts_provider,
        tts_model=_override_str(
            config,
            "tts",
            "model",
            _voice_tts_default(
                base.tts_model,
                tts_provider,
                "model",
                source_provider=base.tts_provider,
            ),
        ),
        tts_voice=_override_str(
            config,
            "tts",
            "voice",
            _voice_tts_default(
                base.tts_voice,
                tts_provider,
                "voice",
                source_provider=base.tts_provider,
            ),
        ),
        tts_api_key=_override_str(config, "tts", "api_key", tts_api_key_fallback),
        tts_base_url=_override_str(
            config,
            "tts",
            "base_url",
            _voice_tts_default(
                base.tts_base_url,
                tts_provider,
                "base_url",
                source_provider=base.tts_provider,
            ),
        ).rstrip("/"),
    )


def _is_dashscope_tts_provider(provider: str) -> bool:
    """判断 TTS provider 是否走 DashScope Qwen realtime。"""

    return _canonical_tts_provider(provider) == "dashscope_realtime"


def _is_stepfun_tts_provider(provider: str) -> bool:
    """判断 TTS provider 是否使用 Step Plan。"""

    return _canonical_tts_provider(provider) == "stepfun"


def _canonical_tts_provider(provider: str) -> str:
    """归一化 TTS provider 别名。"""

    clean = provider.lower().strip()
    if clean in {"dashscope", "dashscope_realtime", "qwen_realtime"}:
        return "dashscope_realtime"
    if clean in {"stepfun", "stepfun_step_plan"}:
        return "stepfun"
    return clean


def _default_stt_model(provider: str) -> str:
    return "stepaudio-2.5-asr" if provider == "stepfun" else "bigmodel"


def _default_stt_api_key(provider: str) -> str:
    if provider == "stepfun":
        return _str_env("STEPFUN_API_KEY", "")
    return _str_env("VOLCENGINE_STT_ACCESS_TOKEN", "")


def _default_stt_base_url(provider: str) -> str:
    if provider == "stepfun":
        return _str_env(
            "STEPFUN_BASE_URL",
            "https://api.stepfun.com/step_plan/v1",
        ).rstrip("/")
    return ""


def _stt_option_ids(provider: str, field_name: str) -> set[str]:
    """读取 STT provider 的选项 ID 集合。"""

    for item in _STT_PROVIDER_OPTIONS:
        if item["provider"] == provider:
            values = item.get(field_name, [])
            if isinstance(values, list):
                return {str(value.get("id")) for value in values if isinstance(value, dict)}
    return set()


def _tts_option_ids(provider: str, field_name: str) -> set[str]:
    """读取 TTS provider 的选项 ID 集合。"""

    for item in _TTS_PROVIDER_OPTIONS:
        if item["provider"] == provider:
            values = item.get(field_name, [])
            if isinstance(values, list):
                return {str(value.get("id")) for value in values if isinstance(value, dict)}
    return set()


def _default_tts_model(provider: str) -> str:
    provider = _canonical_tts_provider(provider)
    if _is_stepfun_tts_provider(provider):
        return "stepaudio-2.5-tts"
    if _is_dashscope_tts_provider(provider):
        return "qwen3-tts-flash-realtime"
    return _str_env("MINIMAX_TTS_MODEL", "speech-02-turbo")


def _default_tts_voice(provider: str) -> str:
    provider = _canonical_tts_provider(provider)
    if _is_stepfun_tts_provider(provider):
        return "cixingnansheng"
    if _is_dashscope_tts_provider(provider):
        return "Cherry"
    return _str_env("MINIMAX_TTS_VOICE", "socialmedia_female_1_v1")


def _default_tts_api_key(provider: str) -> str:
    provider = _canonical_tts_provider(provider)
    if _is_stepfun_tts_provider(provider):
        return _str_env("STEPFUN_API_KEY", "")
    if _is_dashscope_tts_provider(provider):
        return _str_env("DASHSCOPE_API_KEY", "")
    return _str_env("MINIMAX_API_KEY", "")


def _default_tts_base_url(provider: str) -> str:
    provider = _canonical_tts_provider(provider)
    if _is_stepfun_tts_provider(provider):
        return _str_env(
            "STEPFUN_TTS_BASE_URL",
            "wss://api.stepfun.com/step_plan/v1/realtime/audio",
        )
    if _is_dashscope_tts_provider(provider):
        return "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    return _str_env("MINIMAX_BASE_URL", "https://api.minimax.chat")


def _voice_tts_default(
    current: str,
    provider: str,
    field_name: str,
    *,
    source_provider: str | None = None,
) -> str:
    """切换 provider 但未显式传字段时，使用目标 provider 的合理默认值。"""

    provider = _canonical_tts_provider(provider)
    if source_provider is not None and _canonical_tts_provider(source_provider) != provider:
        defaults = {
            "model": _default_tts_model,
            "voice": _default_tts_voice,
            "api_key": _default_tts_api_key,
            "base_url": _default_tts_base_url,
        }
        return defaults[field_name](provider)
    if not _is_dashscope_tts_provider(provider):
        return current
    if field_name == "model" and current in {"speech-02-turbo", "speech-01-turbo"}:
        return _default_tts_model(provider)
    if field_name == "voice" and current == "socialmedia_female_1_v1":
        return _default_tts_voice(provider)
    if field_name == "api_key" and not current:
        return _default_tts_api_key(provider)
    if field_name == "base_url" and current == "https://api.minimax.chat":
        return _default_tts_base_url(provider)
    return current


def _env_context_model_settings() -> ContextModelSettings:
    """读取环境变量中的上下文模型默认配置。"""

    return ContextModelSettings(
        model=_str_env("CONTEXT_MODEL_MODEL", "qwen-max"),
        base_url=_str_env(
            "CONTEXT_MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/"),
        api_key=_str_env("CONTEXT_MODEL_API_KEY", _str_env("DASHSCOPE_API_KEY", "")),
        max_tokens=_int_env("CONTEXT_MODEL_MAX_TOKENS", 2000),
        max_session_chars=_int_env("CONTEXT_MODEL_MAX_SESSION_CHARS", 16000),
        history_reference_limit=_int_env("CONTEXT_MODEL_HISTORY_REFERENCE_LIMIT", 8),
        timeout_ms=_int_env("CONTEXT_MODEL_TIMEOUT_MS", 15000),
        temperature=_float_env("CONTEXT_MODEL_TEMPERATURE", 0.0),
    )


def load_context_model_settings(user_data_dir: Path | None = None) -> ContextModelSettings:
    """按环境默认值和运行时配置覆盖项生成上下文模型配置。"""

    base = _env_context_model_settings()
    config = read_runtime_context_model_config(user_data_dir)
    return ContextModelSettings(
        model=str(config.get("model") or base.model).strip(),
        base_url=str(config.get("base_url") or base.base_url).strip().rstrip("/"),
        api_key=str(config.get("api_key") or base.api_key).strip(),
        max_tokens=int(config.get("max_tokens") or base.max_tokens),
        max_session_chars=int(config.get("max_session_chars") or base.max_session_chars),
        history_reference_limit=int(
            config.get("history_reference_limit") or base.history_reference_limit
        ),
        timeout_ms=int(config.get("timeout_ms") or base.timeout_ms),
        temperature=float(
            config.get("temperature") if config.get("temperature") is not None else base.temperature
        ),
    )


def load_api_settings() -> ApiSettings:
    """读取管理 API 配置。"""

    return ApiSettings(
        rag_gateway_timeout_ms=_int_env("LIVERAG_RAG_GATEWAY_TIMEOUT_MS", 10000),
        rag_gateway_upload_timeout_ms=_int_env("LIVERAG_RAG_GATEWAY_UPLOAD_TIMEOUT_MS", 30000),
        rag_ready_timeout_ms=_int_env("LIVERAG_RAG_READY_TIMEOUT_MS", 15000),
        livekit_api_key=_str_env("LIVEKIT_API_KEY", ""),
        livekit_api_secret=_str_env("LIVEKIT_API_SECRET", ""),
        livekit_public_url=_str_env("LIVEKIT_PUBLIC_URL", _str_env("LIVEKIT_URL", "")),
        voice_session_token_ttl_s=_int_env("VOICE_SESSION_TOKEN_TTL_S", 3600),
        voice_session_created_ttl_s=_int_env("VOICE_SESSION_CREATED_TTL_S", 120),
        voice_session_idle_ttl_s=_int_env("VOICE_SESSION_IDLE_TTL_S", 300),
        voice_session_ending_ttl_s=_int_env("VOICE_SESSION_ENDING_TTL_S", 120),
        voice_session_cleanup_interval_s=_int_env("VOICE_SESSION_CLEANUP_INTERVAL_S", 60),
    )


def public_voice_config(voice: VoiceSettings, *, effective: str) -> dict[str, Any]:
    """返回不含密钥的语音模型配置摘要。"""

    stt_app_id_masked = mask_secret(voice.stt_app_id, prefix_chars=4, suffix_chars=4)
    stt_access_token_masked = mask_secret(voice.stt_access_token)
    llm_api_key_masked = mask_secret(voice.llm_api_key)
    tts_api_key_masked = mask_secret(voice.tts_api_key)
    return {
        "stt": {
            "provider": voice.stt_provider,
            "model": voice.stt_model,
            "app_id_set": bool(voice.stt_app_id),
            "app_id_masked": stt_app_id_masked,
            "access_token_set": bool(voice.stt_access_token),
            "access_token": stt_access_token_masked,
            "access_token_masked": stt_access_token_masked,
            "effective": effective,
        },
        "llm": {
            "model": voice.llm_model,
            "base_url": voice.llm_base_url,
            "api_key_set": bool(voice.llm_api_key),
            "api_key": llm_api_key_masked,
            "api_key_masked": llm_api_key_masked,
            "effective": effective,
        },
        "tts": {
            "provider": voice.tts_provider,
            "model": voice.tts_model,
            "voice": voice.tts_voice,
            "api_key_set": bool(voice.tts_api_key),
            "api_key": tts_api_key_masked,
            "api_key_masked": tts_api_key_masked,
            "effective": effective,
        },
    }


def public_model_options() -> dict[str, Any]:
    """返回前端模型选择页使用的 provider、模型、音色和字段定义。"""

    return {
        "stt": {
            "providers": _STT_PROVIDER_OPTIONS,
            "default_provider": "volcengine_bigmodel",
        },
        "llm": {
            "mode": "manual",
            "description": "对话模型保持现有配置方式，前端继续填写 model、base_url 和 api_key。",
            "config_fields": [
                {"key": "model", "label": "Model", "type": "text", "required": True},
                {"key": "base_url", "label": "Base URL", "type": "url", "required": True},
                {"key": "api_key", "label": "API Key", "type": "secret", "required": True},
            ],
        },
        "tts": {
            "providers": _TTS_PROVIDER_OPTIONS,
            "default_provider": "minimax",
        },
    }


def voice_config_for_storage(config: dict[str, Any]) -> dict[str, Any]:
    """归一化运行时语音配置，移除前端不应管理的固定 provider 字段。"""

    voice = config.get("voice")
    if not isinstance(voice, dict):
        return config

    stt = voice.get("stt")
    if isinstance(stt, dict):
        stt_provider = str(stt.get("provider") or "volcengine_bigmodel").lower()
        stt["provider"] = stt_provider
        if "model" not in stt:
            stt["model"] = _default_stt_model(stt_provider)

    tts_section = voice.get("tts")
    if isinstance(tts_section, dict):
        provider = _canonical_tts_provider(str(tts_section.get("provider") or "minimax"))
        tts_section["provider"] = provider
        if not tts_section.get("model"):
            tts_section["model"] = _default_tts_model(provider)
        if not tts_section.get("voice"):
            tts_section["voice"] = _default_tts_voice(provider)
        tts_section.pop("base_url", None)
    return config


def validate_voice_config_selection(config: dict[str, Any]) -> None:
    """校验前端提交的 provider/model/voice 是否属于后端已适配列表。"""

    voice = config.get("voice")
    if not isinstance(voice, dict):
        return

    stt = voice.get("stt")
    if isinstance(stt, dict):
        stt_provider = str(stt.get("provider") or "volcengine_bigmodel").lower()
        if stt_provider not in {"volcengine_bigmodel", "stepfun"}:
            raise ValueError("voice.stt.provider must be volcengine_bigmodel or stepfun")
        stt_model = str(stt.get("model") or _default_stt_model(stt_provider))
        if stt_model not in _stt_option_ids(stt_provider, "models"):
            raise ValueError("voice.stt.model is not supported by selected provider")

    tts_section = voice.get("tts")
    if isinstance(tts_section, dict):
        provider = _canonical_tts_provider(str(tts_section.get("provider") or "minimax"))
        if provider not in {item["provider"] for item in _TTS_PROVIDER_OPTIONS}:
            raise ValueError("voice.tts.provider must be minimax, dashscope_realtime or stepfun")
        model = str(tts_section.get("model") or _default_tts_model(provider))
        if model not in _tts_option_ids(provider, "models"):
            raise ValueError("voice.tts.model is not supported by selected provider")
        voice_id = str(tts_section.get("voice") or _default_tts_voice(provider))
        if voice_id not in _tts_option_ids(provider, "voices"):
            raise ValueError("voice.tts.voice is not supported by selected provider")


def public_rag_client_config(config: RagClientSettings) -> dict[str, Any]:
    """返回不泄露明文密钥的 RAG 客户端配置。"""

    payload = dict(config.__dict__)
    api_key_masked = mask_secret(config.api_key)
    payload["api_key"] = api_key_masked
    payload["api_key_masked"] = api_key_masked
    payload["api_key_set"] = bool(config.api_key)
    return payload


def public_context_model_config(config: ContextModelSettings) -> dict[str, Any]:
    """返回不泄露明文密钥的上下文模型配置。"""

    api_key_masked = mask_secret(config.api_key)
    return {
        "model": config.model,
        "base_url": config.base_url,
        "api_key": api_key_masked,
        "api_key_masked": api_key_masked,
        "api_key_set": bool(config.api_key),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "max_session_chars": config.max_session_chars,
        "history_reference_limit": config.history_reference_limit,
        "timeout_ms": config.timeout_ms,
        "effective": "next_session",
    }


def load_rag_client_settings(user_data_dir: Path | None = None) -> RagClientSettings:
    """读取环境变量和运行时配置后的 RAG 客户端配置。"""

    base = RagClientSettings()
    overrides = _read_runtime_rag_overrides(user_data_dir)
    values = {**base.__dict__, **overrides}
    return RagClientSettings(**values)


@dataclass(frozen=True)
class AppSettings:
    """LiveRAG Agent 运行总配置。"""

    user_data_dir: Path = field(
        default_factory=lambda: Path(_str_env("LIVERAG_USER_DATA_DIR", "~/.LiveRAG")).expanduser()
    )
    log_dir: Path = field(
        default_factory=lambda: Path(_str_env("LIVERAG_LOG_DIR", "~/.LiveRAG/logs")).expanduser()
    )
    history_limit: int = field(default_factory=lambda: _int_env("LIVERAG_HISTORY_LIMIT", 8))
    voice: VoiceSettings = field(default_factory=load_voice_settings)
    rag: RagClientSettings = field(default_factory=load_rag_client_settings)
    context_model: ContextModelSettings = field(default_factory=load_context_model_settings)
    api: ApiSettings = field(default_factory=load_api_settings)


def load_app_settings() -> AppSettings:
    """重新读取环境变量和运行时配置后生成应用配置。"""

    load_environment()
    user_data_dir = Path(_str_env("LIVERAG_USER_DATA_DIR", "~/.LiveRAG")).expanduser()
    return AppSettings(
        user_data_dir=user_data_dir,
        log_dir=Path(_str_env("LIVERAG_LOG_DIR", "~/.LiveRAG/logs")).expanduser(),
        history_limit=_int_env("LIVERAG_HISTORY_LIMIT", 8),
        voice=load_voice_settings(user_data_dir),
        rag=load_rag_client_settings(user_data_dir),
        context_model=load_context_model_settings(user_data_dir),
        api=load_api_settings(),
    )
