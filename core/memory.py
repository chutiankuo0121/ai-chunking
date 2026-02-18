"""
ProtocolMemory — 全局协议记忆

管理 Scanner 提取的全局知识（文档概述、叙事大纲），
以及 CoordinateSplitter 在切分过程中维护的前文摘要。

同时提供 load_prompt() 和 _safe_fill() 工具函数。
"""

import logging
from typing import List, Optional, Dict
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger("ai_chunking.core.memory")

# ── Prompt 加载 ─────────────────────────────────────────────

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """
    从 prompts/ 目录加载 Prompt 模板。
    
    参数:
        name: 模板名称 (不含 .txt 后缀)
    返回:
        模板字符串
    """
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        logger.warning(f"Prompt 文件未找到: {path}")
        return f"[Prompt '{name}' not found]"
    return path.read_text(encoding="utf-8")


def _safe_fill(template: str, **kwargs) -> str:
    """
    安全地填充 Prompt 模板中的 {key} 占位符。
    
    只替换 kwargs 中提供的 key，避免 KeyError。
    """
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", str(value))
    return template


# ============================================================
# 数据模型
# ============================================================

class NarrativeItem(BaseModel):
    """叙事大纲中的一个条目。"""
    section: str = Field(..., description="章节名称")
    goal: str = Field("", description="该章节的目标/作用")


class GlobalProtocol(BaseModel):
    """
    Scanner 提取的全局协议。
    
    包含文档概述和叙事大纲 — 
    这些信息在整个切分过程中保持不变，
    为 CoordinateSplitter 提供全局语义上下文。
    """
    doc_name: str = ""
    doc_overview: str = ""  # 纯文本概述，不含结构化数据
    narrative: List[NarrativeItem] = Field(default_factory=list)
    section_summaries: Dict[str, str] = Field(default_factory=dict)  # Phase 1.5: 详细章节摘要


# ============================================================
# ProtocolMemory
# ============================================================

class ProtocolMemory:
    """
    统一的记忆管理器。
    
    职责:
    1. 存储 Scanner 提取的 GlobalProtocol (写入方: Scanner)
    2. 提供全局上下文给 CoordinateSplitter (读取方: Splitter)
    3. 维护 running_context — 最近切分块的摘要列表
       (写入方: Splitter, 读取方: Splitter)
    """

    def __init__(self, protocol: Optional[GlobalProtocol] = None):
        self.protocol = protocol or GlobalProtocol()
        self._running_context: List[str] = []
        self._max_running_items = 10

    # ── Scanner 写入接口 ────────────────────────────────────

    def set_protocol(self, protocol: GlobalProtocol):
        """设置全局协议 (由 Scanner 在 Phase 1 完成后调用)。"""
        self.protocol = protocol

    def add_narrative(self, item: NarrativeItem):
        """追加一个叙事条目。"""
        self.protocol.narrative.append(item)

    def add_section_summary(self, section_title: str, summary: str):
        """记录章节详细摘要 (Phase 1.5 调用)。"""
        self.protocol.section_summaries[section_title] = summary

    # ── Splitter 写入接口 ───────────────────────────────────

    def add_chunk_summary(self, summary: str):
        """
        追加一条切分块摘要到 running_context。
        """
        if summary and summary.strip():
            self._running_context.append(summary.strip())
            if len(self._running_context) > self._max_running_items:
                self._running_context = self._running_context[-self._max_running_items:]

    # ── 读取接口 ────────────────────────────────────────────

    def get_running_context(self) -> str:
        """
        返回合并后的全局上下文字符串。
        
        包含: 文档概述 + 叙事大纲 + 最近的块摘要。
        用于注入到 coordinate_splitter 的 Prompt 中。
        """
        parts = []

        # 文档概述
        if self.protocol.doc_overview:
            parts.append("【文档概述】\n" + self.protocol.doc_overview)

        # 叙事大纲
        if self.protocol.narrative:
            sections = [f"- {n.section}: {n.goal}" for n in self.protocol.narrative]
            parts.append("【叙事大纲】\n" + "\n".join(sections))

        # 章节详细摘要 (Phase 1.5)
        if self.protocol.section_summaries:
            s_list = [f"## {k}\n{v}" for k, v in self.protocol.section_summaries.items()]
            joined_summaries = "\n\n".join(s_list)
            parts.append("【章节详细摘要】\n" + joined_summaries)

        # 前文摘要
        if self._running_context:
            recent = self._running_context[-self._max_running_items:]
            parts.append("【前文摘要（最近块）】\n" + "\n".join(f"- {s}" for s in recent))

        if not parts:
            return "无前文语境。"

        return "\n\n".join(parts)


    def stats(self) -> dict:
        """返回统计信息。"""
        return {
            "has_overview": bool(self.protocol.doc_overview),
            "narrative_count": len(self.protocol.narrative),
            "running_context_count": len(self._running_context),
        }

    def __repr__(self):
        s = self.stats()
        return (
            f"ProtocolMemory("
            f"overview={'✓' if s['has_overview'] else '✗'}, "
            f"narrative={s['narrative_count']}, "
            f"running={s['running_context_count']})"
        )
