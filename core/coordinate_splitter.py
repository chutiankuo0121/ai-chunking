"""
CoordinateSplitter — 唯一的坐标切分器

职责:
1. 在文本中插入坐标标记 (每隔 ~interval tokens)
2. 调用 coordinate_split Tool 让 LLM 决定语义切割点
3. 使用 CascadeMatcher 定位 LLM 返回的 end_quote 以物理切割文本
4. 通过 ProtocolMemory 注入全局上下文、记录前文摘要
"""

import logging
import os
import re
from typing import List, Dict, Callable, Awaitable, Optional, AsyncGenerator

from pydantic import BaseModel, Field
import tiktoken

from .memory import ProtocolMemory
from .tools import coordinate_split, quick_summary, ChunkResult
from ..utils.matcher import CascadeMatcher

logger = logging.getLogger("ai_chunking.core")


# ============================================================
# 数据模型
# ============================================================

class SplitterInput(BaseModel):
    """切分器的输入参数。"""
    file_path: str = Field(..., description="目标 markdown 文件路径")
    target_tokens: int = Field(..., description="每块的目标 tokens 数")
    max_llm_context: int = Field(..., description="LLM 的最大上下文窗口 (tokens)")


class ChunkMetadata(BaseModel):
    """切分后每个块的元数据。"""
    chunk_id: str
    title: str
    summary: str
    synthetic_qa: list
    content: str
    length_tokens: int


# ============================================================
# CoordinateSplitter
# ============================================================

class CoordinateSplitter:
    """
    使用坐标系统的语义切分器。

    特性:
    1. 基于坐标的切割: 使用嵌入的 [ID: N] 标记进行精确切割。
    2. 全局上下文感知: 通过 ProtocolMemory 获取术语表 + 叙事大纲 + 前文摘要。
    3. Tool 化的 LLM 调用: 切分、摘要等 LLM 能力全部封装在 tools.py 中。
    4. 混合鲁棒性: CascadeMatcher 处理 LLM 的 end_quote 定位。
    5. 批处理: 按 batch 处理超长文本。
    """

    def __init__(
        self,
        llm_func: Optional[Callable[[str], Awaitable[str]]] = None,
        protocol_memory: Optional[ProtocolMemory] = None,
        encoding_model: str = "cl100k_base",
    ):
        self.llm_func = llm_func
        self.protocol_memory = protocol_memory

        try:
            self.tokenizer = tiktoken.get_encoding(encoding_model)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.matcher = CascadeMatcher()

    # ── Token 工具 ──────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _add_coordinates(self, text: str, interval: int) -> str:
        """每隔 ~interval tokens 插入 [ID: 0], [ID: 1]..."""
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)

        parts = []
        current_idx = 0
        marker_id = 0

        while current_idx < total_tokens:
            end_idx = min(current_idx + interval, total_tokens)
            chunk_tokens = tokens[current_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            parts.append(f"[ID: {marker_id}]")
            parts.append("\n" + chunk_text + "\n")

            marker_id += 1
            current_idx = end_idx

        return "".join(parts)

    # ── 入口点 ─────────────────────────────────────────────────

    async def split_file(self, params: SplitterInput) -> AsyncGenerator[ChunkMetadata, None]:
        """流式切分入口点。逐块 yield ChunkMetadata。"""
        # logger.info(f"🚀 启动坐标切分: {params.file_path}")
        logger.info("🏭 智能分块中... (Intelligent Chunking...)")
        logger.debug(f"Starting coordinate splitting: {params.file_path}")

        if not os.path.exists(params.file_path):
            raise FileNotFoundError(f"文件未找到: {params.file_path}")

        with open(params.file_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        async for c in self.split_text(full_text, params):
            meta = ChunkMetadata(
                chunk_id=str(c["id"]),
                title=c["title"],
                summary=c["summary"],
                synthetic_qa=c.get("synthetic_qa", []),
                content=c["content"],
                length_tokens=c["tokens"],
            )
            yield meta

    async def split_text(self, text: str, params: SplitterInput) -> AsyncGenerator[Dict, None]:
        """
        切分文本。经过三阶段异步管线:
          Orchestrator → [Queue] → Merger → [Queue] → Splitter → Consumer

        管线特性:
        - 真流式: 块就绪即输出，前端实时可见
        - 确定性兜底: 过小块合并、过大块二分，纯确定性不调 LLM
        - 背压控制: asyncio.Queue(maxsize) 自动限流
        - 完成验证: 结束后可通过 splitter.last_validation 获取合规报告
        """
        from .orchestrator import split_text as orchestrate_split
        from .pipeline import ChunkPipeline

        pipeline = ChunkPipeline(target_tokens=params.target_tokens)

        async for chunk in pipeline.run(orchestrate_split(self, text, params)):
            yield chunk

        # 管线完成后，暴露验证报告
        self.last_validation = pipeline.validation

    # ── 辅助: 快速摘要 (委托给 Tool) ──────────────────────────

    async def _quick_summary(self, text: str) -> Dict:
        """为结构化直接保留的 Chunk 生成标题和摘要。"""
        if not self.llm_func:
            return {"title": "无标题", "summary": "未提供 LLM 函数。"}
        return await quick_summary(text, self.llm_func)

    # ── 核心: 坐标切分 ─────────────────────────────────────────

    async def _process_large_block(
        self,
        text: str,
        params: SplitterInput,
        marker_interval: int,
        batch_size_tokens: int,
    ) -> AsyncGenerator[Dict, None]:
        """对超过 target_tokens 的大文本块执行坐标切分。"""
        if not self.llm_func:
            raise ValueError("未提供 LLM 函数。")

        target_chunk_tokens = params.target_tokens

        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        current_token_pos = 0

        while current_token_pos < total_tokens:
            # 1. 准备 batch
            batch_end_pos = min(current_token_pos + batch_size_tokens, total_tokens)

            # 避免最后剩一小段
            remaining_tokens = total_tokens - batch_end_pos
            if 0 < remaining_tokens < target_chunk_tokens // 3:
                batch_end_pos = total_tokens

            batch_tokens = tokens[current_token_pos:batch_end_pos]
            batch_text_raw = self.tokenizer.decode(batch_tokens)

            # 2. 插入坐标标记
            marked_batch_text = self._add_coordinates(batch_text_raw, marker_interval)
            markers_per_chunk = max(target_chunk_tokens // marker_interval, 1)

            # 3. 获取全局上下文
            global_context = "无前文语境。"
            if self.protocol_memory:
                global_context = self.protocol_memory.get_running_context()

            # 4. 调用 coordinate_split Tool
            logger.debug(f"Processing batch {current_token_pos}-{batch_end_pos} tokens...")
            try:
                llm_results: List[ChunkResult] = await coordinate_split(
                    marked_text=marked_batch_text,
                    global_context=global_context,
                    marker_interval=marker_interval,
                    target_chunk_tokens=target_chunk_tokens,
                    markers_per_chunk=markers_per_chunk,
                    llm_func=self.llm_func,
                )
            except Exception as e:
                logger.error(f"coordinate_split Tool 失败: {e}")
                raise e

            # 5. 物理切割 (CascadeMatcher)
            batch_offset = 0

            for res in llm_results:
                search_context = batch_text_raw[batch_offset:]

                quote_rel_idx = -1
                if res.end_quote and len(res.end_quote.strip()) > 5:
                    quote_rel_idx = self.matcher.find_anchor(search_context, res.end_quote)

                if quote_rel_idx == -1:
                    logger.warning(f"引用未找到: '{res.end_quote[:50]}...'. 使用标记回退。")
                    try:
                        m_id = int(re.search(r'\d+', res.end_marker).group())
                        approx_token_offset = m_id * marker_interval
                        slice_tokens = batch_tokens[:approx_token_offset]
                        end_char_idx = len(self.tokenizer.decode(slice_tokens))
                    except Exception:
                        end_char_idx = len(batch_text_raw)
                else:
                    end_char_idx = batch_offset + quote_rel_idx

                chunk_content = batch_text_raw[batch_offset:end_char_idx]

                if len(chunk_content.strip()) > 0:
                    yield {
                        "id": res.chunk_id,
                        "title": res.title,
                        "summary": res.summary,
                        "synthetic_qa": getattr(res, "synthetic_qa", []),
                        "content": chunk_content,
                        "tokens": self._count_tokens(chunk_content),
                    }

                    if self.protocol_memory and res.summary:
                        self.protocol_memory.add_chunk_summary(res.summary)

                batch_offset = end_char_idx

            # ── 处理剩余部分 ──
            if batch_offset < len(batch_text_raw):
                remaining = batch_text_raw[batch_offset:]
                remaining_tokens = self._count_tokens(remaining)

                if remaining_tokens > target_chunk_tokens:
                    # 大块剩余: 递归切分 (Orchestrator 不负责递归，只负责合并逻辑)
                    logger.debug(f"Remainder {remaining_tokens} > target, recursive split...")
                    async for sub in self._process_large_block(
                        remaining, params, marker_interval, batch_size_tokens
                    ):
                        yield sub
                else:
                    # 小/中剩余: 抛出给 Orchestrator 决定是否吸附
                    # 先不用生成 summary (如果被吸附就浪费了), 
                    # 但如果不吸附(中块)还是需要的。
                    # 策略: 先不生成，由 Orchestrator 在决定"不吸附"或"二分"时再补救生成。
                    yield {
                        "id": "remainder",
                        "title": "待定",
                        "summary": "",
                        "content": remaining,
                        "tokens": remaining_tokens,
                        "_is_remainder": True, 
                    }

            current_token_pos = batch_end_pos
