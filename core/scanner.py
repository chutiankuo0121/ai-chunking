"""
ScannerEngine — Phase 1 全局扫描引擎

对文档执行全局扫描，提取:
- 文档概述 (doc_overview) — 纯文本，鲁棒
- 叙事大纲 (Narrative) — 结构化 JSON

结果存入 ProtocolMemory，供 Phase 2 的 CoordinateSplitter 使用。
"""

import asyncio
import logging
from typing import Callable, Awaitable, Optional

import tiktoken

from core.memory import ProtocolMemory, GlobalProtocol, load_prompt, _safe_fill
from core.tools import extract_narrative

logger = logging.getLogger("ai_chunking.core.scanner")


class ScannerEngine:
    """
    Phase 1 扫描引擎。
    
    流程:
    1. 生成全文概述 (doc_overview) — 纯文本，不需要 JSON 解析
    2. 对分片并行提取叙事大纲 (narrative)
    3. 合并到 ProtocolMemory
    
    参数:
        llm_func:         LLM 调用函数
        max_llm_context:  LLM 上下文窗口大小
        on_progress:      可选的进度回调 (用于 WebSocket 推送)
    """

    def __init__(
        self,
        llm_func: Callable[[str], Awaitable[str]],
        max_llm_context: int,
        on_progress: Optional[Callable] = None,
        encoding_model: str = "cl100k_base",
    ):
        self.llm_func = llm_func
        # 限制 chunk_size 上限为 12k，防止 output bottleneck 导致细节丢失
        self.chunk_size = min(int(max_llm_context * 0.33), 12000)
        self.on_progress = on_progress
        
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_model)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split_into_chunks(self, text: str) -> list:
        """将文本按 chunk_size token 数切片。"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks

    async def _generate_overview(self, text: str) -> str:
        """生成文档概述"""
        template = load_prompt("doc_overview")
        # 使用全文生成概述
        overview_text = text
        prompt = _safe_fill(template, text=overview_text)
        
        try:
            overview = await self.llm_func(prompt)
            logger.debug(f"📝 Doc overview generated ({len(overview)} chars)")
            return overview.strip()
        except Exception as e:
            logger.warning(f"Doc overview generation failed: {e}")
            return ""

    async def scan(self, text: str, doc_name: str = "") -> ProtocolMemory:
        """
        执行全局扫描，返回填充好的 ProtocolMemory。
        
        流程:
        1. 生成文档概述 (纯文本)
        2. 将文本切片，对每片提取叙事大纲
        3. 合并结果到 ProtocolMemory
        """
        # logger.info(f"🔍 Scanner starting for '{doc_name}' ({self._count_tokens(text):,} tokens)")
        logger.info("📄 AI总结文章中... (AI Summarizing Document...)")
        logger.debug(f"Scanner starting for '{doc_name}' ({self._count_tokens(text):,} tokens)")

        protocol = GlobalProtocol(doc_name=doc_name)
        memory = ProtocolMemory(protocol=protocol)

        # 切片
        chunks = self._split_into_chunks(text)
        logger.debug(f"Split into {len(chunks)} chunks for scanning")

        # ── 并行: 文档概述 + 叙事大纲 ──
        async def scan_narrative_chunk(i: int, chunk: str):
            """对单个 chunk 提取叙事大纲"""
            logger.debug(f"Scanning chunk {i+1}/{len(chunks)}...")
            narrative_items = await extract_narrative(chunk, self.llm_func)
            return i, narrative_items

        # 同时发出: 概述 + 所有 chunk 的叙事提取
        overview_task = self._generate_overview(text)
        narrative_tasks = [scan_narrative_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        
        all_results = await asyncio.gather(overview_task, *narrative_tasks)
        
        # 第一个结果是概述
        doc_overview = all_results[0]
        narrative_results = all_results[1:]

        # 设置概述
        memory.protocol.doc_overview = doc_overview
        if self.on_progress and doc_overview:
            await self.on_progress("protocol_update", {
                "category": "doc_overview",
                "content": doc_overview,
            })

        # ── 按顺序合并叙事结果 ──
        for i, narrative_items in sorted(narrative_results, key=lambda x: x[0]):
            for item in narrative_items:
                memory.add_narrative(item)

                if self.on_progress:
                    await self.on_progress("protocol_update", {
                        "category": "narrative",
                        "content": {
                            "section": item.section,
                            "goal": item.goal,
                        }
                    })

        stats = memory.stats()
        logger.debug(
            f"✅ Scanner complete: "
            f"overview={'✓' if doc_overview else '✗'}, "
            f"{stats['narrative_count']} sections"
        )

        return memory
