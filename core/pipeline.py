"""
Pipeline — 基于 asyncio.Queue 的生产-消费分块管线

三阶段并发流水线:
  Producer (Orchestrator) → [Queue] → Merger → [Queue] → Splitter → [Queue] → Consumer

设计哲学:
1. 真流式: 块一旦就绪立刻传入下游，前端实时可见
2. 背压控制: Queue(maxsize) 限制内存占用，上游自动阻塞等待
3. 哨兵终止: _SENTINEL 对象标记管线结束，安全关闭
4. 阶段解耦: 每个 Stage 是独立 asyncio.Task，可单独测试/替换
5. 确定性保证: Merger + Splitter 纯确定性，不调 LLM
"""

import asyncio
import re
import logging
from typing import Dict, List, AsyncGenerator

import tiktoken

logger = logging.getLogger("ai_chunking.core.pipeline")

# 哨兵对象 — 放入队列表示 "上游已完成，无更多数据"
_SENTINEL = object()


class ChunkPipeline:
    """
    异步生产者-消费者分块管线。

    用法::

        pipeline = ChunkPipeline(target_tokens=3500)
        async for chunk in pipeline.run(orchestrator_output):
            print(chunk)
    """

    def __init__(
        self,
        target_tokens: int,
        encoding_model: str = "cl100k_base",
        queue_size: int = 32,
    ):
        self.target_tokens = target_tokens
        self.min_tokens = target_tokens // 3  # 过小阈值: 1/3 TARGET

        try:
            self.tokenizer = tiktoken.get_encoding(encoding_model)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self._queue_size = queue_size

        # 验证追踪器 — 记录每个输出 chunk 的合规性
        self._validation: Dict = {}
        self._reset_validation()

    def _reset_validation(self):
        """重置验证状态。每次 run() 调用前自动重置。"""
        self._validation = {
            "passed": True,
            "total": 0,
            "oversized": [],    # [(chunk_id, tokens), ...]
            "undersized": [],   # [(chunk_id, tokens), ...]
            "min_tokens_seen": float("inf"),
            "max_tokens_seen": 0,
        }

    @property
    def validation(self) -> Dict:
        """返回最近一次 run() 的验证报告。"""
        return self._validation

    def _track_chunk(self, chunk: Dict):
        """追踪单个 chunk 的合规性。"""
        v = self._validation
        tokens = chunk.get("tokens", 0)
        chunk_id = chunk.get("id", "?")

        v["total"] += 1
        v["min_tokens_seen"] = min(v["min_tokens_seen"], tokens)
        v["max_tokens_seen"] = max(v["max_tokens_seen"], tokens)

        if tokens > self.target_tokens:
            v["oversized"].append((chunk_id, tokens))
            v["passed"] = False
        elif tokens < self.min_tokens:
            v["undersized"].append((chunk_id, tokens))
            v["passed"] = False

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # ================================================================
    # Public API
    # ================================================================

    async def run(
        self, source: AsyncGenerator[Dict, None]
    ) -> AsyncGenerator[Dict, None]:
        """
        启动管线并流式输出最终 chunks。

        三个 Stage 作为 asyncio.Task 并发运行:
        - Producer: 消费上游 source → raw_q
        - Merger:   raw_q → merged_q (合并过小块)
        - Splitter: merged_q → output_q (二分过大块)

        管线结束后可通过 pipeline.validation 获取验证报告。
        """
        # 重置验证状态
        self._reset_validation()

        # 每次 run 创建新队列 (支持重复调用)
        raw_q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        merged_q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        output_q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)

        # 启动三阶段并发任务
        tasks = [
            asyncio.create_task(
                self._stage_producer(source, raw_q), name="pipeline-producer"
            ),
            asyncio.create_task(
                self._stage_merger(raw_q, merged_q), name="pipeline-merger"
            ),
            asyncio.create_task(
                self._stage_splitter(merged_q, output_q), name="pipeline-splitter"
            ),
        ]

        try:
            # 从输出队列消费 — 真流式 + 逐块验证
            while True:
                chunk = await output_q.get()
                if chunk is _SENTINEL:
                    break
                self._track_chunk(chunk)
                yield chunk
        finally:
            # 确保所有任务正常结束 (即使消费方提前退出)
            await asyncio.gather(*tasks, return_exceptions=True)

            # 输出验证摘要日志
            v = self._validation
            if v["passed"]:
                logger.debug(
                    f"✅ 验证通过: {v['total']} 个块均在 "
                    f"[{self.min_tokens}, {self.target_tokens}] tokens 范围内 "
                    f"(实际 {v['min_tokens_seen']}~{v['max_tokens_seen']})"
                )
            else:
                logger.warning(
                    f"🚫 验证失败: {len(v['oversized'])} 超标, "
                    f"{len(v['undersized'])} 过小 / 共 {v['total']} 块"
                )

    # ================================================================
    # Stage 0: Producer
    # ================================================================

    @staticmethod
    async def _stage_producer(
        source: AsyncGenerator[Dict, None],
        out_q: asyncio.Queue,
    ):
        """从上游 AsyncGenerator 消费，逐个放入队列。"""
        try:
            async for chunk in source:
                await out_q.put(chunk)
        except Exception as e:
            logger.error(f"Producer 异常: {e}")
        finally:
            await out_q.put(_SENTINEL)

    # ================================================================
    # Stage 1: Merger — 合并过小块
    # ================================================================

    async def _stage_merger(
        self,
        in_q: asyncio.Queue,
        out_q: asyncio.Queue,
    ):
        """
        流式合并器。

        规则:
        - 维护一个 buffer chunk
        - 如果 buffer.tokens < 1/3 TARGET → 吞并下一个 chunk，合并摘要
        - 如果 buffer 合格 → 推入下游，换新 buffer
        - 管线结束 → flush buffer
        """
        buf = None

        try:
            while True:
                chunk = await in_q.get()

                if chunk is _SENTINEL:
                    if buf is not None:
                        await out_q.put(buf)
                    break

                if buf is None:
                    buf = chunk.copy()
                    continue

                if buf["tokens"] < self.min_tokens:
                    # buffer 太小 → 吞并，合并摘要
                    buf["content"] += "\n\n" + chunk["content"]
                    buf["tokens"] = self._count_tokens(buf["content"])
                    buf["title"] += " & " + chunk["title"]
                    buf["summary"] = (
                        buf.get("summary", "") + " " + chunk.get("summary", "")
                    ).strip()
                    logger.debug(
                        f"Merged small chunk → buffer now {buf['tokens']} tokens"
                    )
                    continue

                # buffer 合格 → 推入下游
                await out_q.put(buf)
                buf = chunk.copy()

        except Exception as e:
            logger.error(f"Merger 异常: {e}")
        finally:
            await out_q.put(_SENTINEL)

    # ================================================================
    # Stage 2: Splitter — 二分过大块
    # ================================================================

    async def _stage_splitter(
        self,
        in_q: asyncio.Queue,
        out_q: asyncio.Queue,
    ):
        """
        流式拆分器。

        规则:
        - chunk.tokens > TARGET → 在中间句子边界递归二分
        - 标注 "（第N部分/共M部分）" 到摘要
        - 降级链: 段落 > 句子 > 换行 > 硬切
        """
        counter = 0

        try:
            while True:
                chunk = await in_q.get()

                if chunk is _SENTINEL:
                    break

                if chunk["tokens"] > self.target_tokens:
                    logger.debug(
                        f"✂️ 二分: '{chunk['title'][:30]}' "
                        f"({chunk['tokens']} > {self.target_tokens})"
                    )
                    pieces = self._recursive_binary_split(
                        chunk["content"], self.target_tokens
                    )
                    total = len(pieces)
                    original_summary = chunk.get("summary", "")

                    for i, piece in enumerate(pieces, 1):
                        counter += 1
                        label = (
                            f"(Part {i}/{total}) " if total > 1 else ""
                        )
                        await out_q.put(
                            {
                                "id": f"CK-{counter:03d}",
                                "title": chunk["title"],
                                "summary": f"{label}{original_summary}",
                                "content": piece,
                                "tokens": self._count_tokens(piece),
                            }
                        )
                else:
                    counter += 1
                    chunk["id"] = f"CK-{counter:03d}"
                    await out_q.put(chunk)

        except Exception as e:
            logger.error(f"Splitter 异常: {e}")
        finally:
            await out_q.put(_SENTINEL)

    # ================================================================
    # 二分拆分工具方法
    # ================================================================

    def _recursive_binary_split(self, text: str, target: int) -> List[str]:
        """
        递归二分。

        在 token 维度的中间位置找最近的句子/段落边界切一刀，
        直到每段 ≤ target_tokens。
        """
        if self._count_tokens(text) <= target:
            return [text]

        tokens = self.tokenizer.encode(text)
        mid_char = len(self.tokenizer.decode(tokens[: len(tokens) // 2]))

        split_pos = self._find_middle_boundary(text, mid_char)

        left = text[:split_pos].strip()
        right = text[split_pos:].strip()

        # 安全检查: 如果切不动，回退到硬切
        if not left or not right:
            return self._hard_split(text, target)

        return self._recursive_binary_split(
            left, target
        ) + self._recursive_binary_split(right, target)

    def _find_middle_boundary(self, text: str, mid: int) -> int:
        """
        在 mid 附近找最近的段落/句子/换行边界。

        搜索范围: 中间 ± 25%，但保证两边各 ≥ 25% 文本长度。
        优先级: 段落 > 句子 > 换行 > mid 本身
        """
        total = len(text)
        margin = total // 4
        start = max(margin, mid - total // 4)
        end = min(total - margin, mid + total // 4)

        if start >= end:
            start, end = max(0, mid - 500), min(total, mid + 500)

        region = text[start:end]

        # 按优先级搜索: 段落边界 → 句子边界 → 换行符
        for pattern in [r"\n\s*\n", r"[.!?。！？]\s", r"\n"]:
            matches = [start + m.end() for m in re.finditer(pattern, region)]
            if matches:
                return min(matches, key=lambda p: abs(p - mid))

        return mid

    def _hard_split(self, text: str, target: int) -> List[str]:
        """最后的手段: 按 token 数硬切。保证绝不超标。"""
        tokens = self.tokenizer.encode(text)
        return [
            self.tokenizer.decode(tokens[i : i + target])
            for i in range(0, len(tokens), target)
        ]
