"""
Orchestrator — 切分编排器

负责将整篇文档拆解为合理大小的块:
1. 结构化预扫描: 按 Markdown 标题拆分成 structural_nodes
2. 编排决策: 小节合并、大节切分、中等节直接保留
3. 委托执行: 大节交给 CoordinateSplitter._process_large_block()

不做的事:
- 不直接调用 LLM (LLM 调用在 tools.py 或 CoordinateSplitter)
- 不管理记忆 (记忆在 ProtocolMemory)
"""

import re
import logging
import asyncio
from typing import List, Dict, AsyncGenerator, TYPE_CHECKING

if TYPE_CHECKING:
    from core.coordinate_splitter import CoordinateSplitter, SplitterInput

logger = logging.getLogger("ai_chunking.core.orchestrator")


# ============================================================
# 结构化预扫描
# ============================================================

def _structural_prescan(splitter: "CoordinateSplitter", text: str) -> List[Dict]:
    """
    按 Markdown 标题 (# / ## / ###) 将文本拆分成结构化节点。
    
    返回:
        List[Dict]: 每项包含 {"heading": str, "level": int, "content": str, "tokens": int}
    """
    # 按标题分割
    heading_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    
    nodes = []
    last_end = 0
    
    for match in heading_pattern.finditer(text):
        # 标题之前的内容
        if match.start() > last_end:
            before = text[last_end:match.start()].strip()
            if before:
                tokens = splitter._count_tokens(before)
                if tokens > 50:  # 忽略太短的片段
                    nodes.append({
                        "heading": "(前文)",
                        "level": 0,
                        "content": before,
                        "tokens": tokens,
                    })
        
        last_end = match.start()
        level = len(match.group(1))
        heading = match.group(2).strip()
    
    # 最后一段（从最后一个标题到文末）
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            # 如果有标题，把标题和内容一起
            tokens = splitter._count_tokens(remaining)
            if tokens > 50:
                nodes.append({
                    "heading": heading if nodes or last_end > 0 else "(全文)",
                    "level": level if last_end > 0 else 0,
                    "content": remaining,
                    "tokens": tokens,
                })
    
    # 如果没找到任何标题，把整个文本作为一个节点
    if not nodes:
        nodes.append({
            "heading": "(全文)",
            "level": 0,
            "content": text,
            "tokens": splitter._count_tokens(text),
        })
    
    return nodes


# ============================================================
# 编排核心
# ============================================================

async def split_text(
    splitter: "CoordinateSplitter",
    text: str,
    params: "SplitterInput",
) -> AsyncGenerator[Dict, None]:
    """
    编排切分流程。
    
    1. 结构化预扫描 → structural_nodes
    2. 遍历节点，决定 Keep / Merge / Split
    3. 流式 yield 每个切好的块
    """
    logger.debug("Starting coordinate-based splitting...")

    target_chunk_tokens = params.target_tokens
    max_llm_context = params.max_llm_context

    # 动态配置
    marker_interval = max(target_chunk_tokens // 10, 100)
    batch_size_tokens = int(max_llm_context * 0.33)  # 33% 给文本，与 Scanner 保持一致

    logger.debug(f"Target chunk size: {target_chunk_tokens}, Marker interval: {marker_interval}")
    logger.debug(f"Batch size: {batch_size_tokens} (from max context {max_llm_context})")

    # Step 1: 结构化预扫描
    structural_nodes = _structural_prescan(splitter, text)
    logger.debug(f"Structural scan found {len(structural_nodes)} physical nodes.")

    # Step 2: 编排 — Merge vs. Split (Streaming)
    async for chunk in _plan_and_execute(
        splitter=splitter,
        structural_nodes=structural_nodes,
        params=params,
        target_chunk_tokens=target_chunk_tokens,
        marker_interval=marker_interval,
        batch_size_tokens=batch_size_tokens,
    ):
        yield chunk


async def _plan_and_execute(
    splitter: "CoordinateSplitter",
    structural_nodes: List[Dict],
    params: "SplitterInput",
    target_chunk_tokens: int,
    marker_interval: int,
    batch_size_tokens: int,
) -> AsyncGenerator[Dict, None]:
    """
    遍历结构化节点，执行编排决策:
    
    - 小节 (< target * 0.5): 缓冲合并
    - 中节 (0.5x ~ 1.5x target): 直接保留
    - 大节 (> 1.5x target): 交给坐标切分
    """
    
    # ── Phase 1.5: 并发生成章节详细摘要 ──
    # 这使得在切分时，LLM 已经拥有全文档的详细地图，而不仅仅是 Narrative
    logger.debug(f"Phase 1.5: Pre-computing summaries for {len(structural_nodes)} sections...")
    
    async def _summarize_section(node):
        # 只对中篇幅以上的章节做摘要 (太短的没必要)
        if len(node["content"]) < 500: 
            return
        
        try:
            # 复用 splitter 的 _quick_summary (并发调用)
            res = await splitter._quick_summary(node["content"])
            summ = res.get("summary", "")
            if summ:
                splitter.protocol_memory.add_section_summary(node["heading"], summ)
        except Exception as e:
            logger.warning(f"Summarize error {node['heading'][:10]}: {e}")

    # 并发执行摘要任务
    if structural_nodes:
        summary_tasks = [_summarize_section(n) for n in structural_nodes]
        await asyncio.gather(*summary_tasks)
    
    logger.debug("Section summaries pre-computed. Starting main splitting...")
    # ============================================================
    # 并发 worker 封装
    # ============================================================
    
    async def _process_large_node_task(
        node_content: str, 
        base_id_counter: int
    ) -> List[Dict]:
        """
        处理单个大节点的任务 wrapper。
        包含: 坐标切分 generator 消费 + 吸附/二分逻辑
        返回: 该节点产生的所有 Chunks
        """
        results = []
        pending_chunk = None
        local_counter = 0

        async for sub_chunk in splitter._process_large_block(
            node_content, params, marker_interval, batch_size_tokens
        ):
            local_counter += 1
            # 使用 base_id + local_id 避免冲突 (虽然只是显示用)
            # 实际上 orchestrator 还需要全局重排 id，这里先占个位
            sub_chunk["id"] = f"temp_{base_id_counter}_{local_counter}"

            is_remainder = sub_chunk.pop("_is_remainder", False)

            if is_remainder and pending_chunk:
                # ── 有剩余块 & 有前块: 尝试吸附 ──
                combined_tokens = pending_chunk["tokens"] + sub_chunk["tokens"]
                
                if combined_tokens <= target_chunk_tokens * 1.2:
                    # 吸附
                    pending_chunk["content"] += "\n\n" + sub_chunk["content"]
                    pending_chunk["tokens"] = combined_tokens
                else:
                    # 二分
                    combined = pending_chunk["content"] + "\n\n" + sub_chunk["content"]
                    mid = len(combined) // 2
                    split_idx = mid
                    match = re.search(r'\n\s*\n', combined[mid-500:mid+500])
                    if match:
                        split_idx = mid - 500 + match.start()
                    
                    part1 = combined[:split_idx].strip()
                    part2 = combined[split_idx:].strip()

                    # 并发生成摘要
                    m1_task = splitter._quick_summary(part1)
                    m2_task = splitter._quick_summary(part2)
                    m1, m2 = await asyncio.gather(m1_task, m2_task)

                    results.append({
                        "id": "bisect_a",
                        "title": m1.get("title", ""),
                        "summary": m1.get("summary", ""),
                        "content": part1,
                        "tokens": splitter._count_tokens(part1),
                    })
                    
                    pending_chunk = {
                        "id": "bisect_b",
                        "title": m2.get("title", ""),
                        "summary": m2.get("summary", ""),
                        "content": part2,
                        "tokens": splitter._count_tokens(part2),
                    }

            elif is_remainder and not pending_chunk:
                # 孤立剩余
                if not sub_chunk["title"] or sub_chunk["title"] == "待定":
                     meta = await splitter._quick_summary(sub_chunk["content"])
                     sub_chunk["title"] = meta.get("title", "续文")
                     sub_chunk["summary"] = meta.get("summary", "")
                pending_chunk = sub_chunk

            else:
                if pending_chunk:
                    results.append(pending_chunk)
                pending_chunk = sub_chunk

        if pending_chunk:
            results.append(pending_chunk)
        
        return results

    async def _process_medium_node_task(node_content: str, node_tokens: int) -> List[Dict]:
        """处理中节点的任务 wrapper"""
        summary_res = await splitter._quick_summary(node_content)
        return [{
            "id": "temp_medium",
            "title": summary_res["title"],
            "summary": summary_res["summary"],
            "content": node_content,
            "tokens": node_tokens,
        }]

    # ============================================================
    # 主流程: 并发调度
    # ============================================================



    # 1. 生成所有 Tasks
    tasks = []  # List[Coroutine]
    
    buffer_nodes: List[Dict] = []
    buffer_tokens: int = 0
    
    # 辅助: 创建 flush buffer 的 task
    def create_flush_task(nodes_snapshot: List[Dict]):
        async def _flush() -> List[Dict]:
            if not nodes_snapshot: return []
            content = "\n\n".join(n["content"] for n in nodes_snapshot)
            tokens = sum(n["tokens"] for n in nodes_snapshot)
            
            if tokens > target_chunk_tokens * 1.5:
                # buffer 意外过大 -> 降级为 large node 处理
                return await _process_large_node_task(content, 999)
            else:
                # buffer 正常 -> 当作 medium node 处理
                return await _process_medium_node_task(content, tokens)
        return _flush()

    # 遍历节点规划任务
    task_id_counter = 0
    for node in structural_nodes:
        node_tokens = node["tokens"]

        if node_tokens > target_chunk_tokens * 1.5:
            # flush buffer first
            if buffer_nodes:
                tasks.append(create_flush_task(buffer_nodes[:]))
                buffer_nodes = []
                buffer_tokens = 0
            
            # large node task
            task_id_counter += 1
            tasks.append(_process_large_node_task(node["content"], task_id_counter))
            
        elif node_tokens < target_chunk_tokens * 0.5:
            # append to buffer
            buffer_nodes.append(node)
            buffer_tokens += node_tokens
            if buffer_tokens >= target_chunk_tokens * 0.8:
                tasks.append(create_flush_task(buffer_nodes[:]))
                buffer_nodes = []
                buffer_tokens = 0
        
        else:
            # flush buffer first
            if buffer_nodes:
                tasks.append(create_flush_task(buffer_nodes[:]))
                buffer_nodes = []
                buffer_tokens = 0
            
            # medium node task
            tasks.append(_process_medium_node_task(node["content"], node["tokens"]))

    # flush remaining buffer
    if buffer_nodes:
        tasks.append(create_flush_task(buffer_nodes[:]))

    # 2. 并发启动所有 Tasks
    logger.debug(f"Launching {len(tasks)} concurrent splitting tasks...")
    running_tasks = [asyncio.create_task(t) for t in tasks]

    # 3. 按顺序等待并 yield 结果
    final_chunk_counter = 0
    
    for i, task in enumerate(running_tasks):
        try:
            chunk_list = await task
            for chunk in chunk_list:
                final_chunk_counter += 1
                # 重新分配正式 ID
                chunk["id"] = f"ck_{final_chunk_counter:03d}"
                
                # 记录到记忆 (虽然有些为了并发可能没用上, 但为了后续的 RAG 还是需要的)
                if splitter.protocol_memory and chunk.get("summary"):
                    splitter.protocol_memory.add_chunk_summary(chunk["summary"])
                
                yield chunk
                
        except Exception as e:
            logger.error(f"Task {i} failed: {e}", exc_info=True)
            # Error handling: yield an error placeholder or skip?
            # Yielding raw content fallback
            yield {
                "id": f"err_{i}",
                "title": "Error Chunk",
                "summary": "Processing failed",
                "content": "Content processing error",
                "tokens": 0
            }

    logger.debug(f"Orchestrator complete. Total chunks: {final_chunk_counter}")
