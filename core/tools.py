"""
Tool Layer (工具封装层)

把每个 LLM 调用封装成可复用的 "Tool"。
每个 Tool 是一个独立的、可测试的函数，接收明确的输入，返回结构化输出。

设计原则:
1. Prompt 统一从 prompts/ 目录加载，不在代码里硬编码。
2. 每个 Tool 都有自己的输入输出类型。
3. JSON 提取逻辑集中在 extract_json() 一个地方。
4. 使用 _safe_fill() 而非 str.format() 避免大括号冲突。

包含的 Tool:
- extract_glossary:   术语提取 (Scanner Phase 1)
- extract_narrative:  叙事大纲提取 (Scanner Phase 1)
- coordinate_split:   坐标切分 (CoordinateSplitter Phase 2 核心)
- quick_summary:      快速摘要 (小块的标题+摘要生成)
"""

import json
import re
import logging
from typing import List, Dict, Callable, Awaitable, Union

from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_never, retry_if_exception_type, stop_after_attempt

from .memory import NarrativeItem, load_prompt

logger = logging.getLogger("ai_chunking.core.tools")


# ============================================================
# 公共工具函数
# ============================================================

def _safe_fill(template: str, **kwargs) -> str:
    """
    安全的模板填充。
    
    用 str.replace() 代替 str.format()，
    这样 global_context 里可能出现的 {、} 不会被误解析。
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def _fix_latex_escapes(s: str) -> str:
    """
    修复 JSON 中的 LaTeX 反斜杠。
    
    LLM 输出的 JSON 中常含 \\mathcal, \\overline 等 LaTeX 命令，
    但 JSON 标准只允许 \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX。
    其他 \\x 序列都是非法的，需要转成 \\\\x。
    """
    # 合法的 JSON 转义字符
    valid_escapes = set('"\\bfnrtu/')
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            next_char = s[i + 1]
            if next_char in valid_escapes:
                # 合法转义，保留原样
                result.append(s[i])
                result.append(next_char)
                i += 2
            elif next_char == '\\':
                # 已经是双反斜杠，保留
                result.append('\\\\')
                i += 2
            else:
                # 非法转义 (LaTeX 命令) → 加一个反斜杠
                result.append('\\\\')
                result.append(next_char)
                i += 2
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def _repair_truncated_json(s: str) -> str:
    """
    修复被截断的 JSON 数组。
    
    LLM 输出超过 max_tokens 时 JSON 会被截断:
      [{"a":1}, {"b":2}, {"c":3    ← 在这里断了
    
    修复策略: 找到最后一个完整的 }, 截掉后面的不完整部分，加 ]
    """
    stripped = s.strip()
    if not stripped.startswith('['):
        return s
    
    # 从后往前找最后一个 },
    # 然后截掉后面不完整的条目
    last_complete = stripped.rfind('},')
    if last_complete == -1:
        # 试试找 } 后面直接跟空白/换行
        last_complete = stripped.rfind('}')
    
    if last_complete > 0:
        repaired = stripped[:last_complete + 1] + ']'
        return repaired
    
    return s


def _safe_json_loads(s: str) -> Union[dict, list]:
    """
    鲁棒的 JSON 解析，依次尝试:
    1. 直接解析
    2. 修复 LaTeX 转义后解析
    3. 修复截断后解析
    4. 修复 LaTeX + 截断后解析
    """
    # 1. 直接
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    
    # 2. 修复 LaTeX
    try:
        return json.loads(_fix_latex_escapes(s))
    except json.JSONDecodeError:
        pass
    
    # 3. 修复截断
    repaired = _repair_truncated_json(s)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # 4. 修复 LaTeX + 截断
    return json.loads(_repair_truncated_json(_fix_latex_escapes(s)))


def extract_json(text: str) -> Union[dict, list]:
    """
    从 LLM 输出中鲁棒地提取 JSON（支持返回 dict 或 list）。

    尝试顺序：
    1. 直接 json.loads (含 LaTeX 转义修复)
    2. 提取 ```json ``` 代码块  (正则)
    3. 提取通用 ``` ``` 代码块  (正则)
    4. 手动剥离 markdown 围栏   (逐行)
    5. 正则找最外层 [] 或 {}
    """
    cleaned = text.strip()

    # 尝试 1: 直接解析
    try:
        return _safe_json_loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 尝试 2: ```json ... ```
    m = re.search(r'```json\s*([\s\S]*?)\s*```', cleaned)
    if m:
        try:
            return _safe_json_loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试 3: ``` ... ```
    m = re.search(r'```\s*([\s\S]*?)\s*```', cleaned)
    if m:
        try:
            return _safe_json_loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试 4: 手动剥离 markdown 围栏
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        stripped = '\n'.join(lines).strip()
        try:
            return _safe_json_loads(stripped)
        except json.JSONDecodeError:
            pass

    # 尝试 5: 找最外层 []
    for m in reversed(list(re.finditer(r'(\[[\s\S]*\])', cleaned))):
        try:
            return _safe_json_loads(m.group(1))
        except json.JSONDecodeError:
            continue

    # 找最外层 {}
    for m in reversed(list(re.finditer(r'(\{[\s\S]*\})', cleaned))):
        try:
            return _safe_json_loads(m.group(1))
        except json.JSONDecodeError:
            continue

    logger.error(f"无法从 LLM 输出中提取 JSON:\n{cleaned}")
    raise json.JSONDecodeError("无法从 LLM 输出中提取 JSON", text, 0)

# ============================================================
# 数据模型
# ============================================================

class ChunkResult(BaseModel):
    """Tool 4 (coordinate_split) 的返回类型 — LLM 返回的单个切分结果。"""
    chunk_id: int
    start_marker: str
    end_marker: str
    end_quote: str
    title: str
    summary: str



# ============================================================
# Tool 1: 叙事大纲提取器 (Narrative Extractor) — Scanner 使用
# ============================================================

async def extract_narrative(
    text: str,
    llm_func: Callable[[str], Awaitable[str]],
    max_retries: int = 2,
) -> List[NarrativeItem]:
    """Tool: 从文本中提取叙事大纲。JSON 解析失败时自动重试。"""
    template = load_prompt("narrative_extractor")
    prompt = _safe_fill(template, text=text)
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = await llm_func(prompt)
            data = extract_json(response)
            
            entries = data.get("narrative", []) if isinstance(data, dict) else data
            
            items = []
            for entry in entries:
                if isinstance(entry, dict):
                    items.append(NarrativeItem(
                        section=entry.get("section", ""),
                        goal=entry.get("goal", ""),
                    ))
            
            logger.debug(f"Narrative Tool: extracted {len(items)} sections")
            return items
            
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(f"Narrative extraction 重试 ({attempt+1}/{max_retries}): {e}")
                continue
            logger.error(f"Narrative extraction failed after {max_retries+1} attempts: {e}")
            return []



# ============================================================
# Tool 2: 快速摘要生成器 (Quick Summary) — 核心分块摘要
# ============================================================

async def quick_summary(
    text: str,
    llm_func: Callable[[str], Awaitable[str]],
) -> Dict:
    """
    Tool: 为结构化直接保留的小 Chunk 生成标题和摘要。
    
    使用 structural_summary prompt，返回 JSON: {"title": ..., "summary": ...}
    """
    template = load_prompt("structural_summary")
    prompt = _safe_fill(template, text=text)

    try:
        response = await llm_func(prompt, temperature=0.7)
        data = extract_json(response)
        return {
            "title": data.get("title", "未命名章节"),
            "summary": data.get("summary", "无法解析摘要。"),
        }
    except Exception as e:
        logger.warning(f"Quick Summary Tool 失败: {e}")
        return {"title": "章节", "summary": "自动结构化分块（摘要生成失败）。"}



# ============================================================
# Tool 3: 坐标切分器 (Coordinate Split) — CoordinateSplitter 核心
# ============================================================

def _log_retry(retry_state):
    if retry_state.outcome.failed:
        ex = retry_state.outcome.exception()
        logger.warning(f"coordinate_split 重试: {type(ex).__name__}: {ex}")


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    before_sleep=_log_retry,
)
async def coordinate_split(
    marked_text: str,
    global_context: str,
    marker_interval: int,
    target_chunk_tokens: int,
    markers_per_chunk: int,
    llm_func: Callable[[str], Awaitable[str]],
) -> List[ChunkResult]:
    """
    Tool: 坐标切分 — 本库的核心切分能力。

    输入:
        marked_text:         已插入 [ID: N] 坐标标记的文本
        global_context:      全局上下文 (术语表 + 前文摘要)
        marker_interval:     标记间隔 (tokens)
        target_chunk_tokens: 目标块大小 (tokens)
        markers_per_chunk:   每个块预期包含的标记数
        llm_func:            LLM 调用函数

    输出:
        List[ChunkResult]: LLM 决定的切分方案。
    """
    template = load_prompt("coordinate_splitter")
    prompt = _safe_fill(
        template,
        global_context=global_context,
        marker_interval=str(marker_interval),
        target_chunk_tokens=str(target_chunk_tokens),
        markers_per_chunk=str(markers_per_chunk),
        marked_batch_text=marked_text,
    )

    response_text = await llm_func(prompt, temperature=0.7)

    data = extract_json(response_text)
    results = []
    for item in data:
        results.append(ChunkResult(
            chunk_id=item.get("chunk_id", 0),
            start_marker=item.get("start_marker", ""),
            end_marker=item.get("end_marker", ""),
            end_quote=item.get("end_quote", ""),
            title=item.get("title", ""),
            summary=item.get("summary", ""),
        ))

    logger.debug(f"Coordinate Split Tool: LLM 返回 {len(results)} 个切分方案")
    return results