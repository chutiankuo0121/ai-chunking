# ai-chunking package init

import logging
from pathlib import Path
from typing import Callable, Awaitable, AsyncGenerator, Dict, Union

from core.coordinate_splitter import CoordinateSplitter, SplitterInput, ChunkMetadata
from core.memory import ProtocolMemory
from core.scanner import ScannerEngine
from core.tools import ChunkResult, coordinate_split, quick_summary, extract_narrative
from utils.matcher import CascadeMatcher
from utils.logger import setup_logger

# 向后兼容
ContextAwareSplitter = CoordinateSplitter

__all__ = [
    "CoordinateSplitter",
    "ContextAwareSplitter",
    "SplitterInput",
    "ChunkMetadata",
    "ChunkResult",
    "ProtocolMemory",
    "ScannerEngine",
    "CascadeMatcher",
    "coordinate_split",
    "quick_summary",
    "extract_narrative",
    "smart_chunk_file",  # New API
    "get_openai_wrapper" # New API Helper
]

async def smart_chunk_file(
    file_path: Union[str, Path],
    llm_func: Callable[[str], Awaitable[str]],
    target_tokens: int,
    max_llm_context: int,
) -> AsyncGenerator[Dict, None]:
    """
    一站式智能切分 API (Unified Smart Chunking API).
    
    自动执行两阶段处理:
    1. Phase 1 (Scanner): 全文扫描，生成大纲与摘要 (Global Scan & Narrative Extraction)
    2. Phase 2 (Splitter): 基于上下文的智能切分 (Context-Aware Splitting)
    
    Args:
        file_path: 目标文件路径 (Markdown/Text)
        llm_func: 异步 LLM 调用函数 (prompt -> str response)
        target_tokens: 每个块的目标 token 数 (默认 3500)
        max_llm_context: 模型最大上下文窗口 (默认 128000)
        
    Yields:
        Dict: 包含 chunk 信息的字典 (id, title, summary, content, tokens)
    """
    logger = logging.getLogger("ai_chunking.api") # Standard logger
    
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    logger.info(f"🚀 Processing: {path_obj.name}")
    
    with open(path_obj, "r", encoding="utf-8") as f:
        text = f.read()

    # Phase 1: Scanner
    # Logs are handled internally ("AI Summarizing...")
    scanner = ScannerEngine(llm_func=llm_func, max_llm_context=max_llm_context)
    try:
        memory = await scanner.scan(text, doc_name=path_obj.name)
    except Exception as e:
        logger.error(f"Phase 1 (Scanning) failed: {e}")
        raise e

    # Phase 2: Splitting
    # Logs are handled internally ("Intelligent Chunking...")
    splitter = CoordinateSplitter(llm_func=llm_func, protocol_memory=memory)
    params = SplitterInput(
        file_path=str(path_obj.absolute()),
        target_tokens=target_tokens,
        max_llm_context=max_llm_context,
    )
    
    try:
        async for chunk in splitter.split_text(text, params):
            # Standardize output format
            yield {
                "chunk_id": chunk.get("id"),
                "title": chunk.get("title", "Untitled"),
                "summary": chunk.get("summary", ""),
                "content": chunk.get("content", ""),
                "tokens": chunk.get("tokens", 0),
            }
    except Exception as e:
        logger.error(f"Phase 2 (Splitting) failed: {e}")
        raise e
        
    logger.debug("smart_chunk_file iterator complete.")

def get_openai_wrapper(
    api_key: str,
    base_url: str = None,
    model: str,        # No default
    concurrency: int,  # No default
) -> Callable[[str], Awaitable[str]]:
    """
    创建一个预配置的 OpenAI LLM 调用函数。
    包含: 并发控制、自动重试、空响应处理。
    
    Args:
        api_key: OpenAI API Key
        base_url: (Optional) Custom Base URL
        model: Model name
        concurrency: Max concurrent requests
        
    Returns:
        符合 llm_func 签名的异步函数
    """
    from openai import AsyncOpenAI
    from utils.api import llm_call, set_concurrency_limit

    # 配置全局并发
    set_concurrency_limit(concurrency)
    
    # 初始化客户端
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    async def wrapper(prompt: str, temperature: float = 0.7) -> str:
        return await llm_call(client, model, prompt, temperature)
        
    return wrapper
