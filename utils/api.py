import time
import asyncio
import logging
from typing import Optional
import tiktoken
from openai import AsyncOpenAI

logger = logging.getLogger("ai_chunking.utils.api")
_enc = tiktoken.get_encoding("cl100k_base")

# 全局并发限制
_semaphore = asyncio.Semaphore(10)

def set_concurrency_limit(limit: int):
    """设置全局 LLM 并发限制。"""
    global _semaphore
    _semaphore = asyncio.Semaphore(limit)
    logger.debug(f"LLM Concurrency limit set to {limit}")

def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))

async def llm_call(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    _max_empty_retries: int = 5
) -> str:
    """
    流式 LLM 调用，空响应自动重试。
    受全局 _semaphore 并发控制。

    NIM API 有 ~20% 概率返回空 SSE 流 (delta 全为空)。
    重试策略: 检测到空响应 → 等 2s → 重试，最多 5 次。
    """
    async with _semaphore:
        for attempt in range(1, _max_empty_retries + 1):
            start_t = time.time()
            parts = []
            try:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=32768,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        parts.append(delta.content)
                
                content = "".join(parts)
                elapsed = time.time() - start_t

                if content.strip():
                    logger.debug(f"✅ LLM: {_count_tokens(content)} tokens, {elapsed:.1f}s")
                    return content
                
                # 空响应 → 重试
                if attempt < _max_empty_retries:
                    logger.warning(f"⚠️ LLM 空响应 ({elapsed:.1f}s), 2s 后重试 ({attempt}/{_max_empty_retries})")
                    await asyncio.sleep(2)
                else:
                    msg = (
                        f"❌ 模型 {model} 连续 {_max_empty_retries} 次返回空响应，"
                        f"上游模型质量较差，请更换模型"
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)
                    
            except Exception as e:
                # 只有网络/API异常才重试，逻辑错误直接抛出? 这里保持原样简单重试
                if attempt < _max_empty_retries:
                    logger.warning(f"LLM调用异常: {e}, 重试中({attempt}/{_max_empty_retries})...")
                    await asyncio.sleep(2)
                    continue
                raise e
    return ""  # Should not reach here if successful

