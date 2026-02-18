"""
CascadeMatcher — 级联锚点匹配器

在原文中定位 LLM 返回的 end_quote 文本片段。
使用多级回退策略，从精确匹配逐步降级到模糊匹配。

匹配策略 (按优先级):
1. 精确子串匹配
2. 去空格正则匹配 (忽略空白符差异)
3. 前缀模糊匹配 (前 50% 内容)
4. 关键词密度匹配 (基于稀疏关键词定位)
"""

import re
import logging


logger = logging.getLogger("ai_chunking.utils.matcher")


class CascadeMatcher:
    """
    级联锚点匹配器。
    
    用于在原文中找到 LLM 返回的引用文本的位置。
    LLM 可能返回略有偏差的引用（多余空格、截断等），
    所以需要多级回退策略。
    """

    def find_anchor(self, text: str, quote: str) -> int:
        """
        在 text 中查找 quote 的位置。
        
        返回:
            匹配位置的字符索引 (quote 结尾处)，找不到返回 -1
        """
        if not quote or not text:
            return -1

        quote = quote.strip()
        if len(quote) < 3:
            return -1

        # Level 1: 精确匹配
        idx = text.find(quote)
        if idx != -1:
            return idx + len(quote)

        # Level 2: 正则模糊匹配 (忽略所有空白字符差异)
        # 将 quote 中的同时也 escape 特殊字符
        try:
            # 1. Escape regex special chars
            escaped_quote = re.escape(quote)
            # 2. Replace escaped spaces '\ ' with '\s+' to match any whitespace
            # Note: re.escape escapes spaces as '\ ' in Python < 3.7, or ' ' in >= 3.7
            # simpler approach: split by space and join with \s+
            parts = quote.split()
            if parts:
                pattern_str = r'\s*'.join(re.escape(p) for p in parts)
                # limit pattern length to avoid regex DOS on huge quotes
                if len(pattern_str) < 500:
                    match = re.search(pattern_str, text, re.IGNORECASE)
                    if match:
                        return match.end()
        except Exception:
            pass # Fallback if regex fails

        # Level 3: 宽松的前缀匹配 (取前 60% 字符)
        n_chars = max(len(quote) * 60 // 100, 10)
        prefix = quote[:n_chars]
        
        # 尝试对前缀进行正则匹配
        try:
            parts = prefix.split()
            if parts:
                pattern_str = r'\s*'.join(re.escape(p) for p in parts)
                match = re.search(pattern_str, text, re.IGNORECASE)
                if match:
                    # 找到前缀后，预估结束位置 (假设长度差异不大)
                    return min(match.start() + len(quote) + 20, len(text))
        except Exception:
            pass

        # Level 4: 关键词密度匹配 (Full Text Search)
        # 提取长词(>4 chars)，在全文寻找包含最多长词的窗口
        keywords = [w for w in re.split(r'\s+', quote) if len(w) > 4]
        if len(keywords) >= 2:
            best_end_pos = -1
            max_score = 0
            
            # 使用滑动窗口，窗口大小略大于 quote
            window_size = int(len(quote) * 1.5)
            step = max(window_size // 4, 10)
            
            for i in range(0, len(text), step):
                chunk = text[i : i + window_size]
                chunk_lower = chunk.lower()
                
                score = 0
                for kw in keywords:
                    if kw.lower() in chunk_lower:
                        score += 1
                
                # 如果匹配了大部分关键词
                if score > max_score:
                    max_score = score
                    # 粗略定位到窗口结束作为引用结束
                    # 实际上应该在窗口内再精确定位，这里简化处理
                    best_end_pos = i + min(len(quote), len(chunk))
            
            token_ratio = max_score / len(keywords)
            if token_ratio > 0.6: # 至少匹配 60% 的关键词
                return min(best_end_pos + 20, len(text))

        return -1
