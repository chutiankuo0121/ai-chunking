import logging
import sys

def setup_logger(level=logging.INFO):
    """
    配置精简的项目日志。
    只允许 ai_chunking 命名空间的 INFO+ 日志通过，
    并屏蔽其他第三方库的 INFO 日志。
    """
    # 获取根 Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除旧 Handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 创建控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    # 精简格式: 时间 - 消息
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 屏蔽第三方库的 INFO 日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # 确保项目内的日志能显示
    project_logger = logging.getLogger("ai_chunking")
    project_logger.setLevel(level)
    
    return project_logger
