
import asyncio
import json
import logging
import sys
from pathlib import Path

# 1. Force using local source code
# Add current directory to sys.path so we import local 'ai_chunking' package
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import from local source
try:
    from ai_chunking import smart_chunk_file, get_openai_wrapper
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running this script from the 'ai-chunking' directory.")
    sys.exit(1)

# Configuration
OPENAI_BASE_URL = "https://ai.123719141.xyz/v1"
OPENAI_API_KEY = "CTK-NUXnyF7bV9l99sPQnrCErwbrXlpTLH44"
LLM_MODEL = "deepseek-ai/deepseek-v3.2"

# Setup Verbose Logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG level!
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
# Silence httpx/openai noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("debug_run")

async def main():
    # Targeted file (using relative path from THIS script)
    # Script is in: .../ai-chunking/
    # Target is in: .../test_chunking/
    script_dir = Path(__file__).resolve().parent
    test_file_path = (script_dir.parent / "test_chunking/2409.11569v1.md").resolve()
    
    if not test_file_path.exists():
        logger.error(f"❌ Test file not found at: {test_file_path}")
        # Try local README if external file not found (relative to script)
        test_file_path = (script_dir / "README.md").resolve()
        logger.info(f"Fallback to local file: {test_file_path}")

    # Reduce concurrency for debugging
    concurrency = 5  # Lower concurrency to prevent rate limits during debug
    target_tokens = 3500
    max_llm_context = 256000

    logger.info("Initializing LLM client (Local Source)...")
    my_llm = get_openai_wrapper(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        concurrency=concurrency,
        base_url=OPENAI_BASE_URL
    )

    logger.info(f"=== Debugging Local Source: {test_file_path.name} ===")
    
    output_chunks = []
    
    try:
        async for chunk in smart_chunk_file(
            file_path=test_file_path,
            llm_func=my_llm,
            target_tokens=target_tokens,
            max_llm_context=max_llm_context
        ):
            print(f"  [Chunk Yielded] {chunk['title']} ({chunk['tokens']} tokens)")
            output_chunks.append(chunk)

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        return

    logger.info(f"✅ Run Complete! Generated {len(output_chunks)} chunks.")

if __name__ == "__main__":
    asyncio.run(main())
