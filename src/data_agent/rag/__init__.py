"""RAG：文档检索。每一环都可以换，配置分成两类：

    建索引时（改了要重建，按指纹各存一份）：IndexSpec
        解析     parsers/    PDF → Document：一串有类型的元素（标题、段落、表格、页眉页脚），带页码和章节路径
        分片     chunk.py    fixed（朴素滑窗）/ page（一页一片）/ structure（按结构合并，表格带表头，片前加章节前缀）
        嵌入     dense.py    本地 sentence-transformers 模型（bge-m3 …）；可以不要，只用 BM25
    查询时（改了立刻生效）：SearchSpec
        召回     bm25.py / dense.py，几路用 RRF 融合
        过滤     只在指定的文档里找（比如按公司、年份先定到一份 10-K）
        重排     rerank.py   交叉编码器（bge-reranker-v2-m3 …），可以不要

索引存在本地目录，按文档分开存（片 .jsonl + 向量 .npy），几十万片暴力算点积就够快。
增量更新：Index.sync 只处理新增、改过、删掉的文档；解析按文件内容缓存，嵌入按片文本缓存（embed_cache.py）。
知识库（collection.py）：一个文档目录一个索引，项目用 DOCS_DIRS 挂载，Agent 的 list_docs / search_docs / read_doc 用它。
"""

from .chunk import Chunk, ChunkSpec
from .collection import Collection
from .document import Document, Element
from .index import Hit, Index, IndexSpec, SearchSpec
from .parsers import ParsedCache

__all__ = ["Chunk", "ChunkSpec", "Collection", "Document", "Element", "Hit", "Index", "IndexSpec", "ParsedCache", "SearchSpec"]
