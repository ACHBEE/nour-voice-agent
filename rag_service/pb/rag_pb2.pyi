from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SearchMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SEARCH_MODE_HYBRID: _ClassVar[SearchMode]
    SEARCH_MODE_VECTOR: _ClassVar[SearchMode]
    SEARCH_MODE_KEYWORD: _ClassVar[SearchMode]
SEARCH_MODE_HYBRID: SearchMode
SEARCH_MODE_VECTOR: SearchMode
SEARCH_MODE_KEYWORD: SearchMode

class SearchRequest(_message.Message):
    __slots__ = ("query", "top_k", "language", "mode")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    query: str
    top_k: int
    language: str
    mode: SearchMode
    def __init__(self, query: _Optional[str] = ..., top_k: _Optional[int] = ..., language: _Optional[str] = ..., mode: _Optional[_Union[SearchMode, str]] = ...) -> None: ...

class Chunk(_message.Message):
    __slots__ = ("chunk_id", "doc_id", "title", "section", "content", "lang", "source_path", "score", "vector_rank", "keyword_rank")
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    SECTION_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    LANG_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PATH_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    VECTOR_RANK_FIELD_NUMBER: _ClassVar[int]
    KEYWORD_RANK_FIELD_NUMBER: _ClassVar[int]
    chunk_id: int
    doc_id: str
    title: str
    section: str
    content: str
    lang: str
    source_path: str
    score: float
    vector_rank: int
    keyword_rank: int
    def __init__(self, chunk_id: _Optional[int] = ..., doc_id: _Optional[str] = ..., title: _Optional[str] = ..., section: _Optional[str] = ..., content: _Optional[str] = ..., lang: _Optional[str] = ..., source_path: _Optional[str] = ..., score: _Optional[float] = ..., vector_rank: _Optional[int] = ..., keyword_rank: _Optional[int] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("chunks", "embed_ms", "search_ms")
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    EMBED_MS_FIELD_NUMBER: _ClassVar[int]
    SEARCH_MS_FIELD_NUMBER: _ClassVar[int]
    chunks: _containers.RepeatedCompositeFieldContainer[Chunk]
    embed_ms: float
    search_ms: float
    def __init__(self, chunks: _Optional[_Iterable[_Union[Chunk, _Mapping]]] = ..., embed_ms: _Optional[float] = ..., search_ms: _Optional[float] = ...) -> None: ...
