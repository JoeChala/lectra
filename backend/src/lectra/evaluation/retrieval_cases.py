from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalCase:
    """Defines one retrieval evaluation case."""

    query: str
    relevant_chunk_ids: tuple[str, ...]
    topic: str
    query_type: str


RETRIEVAL_CASES = [
    RetrievalCase(
        query="What is a process?",
        relevant_chunk_ids=("os-processes",),
        topic="processes",
        query_type="direct",
    ),
    RetrievalCase(
        query="What is the smallest unit of CPU execution inside a process?",
        relevant_chunk_ids=("os-threads",),
        topic="threads",
        query_type="conceptual",
    ),
    RetrievalCase(
        query="How does the operating system decide which process gets CPU time?",
        relevant_chunk_ids=("os-scheduling",),
        topic="cpu_scheduling",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query=(
            "What happens when the operating system switches execution "
            "from one process to another?"
        ),
        relevant_chunk_ids=("os-context-switching",),
        topic="context_switching",
        query_type="conceptual",
    ),
    RetrievalCase(
        query=(
            "Why can multiple processes remain permanently blocked while "
            "waiting for resources held by each other?"
        ),
        relevant_chunk_ids=("os-deadlocks",),
        topic="deadlocks",
        query_type="conceptual",
    ),
    RetrievalCase(
        query="How are virtual addresses divided and mapped to physical memory?",
        relevant_chunk_ids=(
            "os-paging",
            "os-virtual-memory",
        ),
        topic="paging",
        query_type="conceptual",
    ),
    RetrievalCase(
        query="How can a program use an address space larger than the available RAM?",
        relevant_chunk_ids=("os-virtual-memory",),
        topic="virtual_memory",
        query_type="conceptual",
    ),
    RetrievalCase(
        query=(
            "Which memory page should be removed when there is no free frame "
            "for a newly needed page?"
        ),
        relevant_chunk_ids=("os-page-replacement",),
        topic="page_replacement",
        query_type="confusable",
    ),
    RetrievalCase(
        query=(
            "How does the operating system organize files and directories "
            "on a storage device?"
        ),
        relevant_chunk_ids=("os-file-system",),
        topic="file_systems",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query=(
            "How can concurrent threads safely access shared resources "
            "without causing race conditions?"
        ),
        relevant_chunk_ids=("os-synchronization",),
        topic="synchronization",
        query_type="conceptual",
    ),
]

HARD_RETRIEVAL_CASES = [
    RetrievalCase(
        query=(
            "What operating-system mechanism allows an application to "
            "operate as if it has more memory than the machine physically has?"
        ),
        relevant_chunk_ids=("os-virtual-memory",),
        topic="virtual_memory",
        query_type="terminology_mismatch",
    ),
    RetrievalCase(
        query=(
            "When one running program stops and another begins using the "
            "processor, what state transition does the OS perform?"
        ),
        relevant_chunk_ids=("os-context-switching",),
        topic="context_switching",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query=(
            "If main memory has no available space for data that must be "
            "loaded, how does the operating system choose what to discard?"
        ),
        relevant_chunk_ids=("os-page-replacement",),
        topic="page_replacement",
        query_type="terminology_mismatch",
    ),
    RetrievalCase(
        query=(
            "What abstraction separates the addresses used by a program "
            "from the actual locations available in RAM?"
        ),
        relevant_chunk_ids=("os-virtual-memory", "os-paging"),
        topic="virtual_memory",
        query_type="conceptual",
    ),
    RetrievalCase(
        query=(
            "How does the OS coordinate simultaneous execution when "
            "different execution units need access to the same resource?"
        ),
        relevant_chunk_ids=("os-synchronization",),
        topic="synchronization",
        query_type="conceptual",
    ),
    RetrievalCase(
        query=(
            "What situation occurs when several executing programs cannot "
            "make progress because each is waiting for something controlled "
            "by another?"
        ),
        relevant_chunk_ids=("os-deadlocks",),
        topic="deadlocks",
        query_type="conceptual",
    ),
    RetrievalCase(
        query=(
            "How does the operating system decide which waiting task gets "
            "the processor next?"
        ),
        relevant_chunk_ids=("os-scheduling",),
        topic="cpu_scheduling",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query=(
            "What is the execution unit that can run independently while "
            "remaining within the resources of a larger program?"
        ),
        relevant_chunk_ids=("os-threads",),
        topic="threads",
        query_type="conceptual",
    ),
    RetrievalCase(
        query=(
            "What component provides the structure through which stored "
            "data is represented as files and folders?"
        ),
        relevant_chunk_ids=("os-file-system",),
        topic="file_systems",
        query_type="terminology_mismatch",
    ),
    RetrievalCase(
        query=(
            "What happens internally when the processor stops executing "
            "one process so that another process can continue?"
        ),
        relevant_chunk_ids=("os-context-switching",),
        topic="context_switching",
        query_type="paraphrase",
    ),
]
