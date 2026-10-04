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
EXPANDED_RETRIEVAL_CASES = [
    # ------------------------------------------------------------------
    # Terminology mismatch
    # ------------------------------------------------------------------
    RetrievalCase(
        query="What abstraction gives a process its own logical memory space?",
        relevant_chunk_ids=("os-virtual-memory",),
        topic="virtual_memory",
        query_type="terminology_mismatch",
    ),
    RetrievalCase(
        query="What method decides which resident memory block gets evicted?",
        relevant_chunk_ids=("os-page-replacement",),
        topic="page_replacement",
        query_type="terminology_mismatch",
    ),
    RetrievalCase(
        query="What mechanism prevents competing execution \
            units from corrupting shared state?",
        relevant_chunk_ids=("os-synchronization",),
        topic="synchronization",
        query_type="terminology_mismatch",
    ),
    RetrievalCase(
        query="What OS facility arranges persistent data into \
            named containers and directories?",
        relevant_chunk_ids=("os-file-system",),
        topic="file_systems",
        query_type="terminology_mismatch",
    ),
    RetrievalCase(
        query="What operation preserves one execution context \
            before another takes the processor?",
        relevant_chunk_ids=("os-context-switching",),
        topic="context_switching",
        query_type="terminology_mismatch",
    ),
    RetrievalCase(
        query="What mechanism determines which waiting task receives \
          processor service?",
        relevant_chunk_ids=("os-scheduling",),
        topic="cpu_scheduling",
        query_type="terminology_mismatch",
    ),
    RetrievalCase(
        query="What execution entity shares a program's address space \
            while executing independently?",
        relevant_chunk_ids=("os-threads",),
        topic="threads",
        query_type="terminology_mismatch",
    ),
    RetrievalCase(
        query="What condition prevents a group of programs from\
              progressing because of circular resource dependence?",
        relevant_chunk_ids=("os-deadlocks",),
        topic="deadlocks",
        query_type="terminology_mismatch",
    ),
    RetrievalCase(
        query="What memory-management scheme maps fixed-size logical\
              units onto physical storage units?",
        relevant_chunk_ids=("os-paging",),
        topic="paging",
        query_type="terminology_mismatch",
    ),
    RetrievalCase(
        query="What abstraction lets software address more memory than\
              is physically installed?",
        relevant_chunk_ids=("os-virtual-memory",),
        topic="virtual_memory",
        query_type="terminology_mismatch",
    ),
    # ------------------------------------------------------------------
    # Conceptual
    # ------------------------------------------------------------------
    RetrievalCase(
        query="Why can a program continue using an address space even \
            when RAM cannot hold all of it?",
        relevant_chunk_ids=("os-virtual-memory",),
        topic="virtual_memory",
        query_type="conceptual",
    ),
    RetrievalCase(
        query="Why must an operating system sometimes remove an existing \
            memory page before loading another?",
        relevant_chunk_ids=("os-page-replacement",),
        topic="page_replacement",
        query_type="conceptual",
    ),
    RetrievalCase(
        query="How does an operating system prevent concurrent execution \
            from producing inconsistent shared data?",
        relevant_chunk_ids=("os-synchronization",),
        topic="synchronization",
        query_type="conceptual",
    ),
    RetrievalCase(
        query="How can several programs become unable to proceed even \
            though each is still running?",
        relevant_chunk_ids=("os-deadlocks",),
        topic="deadlocks",
        query_type="conceptual",
    ),
    RetrievalCase(
        query="Why does switching from one process to another require \
            the operating system to save information?",
        relevant_chunk_ids=("os-context-switching",),
        topic="context_switching",
        query_type="conceptual",
    ),
    RetrievalCase(
        query="Why can several execution flows exist inside one program \
            without each having separate process resources?",
        relevant_chunk_ids=("os-threads",),
        topic="threads",
        query_type="conceptual",
    ),
    RetrievalCase(
        query="How does the operating system choose among tasks that \
            are ready to execute?",
        relevant_chunk_ids=("os-scheduling",),
        topic="cpu_scheduling",
        query_type="conceptual",
    ),
    RetrievalCase(
        query="How does the operating system translate a program's \
            logical memory view into physical memory?",
        relevant_chunk_ids=("os-paging", "os-virtual-memory"),
        topic="paging",
        query_type="conceptual",
    ),
    RetrievalCase(
        query="How does persistent storage remain organized so \
            applications can locate their data?",
        relevant_chunk_ids=("os-file-system",),
        topic="file_systems",
        query_type="conceptual",
    ),
    RetrievalCase(
        query="Why can threads operating concurrently interfere \
            with one another when accessing shared resources?",
        relevant_chunk_ids=("os-synchronization",),
        topic="synchronization",
        query_type="conceptual",
    ),
    # ------------------------------------------------------------------
    # Paraphrase
    # ------------------------------------------------------------------
    RetrievalCase(
        query="What does the operating system do when it changes \
            execution from one process to another?",
        relevant_chunk_ids=("os-context-switching",),
        topic="context_switching",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query="How is processor time assigned to ready processes?",
        relevant_chunk_ids=("os-scheduling",),
        topic="cpu_scheduling",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query="What is the smallest independently schedulable \
            execution path within a process?",
        relevant_chunk_ids=("os-threads",),
        topic="threads",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query="What happens when multiple processes each wait \
            for a resource held by another?",
        relevant_chunk_ids=("os-deadlocks",),
        topic="deadlocks",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query="How are files and folders managed on a storage device?",
        relevant_chunk_ids=("os-file-system",),
        topic="file_systems",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query="How are logical memory pages associated with physical memory frames?",
        relevant_chunk_ids=("os-paging",),
        topic="paging",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query="How can an application work with a memory space \
            larger than physical RAM?",
        relevant_chunk_ids=("os-virtual-memory",),
        topic="virtual_memory",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query="Which page is selected when memory is full and another page must enter?",
        relevant_chunk_ids=("os-page-replacement",),
        topic="page_replacement",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query="How are simultaneous accesses to shared resources coordinated?",
        relevant_chunk_ids=("os-synchronization",),
        topic="synchronization",
        query_type="paraphrase",
    ),
    RetrievalCase(
        query="What does a process represent while it is executing?",
        relevant_chunk_ids=("os-processes",),
        topic="processes",
        query_type="paraphrase",
    ),
    # ------------------------------------------------------------------
    # Exact terminology
    # ------------------------------------------------------------------
    RetrievalCase(
        query="What is virtual memory?",
        relevant_chunk_ids=("os-virtual-memory",),
        topic="virtual_memory",
        query_type="exact_terminology",
    ),
    RetrievalCase(
        query="What is page replacement?",
        relevant_chunk_ids=("os-page-replacement",),
        topic="page_replacement",
        query_type="exact_terminology",
    ),
    RetrievalCase(
        query="What is process synchronization?",
        relevant_chunk_ids=("os-synchronization",),
        topic="synchronization",
        query_type="exact_terminology",
    ),
    RetrievalCase(
        query="What is a deadlock?",
        relevant_chunk_ids=("os-deadlocks",),
        topic="deadlocks",
        query_type="exact_terminology",
    ),
    RetrievalCase(
        query="What is a context switch?",
        relevant_chunk_ids=("os-context-switching",),
        topic="context_switching",
        query_type="exact_terminology",
    ),
    RetrievalCase(
        query="What is CPU scheduling?",
        relevant_chunk_ids=("os-scheduling",),
        topic="cpu_scheduling",
        query_type="exact_terminology",
    ),
    RetrievalCase(
        query="What is a thread?",
        relevant_chunk_ids=("os-threads",),
        topic="threads",
        query_type="exact_terminology",
    ),
    RetrievalCase(
        query="What is paging?",
        relevant_chunk_ids=("os-paging",),
        topic="paging",
        query_type="exact_terminology",
    ),
    RetrievalCase(
        query="What is a file system?",
        relevant_chunk_ids=("os-file-system",),
        topic="file_systems",
        query_type="exact_terminology",
    ),
    RetrievalCase(
        query="What is a process?",
        relevant_chunk_ids=("os-processes",),
        topic="processes",
        query_type="exact_terminology",
    ),
    # ------------------------------------------------------------------
    # Confusable / multi-concept
    # ------------------------------------------------------------------
    RetrievalCase(
        query="What is the difference between paging and virtual memory?",
        relevant_chunk_ids=("os-paging", "os-virtual-memory"),
        topic="paging",
        query_type="confusable",
    ),
    RetrievalCase(
        query="How is page replacement different from virtual memory?",
        relevant_chunk_ids=("os-page-replacement", "os-virtual-memory"),
        topic="page_replacement",
        query_type="confusable",
    ),
    RetrievalCase(
        query="How are threads different from processes?",
        relevant_chunk_ids=("os-threads", "os-processes"),
        topic="threads",
        query_type="confusable",
    ),
    RetrievalCase(
        query="How does CPU scheduling differ from context switching?",
        relevant_chunk_ids=("os-scheduling", "os-context-switching"),
        topic="cpu_scheduling",
        query_type="confusable",
    ),
    RetrievalCase(
        query="How are synchronization and deadlock related?",
        relevant_chunk_ids=("os-synchronization", "os-deadlocks"),
        topic="synchronization",
        query_type="confusable",
    ),
    RetrievalCase(
        query="What is the relationship between a process and its threads?",
        relevant_chunk_ids=("os-processes", "os-threads"),
        topic="processes",
        query_type="confusable",
    ),
    RetrievalCase(
        query="How do paging and page replacement work together?",
        relevant_chunk_ids=("os-paging", "os-page-replacement"),
        topic="paging",
        query_type="confusable",
    ),
    RetrievalCase(
        query="How does synchronization help prevent problems \
            caused by concurrent threads?",
        relevant_chunk_ids=("os-synchronization", "os-threads"),
        topic="synchronization",
        query_type="confusable",
    ),
    RetrievalCase(
        query="How does virtual memory relate to physical memory frames?",
        relevant_chunk_ids=("os-virtual-memory", "os-paging"),
        topic="virtual_memory",
        query_type="confusable",
    ),
    RetrievalCase(
        query="How does a context switch relate to CPU scheduling?",
        relevant_chunk_ids=("os-context-switching", "os-scheduling"),
        topic="context_switching",
        query_type="confusable",
    ),
]


ALL_RETRIEVAL_CASES = HARD_RETRIEVAL_CASES + EXPANDED_RETRIEVAL_CASES
