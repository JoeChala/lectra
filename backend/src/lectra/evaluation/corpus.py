from lectra.domain.retrieval import Chunk

BENCHMARK_CHUNKS = [
    Chunk(
        id="os-processes",
        document_id="os",
        content=(
            "A process is a program in execution. A process has its own "
            "execution state, memory mappings, and operating-system resources."
        ),
        metadata={
            "subject": "operating_systems",
            "topic": "processes",
        },
    ),
    Chunk(
        id="os-threads",
        document_id="os",
        content=(
            "A thread is the smallest unit of CPU execution within a process. "
            "Multiple threads belonging to the same process share its address "
            "space and other process resources."
        ),
        metadata={
            "subject": "operating_systems",
            "topic": "threads",
        },
    ),
    Chunk(
        id="os-scheduling",
        document_id="os",
        content=(
            "CPU scheduling determines which ready process should receive "
            "CPU time. Common scheduling policies include FCFS, SJF, "
            "priority scheduling, and round robin."
        ),
        metadata={
            "subject": "operating_systems",
            "topic": "cpu_scheduling",
        },
    ),
    Chunk(
        id="os-context-switching",
        document_id="os",
        content=(
            "A context switch occurs when the operating system saves the "
            "execution state of one process or thread and restores the state "
            "of another so that execution can continue."
        ),
        metadata={
            "subject": "operating_systems",
            "topic": "context_switching",
        },
    ),
    Chunk(
        id="os-deadlocks",
        document_id="os",
        content=(
            "Deadlock occurs when processes remain blocked because each "
            "process is waiting for resources held by another process. "
            "The processes cannot continue until the circular waiting "
            "condition is resolved."
        ),
        metadata={
            "subject": "operating_systems",
            "topic": "deadlocks",
        },
    ),
    Chunk(
        id="os-paging",
        document_id="os",
        content=(
            "Paging divides virtual memory into fixed-size pages and "
            "physical memory into fixed-size frames. Pages can be mapped "
            "to available physical frames by the operating system."
        ),
        metadata={
            "subject": "operating_systems",
            "topic": "paging",
        },
    ),
    Chunk(
        id="os-virtual-memory",
        document_id="os",
        content=(
            "Virtual memory allows a process to use an address space that "
            "can be larger than the available physical memory. The operating "
            "system can keep some portions of the address space outside "
            "physical memory until they are needed."
        ),
        metadata={
            "subject": "operating_systems",
            "topic": "virtual_memory",
        },
    ),
    Chunk(
        id="os-page-replacement",
        document_id="os",
        content=(
            "Page replacement determines which page in physical memory "
            "should be removed when a new page must be loaded and no free "
            "frame is available. FIFO and LRU are examples of page "
            "replacement algorithms."
        ),
        metadata={
            "subject": "operating_systems",
            "topic": "page_replacement",
        },
    ),
    Chunk(
        id="os-file-system",
        document_id="os",
        content=(
            "A file system organizes files and directories on storage "
            "devices. It maintains metadata about files and provides "
            "operations for creating, reading, writing, and deleting files."
        ),
        metadata={
            "subject": "operating_systems",
            "topic": "file_systems",
        },
    ),
    Chunk(
        id="os-synchronization",
        document_id="os",
        content=(
            "Process synchronization coordinates concurrent execution when "
            "multiple processes or threads access shared resources. "
            "Synchronization mechanisms such as mutexes and semaphores "
            "can help prevent race conditions."
        ),
        metadata={
            "subject": "operating_systems",
            "topic": "synchronization",
        },
    ),
]
