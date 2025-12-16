The main goal is the **maximum parallelization** of the computationally intensive matrix calculation using **OpenMP Tasks** to significantly reduce runtime on multi-core processors. The focus is on the **efficient resolution of complex data dependencies** within the Dynamic Programming (DP) grid.

## 🔑 Core Functionality and Methodology

The project includes two main approaches to parallelize the DP matrix computation:

1.  **Taskloop Parallelization (Diagonal):** Utilizing `#pragma omp taskloop` to parallelize loops along the anti-diagonals, with implicit synchronization via `#pragma omp taskwait`.
2.  **Explicit Task Parallelization with Dependencies:** The advanced approach uses **`#pragma omp task depend`** clauses to precisely control the strict order (top-left-up-left) of the DP calculation. This allows for asynchronous execution and better load balancing.
3.  **Performance Engineering:** Implementation of optimization techniques to reduce parallelization overhead, including:
    * **False Sharing Prevention:** Padding of thread-local counters (`CACHE_LINE_PADDING`) to avoid cache line contention.
    * **Thread-local Reduction:** Use of thread-specific counters for efficient aggregation of calculations.

## 🛠️ Technologies

* **Language:** C++
* **Parallelization:** OpenMP (Tasks, Taskloop, `depend` Clauses, Reduction)
* **Algorithm:** Dynamic Programming (Needleman-Wunsch equivalent for Sequence Alignment)
  
