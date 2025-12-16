#include <unordered_map>
#include <omp.h>
#include "helpers.hpp"

unsigned long SequenceInfo::gpsa_sequential(float** S) {
    unsigned long visited = 0;

	// Boundary
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
		visited++;
	}

    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
		visited++;
	}
	
	// Main part
	for (unsigned int i = 1; i < rows; i++) {
		for (unsigned int j = 1; j < cols; j++) {
			float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
			float del = S[i - 1][j] + gap_penalty;
			float insert = S[i][j - 1] + gap_penalty;
			S[i][j] = std::max({match, del, insert});
		
			visited++;
		}
	}

    return visited;
}


unsigned long SequenceInfo::gpsa_taskloop(float** S, long grain_size=1, int block_size_x=1, int block_size_y=1) {
	
	
    const int BS_X = (block_size_x > 0) ? block_size_x : 1;
    const int BS_Y = (block_size_y > 0) ? block_size_y : 1;

    unsigned long visited_boundary = 0;
    for (int i = 1; i < (int)rows; i++) {
        S[i][0] = i * gap_penalty;
        visited_boundary++;
    }
    for (int j = 0; j < (int)cols; j++) {
        S[0][j] = j * gap_penalty;
        visited_boundary++;
    }

    const int num_rows_to_compute = (int)rows - 1;
    const int num_cols_to_compute = (int)cols - 1;
    const int NUM_BLOCKS_I = (num_rows_to_compute + BS_X - 1) / BS_X;
    const int NUM_BLOCKS_J = (num_cols_to_compute + BS_Y - 1) / BS_Y;
    
    unsigned long visited_main_grid = 0; 
    
    #pragma omp parallel default(none) \
        shared(S, X, Y, rows, cols, match_score, mismatch_score, gap_penalty, \
               BS_X, BS_Y, NUM_BLOCKS_I, NUM_BLOCKS_J, std::cout) \
        reduction(+:visited_main_grid)
    {
        #pragma omp single
        {
            const int max_block_diag = (NUM_BLOCKS_I - 1) + (NUM_BLOCKS_J - 1);
            
            for (int diag_b = 0; diag_b <= max_block_diag; ++diag_b) {

                #pragma omp taskloop grainsize(1) reduction(+:visited_main_grid)
                for (int bi = 0; bi < NUM_BLOCKS_I; ++bi) {
                    
                    int bj = diag_b - bi; 

                    if (bj >= 0 && bj < NUM_BLOCKS_J) {
                        
                        int i_start = bi * BS_X + 1;
                        int i_end   = std::min(i_start + BS_X, (int)rows);
                        int j_start = bj * BS_Y + 1;
                        int j_end   = std::min(j_start + BS_Y, (int)cols);

                        for (int i = i_start; i < i_end; ++i) {
                            for (int j = j_start; j < j_end; ++j) {
                                
                                float match = S[i-1][j-1] + (X[i-1] == Y[j-1] ? match_score : mismatch_score);
                                float del   = S[i-1][j] + gap_penalty;
                                float insert = S[i][j-1] + gap_penalty;
                                S[i][j] = std::max({match, del, insert});
                                
                                visited_main_grid++;
                            }
                        }
                    } 
                }

                #pragma omp taskwait
                
            } 
        } 
    } 

    return visited_boundary + visited_main_grid;
}

// Explicit tasks version grain size can be specified with grain_size, or block sizes. You can use both, or just one of them.
// Explicit tasks version with correct parallel counting
// Explicit tasks version with atomic counting

// Explicit tasks version with 'depend' and False Sharing fix
unsigned long SequenceInfo::gpsa_tasks(float** S, long grain_size=1, int block_size_x=1, int block_size_y=1) {
    
    
    unsigned long visited_boundary = 0; 
    
    const int BS_X = (block_size_x > 0) ? block_size_x : 1;
    const int BS_Y = (block_size_y > 0) ? block_size_y : 1;

    for (int i = 1; i < (int)rows; i++) {
        S[i][0] = i * gap_penalty;
        visited_boundary++;
    }
    for (int j = 0; j < (int)cols; j++) {
        S[0][j] = j * gap_penalty;
        visited_boundary++;
    }

    const int NUM_BLOCKS_I = ((int)rows - 1 + BS_X - 1) / BS_X;
    const int NUM_BLOCKS_J = ((int)cols - 1 + BS_Y - 1) / BS_Y;
    
    unsigned long visited_main_grid = 0; 
    
    unsigned long* thread_local_counts = nullptr;
    int max_threads = 1;

    const int CACHE_LINE_PADDING = 8; 

    void*** dep_grid = new void**[NUM_BLOCKS_I];
    for (int i = 0; i < NUM_BLOCKS_I; ++i) {
        dep_grid[i] = new void*[NUM_BLOCKS_J](); 
    }
        
    #pragma omp parallel default(none) \
        shared(S, X, Y, rows, cols, match_score, mismatch_score, gap_penalty, \
               BS_X, BS_Y, NUM_BLOCKS_I, NUM_BLOCKS_J, \
               thread_local_counts, max_threads, dep_grid, CACHE_LINE_PADDING)
    {
        #pragma omp single
        {
            max_threads = omp_get_max_threads();
            
            thread_local_counts = new unsigned long[max_threads * CACHE_LINE_PADDING](); 
            
            for (int bi = 0; bi < NUM_BLOCKS_I; ++bi) {
                for (int bj = 0; bj < NUM_BLOCKS_J; ++bj) {
                    
                    int i_start = bi * BS_X + 1;
                    int i_end   = std::min(i_start + BS_X, (int)rows);
                    int j_start = bj * BS_Y + 1;
                    int j_end   = std::min(j_start + BS_Y, (int)cols);
                    
                    #pragma omp task firstprivate(i_start, i_end, j_start, j_end, bi, bj) \
                        depend(in: dep_grid[bi-1][bj]) \
                        depend(in: dep_grid[bi][bj-1]) \
                        depend(out: dep_grid[bi][bj])
                    {
                        int tid = omp_get_thread_num();
                        int i, j;
                        float match, del, insert;

                        for (i = i_start; i < i_end; ++i) {
                            for (j = j_start; j < j_end; ++j) {
                                
                                match = S[i-1][j-1] + (X[i-1] == Y[j-1] ? match_score : mismatch_score);
                                del = S[i-1][j] + gap_penalty;
                                insert = S[i][j-1] + gap_penalty;
                                S[i][j] = std::max({match, del, insert});
                                
                                thread_local_counts[tid * CACHE_LINE_PADDING]++;
                            }
                        }
                    } 
                } 
            } 
        } 
    } 

    for (int i = 0; i < max_threads; i++) {
        visited_main_grid += thread_local_counts[i * CACHE_LINE_PADDING];
    }

    delete[] thread_local_counts;
    for (int i = 0; i < NUM_BLOCKS_I; ++i) {
        delete[] dep_grid[i];
    }
    delete[] dep_grid;

    return visited_boundary + visited_main_grid;
}