// Copyright (c) 2009-2021, Tor M. Aamodt, Inderpreet Singh, Vijay Kandiah,
// Nikos Hardavellas, Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The
// University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
#define ABSTRACT_HARDWARE_MODEL_INCLUDED

// Forward declarations
class gpgpu_sim;
class kernel_info_t;
class gpgpu_context;

// Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32
#define MAX_BARRIERS_PER_CTA 16

// After expanding the vector input and output operands
#define MAX_INPUT_VALUES 24
#define MAX_OUTPUT_VALUES 8

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned u32;

enum _memory_space_t {
  undefined_space = 0,
  reg_space,
  local_space,
  shared_space,
  sstarr_space,
  param_space_unclassified,
  param_space_kernel, /* global to all threads in a kernel : read-only */
  param_space_local,  /* local to a thread : read-writable */
  const_space,
  tex_space,
  surf_space,
  global_space,
  generic_space,
  instruction_space,
  NUM_MEMORY_SPACE
};
const char* memory_space_str(enum _memory_space_t type);

#ifndef COEFF_STRUCT
#define COEFF_STRUCT

struct PowerscalingCoefficients {
  double int_coeff;
  double int_mul_coeff;
  double int_mul24_coeff;
  double int_mul32_coeff;
  double int_div_coeff;
  double fp_coeff;
  double dp_coeff;
  double fp_mul_coeff;
  double fp_div_coeff;
  double dp_mul_coeff;
  double dp_div_coeff;
  double sqrt_coeff;
  double log_coeff;
  double sin_coeff;
  double exp_coeff;
  double tensor_coeff;
  double tex_coeff;
};
#endif

enum FuncCache {
  FuncCachePreferNone = 0,
  FuncCachePreferShared = 1,
  FuncCachePreferL1 = 2
};

enum AdaptiveCache { FIXED = 0, ADAPTIVE_CACHE = 1 };

#ifdef __cplusplus

#include <stdio.h>
#include <string.h>
#include <set>
#include <unordered_map>
#include <sstream>

typedef u64 new_addr_type;
typedef u64 cudaTextureObject_t;
typedef u64 address_type;
typedef u64 addr_t;

// the following are operations the timing model can see
#define SPECIALIZED_UNIT_NUM 8
#define SPEC_UNIT_START_ID 100

enum uarch_op_t {
  NO_OP = 0,
  ALU_OP = 1,
  SFU_OP,
  TENSOR_CORE_OP,
  DP_OP,
  SP_OP,
  INTP_OP,
  ALU_SFU_OP,
  LOAD_OP,
  TENSOR_CORE_LOAD_OP,
  TENSOR_CORE_STORE_OP,
  STORE_OP,
  BRANCH_OP,
  BARRIER_OP,
  MEMORY_BARRIER_OP,
  CALL_OPS,
  RET_OPS,
  EXIT_OPS,
  SPECIALIZED_UNIT_1_OP = SPEC_UNIT_START_ID,
  SPECIALIZED_UNIT_2_OP,
  SPECIALIZED_UNIT_3_OP,
  SPECIALIZED_UNIT_4_OP,
  SPECIALIZED_UNIT_5_OP,
  SPECIALIZED_UNIT_6_OP,
  SPECIALIZED_UNIT_7_OP,
  SPECIALIZED_UNIT_8_OP
};
typedef enum uarch_op_t op_type;

enum uarch_bar_t { NOT_BAR = -1, SYNC = 1, ARRIVE, RED };
typedef enum uarch_bar_t barrier_type;

enum uarch_red_t { NOT_RED = -1, POPC_RED = 1, AND_RED, OR_RED };
typedef enum uarch_red_t reduction_type;

enum uarch_operand_type_t { UN_OP = -1, INT_OP, FP_OP };
typedef enum uarch_operand_type_t types_of_operands;

enum special_operations_t {
  OTHER_OP,
  INT__OP,
  INT_MUL24_OP,
  INT_MUL32_OP,
  INT_MUL_OP,
  INT_DIV_OP,
  FP_MUL_OP,
  FP_DIV_OP,
  FP__OP,
  FP_SQRT_OP,
  FP_LG_OP,
  FP_SIN_OP,
  FP_EXP_OP,
  DP_MUL_OP,
  DP_DIV_OP,
  DP___OP,
  TENSOR__OP,
  TEX__OP
};

// Required to identify for the power model
typedef enum special_operations_t special_ops;
enum operation_pipeline_t {
  UNKOWN_OP,
  SP__OP,
  DP__OP,
  INTP__OP,
  SFU__OP,
  TENSOR_CORE__OP,
  MEM__OP,
  SPECIALIZED__OP,
}; // lat > 1
typedef enum operation_pipeline_t operation_pipeline;
enum mem_operation_t { NOT_TEX, TEX };
typedef enum mem_operation_t mem_operation;

enum _memory_op_t { no_memory_op = 0, memory_load, memory_store };

#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <vector>

#if !defined(__VECTOR_TYPES_H__)
#include "vector_types.h"
#endif
struct dim3comp {
  bool operator()(const dim3 &a, const dim3 &b) const {
    if (a.z < b.z)
      return true;
    else if (a.y < b.y)
      return true;
    else if (a.x < b.x)
      return true;
    else
      return false;
  }
};

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound);

// Jin: child kernel information for CDP
#include "stream_manager.h"
class stream_manager;
struct CUstream_st;
// extern stream_manager * g_stream_manager;
// support for pinned memories added
extern std::map<void *, void **> pinned_memory;
extern std::map<void *, size_t> pinned_memory_size;

class kernel_info_t {
 public:
  kernel_info_t(dim3 gridDim, dim3 blockDim, class function_info *entry, u64 streamID);
  kernel_info_t(
      dim3 gridDim, dim3 blockDim, class function_info *entry,
      std::map<std::string, const struct cudaArray *> nameToCudaArray,
      std::map<std::string, const struct textureInfo *> nameToTextureInfo);
  ~kernel_info_t();

  void inc_running() { m_num_cores_running++; }
  void dec_running() {
    assert(m_num_cores_running > 0);
    m_num_cores_running--;
  }
  bool running() const { return m_num_cores_running > 0; }
  bool done() const { return no_more_ctas_to_run() && !running(); }
  class function_info *entry() {
    return m_kernel_entry;
  }
  const class function_info *entry() const { return m_kernel_entry; }

  size_t num_blocks() const {
    return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
  }

  size_t threads_per_cta() const {
    return m_block_dim.x * m_block_dim.y * m_block_dim.z;
  }

  dim3 get_grid_dim() const { return m_grid_dim; }
  dim3 get_cta_dim() const { return m_block_dim; }

  void increment_cta_id() {
    increment_x_then_y_then_z(m_next_cta, m_grid_dim);
    m_next_tid.x = 0;
    m_next_tid.y = 0;
    m_next_tid.z = 0;
  }
  dim3 get_next_cta_id() const { return m_next_cta; }
  u32 get_next_cta_id_single() const {
    return m_next_cta.x + m_grid_dim.x * m_next_cta.y +
           m_grid_dim.x * m_grid_dim.y * m_next_cta.z;
  }
  bool no_more_ctas_to_run() const {
    return (m_next_cta.x >= m_grid_dim.x || m_next_cta.y >= m_grid_dim.y ||
            m_next_cta.z >= m_grid_dim.z);
  }

  void increment_thread_id() {
    increment_x_then_y_then_z(m_next_tid, m_block_dim);
  }
  dim3 get_next_thread_id_3d() const { return m_next_tid; }
  u32 get_next_thread_id() const {
    return m_next_tid.x + m_block_dim.x * m_next_tid.y +
           m_block_dim.x * m_block_dim.y * m_next_tid.z;
  }
  bool more_threads_in_cta() const {
    return m_next_tid.z < m_block_dim.z && m_next_tid.y < m_block_dim.y &&
           m_next_tid.x < m_block_dim.x;
  }
  u32 get_uid() const { return m_uid; }
  u64 get_streamID() const { return m_streamID; }
  std::string get_name() const { return name(); }
  std::string name() const;

  std::list<class ptx_thread_info *> &active_threads() {
    return m_active_threads;
  }
  class memory_space *get_param_memory() {
    return m_param_mem;
  }

  // The following functions access texture bindings present at the kernel's
  // launch

  const struct cudaArray *get_texarray(const std::string &texname) const {
    std::map<std::string, const struct cudaArray *>::const_iterator t =
        m_NameToCudaArray.find(texname);
    assert(t != m_NameToCudaArray.end());
    return t->second;
  }

  const struct textureInfo *get_texinfo(const std::string &texname) const {
    std::map<std::string, const struct textureInfo *>::const_iterator t =
        m_NameToTextureInfo.find(texname);
    assert(t != m_NameToTextureInfo.end());
    return t->second;
  }

 private:
  kernel_info_t(const kernel_info_t &);   // disable copy constructor
  void operator=(const kernel_info_t &);  // disable copy operator

  class function_info *m_kernel_entry;

  u32 m_uid;  // Kernel ID
  u64 m_streamID;

  // These maps contain the snapshot of the texture mappings at kernel launch
  std::map<std::string, const struct cudaArray *> m_NameToCudaArray;
  std::map<std::string, const struct textureInfo *> m_NameToTextureInfo;

  dim3 m_grid_dim;
  dim3 m_block_dim;
  dim3 m_next_cta;
  dim3 m_next_tid;

  u32 m_num_cores_running;

  std::list<class ptx_thread_info *> m_active_threads;
  class memory_space *m_param_mem;

 public:
  // Jin: parent and child kernel management for CDP
  void set_parent(kernel_info_t *parent, dim3 parent_ctaid, dim3 parent_tid);
  void set_child(kernel_info_t *child);
  void remove_child(kernel_info_t *child);
  bool is_finished();
  bool children_all_finished();
  void notify_parent_finished();
  CUstream_st *create_stream_cta(dim3 ctaid);
  CUstream_st *get_default_stream_cta(dim3 ctaid);
  bool cta_has_stream(dim3 ctaid, CUstream_st *stream);
  void destroy_cta_streams();
  void print_parent_info();
  kernel_info_t *get_parent() { return m_parent_kernel; }

 private:
  kernel_info_t *m_parent_kernel;
  dim3 m_parent_ctaid;
  dim3 m_parent_tid;
  std::list<kernel_info_t *> m_child_kernels;  // child kernel launched
  std::map<dim3, std::list<CUstream_st *>, dim3comp>
      m_cta_streams;  // streams created in each CTA

  // Jin: kernel timing
 public:
  u32 allocated_ctas;
  u64 launch_cycle;
  u64 start_cycle;
  u64 end_cycle;
  u32 m_launch_latency;

  mutable bool cache_config_set;

  // this used for any CPU-GPU kernel latency and counted in the gpu_cycle
  u32 m_kernel_TB_latency;  
};

class core_config {
 public:
  core_config(gpgpu_context *ctx) {
    gpgpu_ctx = ctx;
    m_valid = false;
    num_shmem_bank = 16;
    shmem_limited_broadcast = false;
    gpgpu_shmem_sizeDefault = (unsigned)-1;
    gpgpu_shmem_sizePrefL1 = (unsigned)-1;
    gpgpu_shmem_sizePrefShared = (unsigned)-1;
  }
  virtual void init() = 0;

  bool m_valid;
  u32 warp_size;
  // backward pointer
  class gpgpu_context *gpgpu_ctx;

  // off-chip memory request architecture parameters
  int gpgpu_coalesce_arch;

  // shared memory bank conflict checking parameters
  bool shmem_limited_broadcast;
  static const address_type WORD_SIZE = 4;
  u32 num_shmem_bank;
  u32 shmem_bank_func(address_type addr) const {
    return ((addr / WORD_SIZE) % num_shmem_bank);
  }
  u32 mem_warp_parts;
  mutable u32 gpgpu_shmem_size;
  char *gpgpu_shmem_option;
  std::vector<unsigned> shmem_opt_list;
  u32 gpgpu_shmem_sizeDefault;
  u32 gpgpu_shmem_sizePrefL1;
  u32 gpgpu_shmem_sizePrefShared;
  u32 mem_unit_ports;

  // texture and constant cache line sizes (used to determine number of memory
  // accesses)
  u32 gpgpu_cache_texl1_linesize;
  u32 gpgpu_cache_constl1_linesize;

  u32 gpgpu_max_insn_issue_per_warp;
  bool gmem_skip_L1D;  // on = global memory access always skip the L1 cache

  bool adaptive_cache_config;
};

// bounded stack that implements simt reconvergence using pdom mechanism from
// MICRO'07 paper
const u32 MAX_WARP_SIZE = 32;
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;
#define MAX_WARP_SIZE_SIMT_STACK MAX_WARP_SIZE
typedef std::bitset<MAX_WARP_SIZE_SIMT_STACK> simt_mask_t;
typedef std::vector<address_type> addr_vector_t;

class simt_stack {
 public:
  simt_stack(u32 wid, u32 warpSize, class gpgpu_sim *gpu);

  void reset();
  void launch(address_type start_pc, const simt_mask_t &active_mask);
  void update(simt_mask_t &thread_done, addr_vector_t &next_pc,
              address_type recvg_pc, op_type next_inst_op,
              u32 next_inst_size, address_type next_inst_pc);

  const simt_mask_t &get_active_mask() const;
  void get_pdom_stack_top_info(u32 *pc, u32 *rpc) const;
  u32 get_rp() const;
  void print(FILE *fp) const;
  void resume(char *fname);
  void print_checkpoint(FILE *fout) const;

 protected:
  u32 m_warp_id;
  u32 m_warp_size;

  enum stack_entry_type { STACK_ENTRY_TYPE_NORMAL = 0, STACK_ENTRY_TYPE_CALL };

  struct simt_stack_entry {
    address_type m_pc;
    u32 m_calldepth;
    simt_mask_t m_active_mask;
    address_type m_recvg_pc;
    u64 m_branch_div_cycle;
    stack_entry_type m_type;
    simt_stack_entry()
        : m_pc(-1),
          m_calldepth(0),
          m_active_mask(),
          m_recvg_pc(-1),
          m_branch_div_cycle(0),
          m_type(STACK_ENTRY_TYPE_NORMAL){};
  };

  std::deque<simt_stack_entry> m_stack;

  class gpgpu_sim *m_gpu;
};

// Let's just upgrade to C++11 so we can use constexpr here...
// start allocating from this address (lower values used for allocating globals
// in .ptx file)
const u64 GLOBAL_HEAP_START = 0xC0000000;
// Volta max shmem size is 96kB
const u64 SHARED_MEM_SIZE_MAX = 96 * (1 << 10);
// Volta max local mem is 16kB
const u64 LOCAL_MEM_SIZE_MAX = 1 << 14;
// Volta Titan V has 80 SMs
const u32 MAX_STREAMING_MULTIPROCESSORS = 80;
// Max 2048 threads / SM
const u32 MAX_THREAD_PER_SM = 1 << 11;
// MAX 64 warps / SM
const u32 MAX_WARP_PER_SM = 1 << 6;
const u64 TOTAL_LOCAL_MEM_PER_SM =
    MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
const u64 TOTAL_SHARED_MEM =
    MAX_STREAMING_MULTIPROCESSORS * SHARED_MEM_SIZE_MAX;
const u64 TOTAL_LOCAL_MEM =
    MAX_STREAMING_MULTIPROCESSORS * MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
const u64 SHARED_GENERIC_START =
    GLOBAL_HEAP_START - TOTAL_SHARED_MEM;
const u64 LOCAL_GENERIC_START =
    SHARED_GENERIC_START - TOTAL_LOCAL_MEM;
const u64 STATIC_ALLOC_LIMIT =
    GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM + TOTAL_SHARED_MEM);

#if !defined(__CUDA_RUNTIME_API_H__)

#include "builtin_types.h"

struct cudaArray {
  void *devPtr;
  int devPtr32;
  struct cudaChannelFormatDesc desc;
  int width;
  int height;
  int size;  // in bytes
  u32 dimensions;
};

#endif

// Struct that record other attributes in the textureReference declaration
// - These attributes are passed thru __cudaRegisterTexture()
struct textureReferenceAttr {
  const struct textureReference *m_texref;
  int m_dim;
  enum cudaTextureReadMode m_readmode;
  int m_ext;
  textureReferenceAttr(const struct textureReference *texref, int dim,
                       enum cudaTextureReadMode readmode, int ext)
      : m_texref(texref), m_dim(dim), m_readmode(readmode), m_ext(ext) {}
};

class gpgpu_functional_sim_config {
 public:
  void reg_options(class OptionParser *opp);

  void ptx_set_tex_cache_linesize(u32 linesize);

  u32 get_forced_max_capability() const {
    return m_ptx_force_max_capability;
  }
  bool convert_to_ptxplus() const { return m_ptx_convert_to_ptxplus; }
  bool use_cuobjdump() const { return m_ptx_use_cuobjdump; }
  bool experimental_lib_support() const { return m_experimental_lib_support; }

  int get_ptx_inst_debug_to_file() const { return g_ptx_inst_debug_to_file; }
  const char *get_ptx_inst_debug_file() const { return g_ptx_inst_debug_file; }
  int get_ptx_inst_debug_thread_uid() const {
    return g_ptx_inst_debug_thread_uid;
  }
  u32 get_texcache_linesize() const { return m_texcache_linesize; }
  int get_checkpoint_option() const { return checkpoint_option; }
  int get_checkpoint_kernel() const { return checkpoint_kernel; }
  int get_checkpoint_CTA() const { return checkpoint_CTA; }
  int get_resume_option() const { return resume_option; }
  int get_resume_kernel() const { return resume_kernel; }
  int get_resume_CTA() const { return resume_CTA; }
  int get_checkpoint_CTA_t() const { return checkpoint_CTA_t; }
  int get_checkpoint_insn_Y() const { return checkpoint_insn_Y; }

 private:
  // PTX options
  int m_ptx_convert_to_ptxplus;
  int m_ptx_use_cuobjdump;
  int m_experimental_lib_support;
  u32 m_ptx_force_max_capability;
  int checkpoint_option;
  int checkpoint_kernel;
  int checkpoint_CTA;
  u32 resume_option;
  u32 resume_kernel;
  u32 resume_CTA;
  u32 checkpoint_CTA_t;
  int checkpoint_insn_Y;
  int g_ptx_inst_debug_to_file;
  char *g_ptx_inst_debug_file;
  int g_ptx_inst_debug_thread_uid;

  u32 m_texcache_linesize;
};

class gpgpu_t {
 public:
  gpgpu_t(const gpgpu_functional_sim_config &config, gpgpu_context *ctx);
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  int checkpoint_option;
  int checkpoint_kernel;
  int checkpoint_CTA;
  u32 resume_option;
  u32 resume_kernel;
  u32 resume_CTA;
  u32 checkpoint_CTA_t;
  int checkpoint_insn_Y;

  u64 get_cycle() const;

  // Move some cycle core stats here instead of being global
  u64 gpu_sim_cycle;
  u64 gpu_tot_sim_cycle;

  u64 tot_l1d_lat_from_sched_to_access;
  u64 tot_l1d_accesses;
  float avg_l1d_lat_from_sched_to_access;
  u64 tot_l1d_hits;

  u64 tot_l1d_wr_lat_from_sched;
  u64 tot_l1d_writes;
  float avg_l1d_wr_lat_from_sched;
  u64 tot_l1d_wr_hits;
  float l1d_wr_hit_rate;

  u64 tot_l1d_rd_lat_from_sched;
  u64 tot_l1d_reads;  
  float avg_l1d_rd_lat_from_sched;
  u64 tot_l1d_rd_hits;
  float l1d_rd_hit_rate;

  std::unordered_map<u64 /* pc */, u64 /* cycle */> sched_cycle;
  float avg_alu_lat; // from scheduling to wb

  void *gpu_malloc(size_t size);
  void *gpu_mallocarray(size_t count);
  void gpu_memset(size_t dst_start_addr, int c, size_t count);
  void memcpy_to_gpu(size_t dst_start_addr, const void *src, size_t count);
  void memcpy_from_gpu(void *dst, size_t src_start_addr, size_t count);
  void memcpy_gpu_to_gpu(size_t dst, size_t src, size_t count);

  class memory_space *get_global_memory() {
    return m_global_mem;
  }
  class memory_space *get_tex_memory() {
    return m_tex_mem;
  }
  class memory_space *get_surf_memory() {
    return m_surf_mem;
  }

  void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference *texref,
                                        const struct cudaArray *array);
  void gpgpu_ptx_sim_bindNameToTexture(const char *name,
                                       const struct textureReference *texref,
                                       int dim, int readmode, int ext);
  void gpgpu_ptx_sim_unbindTexture(const struct textureReference *texref);
  const char *gpgpu_ptx_sim_findNamefromTexture(
      const struct textureReference *texref);

  const struct textureReference *get_texref(const std::string &texname) const {
    std::map<std::string,
             std::set<const struct textureReference *> >::const_iterator t =
        m_NameToTextureRef.find(texname);
    assert(t != m_NameToTextureRef.end());
    return *(t->second.begin());
  }

  const struct cudaArray *get_texarray(const std::string &texname) const {
    std::map<std::string, const struct cudaArray *>::const_iterator t =
        m_NameToCudaArray.find(texname);
    assert(t != m_NameToCudaArray.end());
    return t->second;
  }

  const struct textureInfo *get_texinfo(const std::string &texname) const {
    std::map<std::string, const struct textureInfo *>::const_iterator t =
        m_NameToTextureInfo.find(texname);
    assert(t != m_NameToTextureInfo.end());
    return t->second;
  }

  const struct textureReferenceAttr *get_texattr(
      const std::string &texname) const {
    std::map<std::string, const struct textureReferenceAttr *>::const_iterator
        t = m_NameToAttribute.find(texname);
    assert(t != m_NameToAttribute.end());
    return t->second;
  }

  const gpgpu_functional_sim_config &get_config() const {
    return m_function_model_config;
  }
  FILE *get_ptx_inst_debug_file() { return ptx_inst_debug_file; }

  //  These maps return the current texture mappings for the GPU at any given
  //  time.
  std::map<std::string, const struct cudaArray *> getNameArrayMapping() {
    return m_NameToCudaArray;
  }
  std::map<std::string, const struct textureInfo *> getNameInfoMapping() {
    return m_NameToTextureInfo;
  }

  virtual ~gpgpu_t() {}

 protected:
  const gpgpu_functional_sim_config &m_function_model_config;
  FILE *ptx_inst_debug_file;

  class memory_space *m_global_mem;
  class memory_space *m_tex_mem;
  class memory_space *m_surf_mem;

  u64 m_dev_malloc;
  //  These maps contain the current texture mappings for the GPU at any given
  //  time.
  std::map<std::string, std::set<const struct textureReference *>> m_NameToTextureRef;
  std::map<const struct textureReference *, std::string> m_TextureRefToName;
  std::map<std::string, const struct cudaArray *> m_NameToCudaArray;
  std::map<std::string, const struct textureInfo *> m_NameToTextureInfo;
  std::map<std::string, const struct textureReferenceAttr *> m_NameToAttribute;
};

struct gpgpu_ptx_sim_info {
  // Holds properties of the kernel (Kernel's resource use).
  // These will be set to zero if a ptxinfo file is not present.
  u32 lmem;
  u32 smem;
  u32 cmem;
  u32 gmem;
  u32 regs; // per thread allocated regs
  u32 barriers;
  u32 maxthreads;
  u32 ptx_version;
  u32 sm_target;
};

struct gpgpu_ptx_sim_arg {
  gpgpu_ptx_sim_arg() { m_start = NULL; }
  gpgpu_ptx_sim_arg(const void *arg, size_t size, size_t offset) {
    m_start = arg;
    m_nbytes = size;
    m_offset = offset;
  }
  const void *m_start;
  size_t m_nbytes;
  size_t m_offset;
};

typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

class memory_space_t {
 public:
  memory_space_t() {
    m_type = undefined_space;
    m_bank = 0;
  }
  memory_space_t(const enum _memory_space_t &from) {
    m_type = from;
    m_bank = 0;
  }
  bool operator==(const memory_space_t &x) const {
    return (m_bank == x.m_bank) && (m_type == x.m_type);
  }
  bool operator!=(const memory_space_t &x) const { return !(*this == x); }
  bool operator<(const memory_space_t &x) const {
    if (m_type < x.m_type)
      return true;
    else if (m_type > x.m_type)
      return false;
    else if (m_bank < x.m_bank)
      return true;
    return false;
  }
  enum _memory_space_t get_type() const { return m_type; }
  void set_type(enum _memory_space_t t) { m_type = t; }
  u32 get_bank() const { return m_bank; }
  void set_bank(u32 b) { m_bank = b; }
  bool is_const() const {
    return (m_type == const_space) || (m_type == param_space_kernel);
  }
  bool is_local() const {
    return (m_type == local_space) || (m_type == param_space_local);
  }
  bool is_global() const { return (m_type == global_space); }

 private:
  enum _memory_space_t m_type;
  u32 m_bank;  // n in ".const[n]"; note .const == .const[0] (see PTX 2.1
                    // manual, sec. 5.1.3)
};

// u32 SECTOR_SIZE;
// u32 MAX_MEMORY_ACCESS_SIZE;

// const char* arch_option = std::getenv("USE_ARISE_ARCH");
// #ifdef ARISE2_L1P5
// const u32 SECTOR_SIZE = 64;        // sector is 64 bytes width
// const u32 MAX_MEMORY_ACCESS_SIZE = 256;
// #else
// const u32 SECTOR_SIZE = 32;        // sector is 32 bytes width
// const u32 MAX_MEMORY_ACCESS_SIZE = 128;
// #endif

const u32 SECTOR_SIZE = 32;        // sector is 32 bytes width
const u32 MAX_MEMORY_ACCESS_SIZE = 128;
typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
const u32 SECTOR_CHUNK_SIZE = 4;  // four sectors
typedef std::bitset<SECTOR_CHUNK_SIZE> mem_access_sector_mask_t;
#define NO_PARTIAL_WRITE (mem_access_byte_mask_t())

enum mem_access_type {
  GLOBAL_ACC_R = 0,
  LOCAL_ACC_R = 1,
  CONST_ACC_R,
  TEXTURE_ACC_R,
  GLOBAL_ACC_W,
  LOCAL_ACC_W,
  L1_WRBK_ACC,
  L2_WRBK_ACC,
  INST_ACC_R,
  L1_WR_ALLOC_R,
  L2_WR_ALLOC_R,
  NUM_MEM_ACCESS_TYPE
};
const char *mem_access_type_str(enum mem_access_type access_type);

const char *uarch_op_str(enum uarch_op_t op_type);
const char *memory_op_str(enum _memory_op_t memory_op);

enum cache_operator_type {
  CACHE_UNDEFINED,

  // loads
  CACHE_ALL,       // .ca
  CACHE_LAST_USE,  // .lu
  CACHE_VOLATILE,  // .cv
  CACHE_L1,        // .nc

  // loads and stores
  CACHE_STREAMING,  // .cs
  CACHE_GLOBAL,     // .cg

  // stores
  CACHE_WRITE_BACK,    // .wb
  CACHE_WRITE_THROUGH  // .wt
};
const char *cache_op_str(enum cache_operator_type cache_op);

class mem_access_t {
 public:
  mem_access_t(gpgpu_context *ctx) { init(ctx); }
  mem_access_t(mem_access_type type, new_addr_type address, u32 size,
               bool wr, gpgpu_context *ctx) {
    init(ctx);
    m_type = type;
    m_addr = address;
    m_req_size = size;
    m_write = wr;
  }
  mem_access_t(mem_access_type type, new_addr_type address, u32 size,
               bool wr, const active_mask_t &active_mask,
               const mem_access_byte_mask_t &byte_mask,
               const mem_access_sector_mask_t &sector_mask, gpgpu_context *ctx)
      : m_warp_mask(active_mask),
        m_byte_mask(byte_mask),
        m_sector_mask(sector_mask) {
    init(ctx);
    m_type = type;
    m_addr = address;
    m_req_size = size;
    m_write = wr;
  }

  new_addr_type get_addr() const { return m_addr; }
  void set_addr(new_addr_type addr) { m_addr = addr; }
  u32 get_size() const { return m_req_size; }
  const active_mask_t &get_warp_mask() const { return m_warp_mask; }
  bool is_write() const { return m_write; }
  enum mem_access_type get_type() const { return m_type; }
  mem_access_byte_mask_t get_byte_mask() const { return m_byte_mask; }
  mem_access_sector_mask_t get_sector_mask() const { return m_sector_mask; }

  void print(FILE *fp) const {
    fprintf(fp, "addr=0x%llx, %s, size=%u, ", m_addr,
            m_write ? "store" : "load ", m_req_size);
    switch (m_type) {
      case GLOBAL_ACC_R:
        fprintf(fp, "GLOBAL_R");
        break;
      case LOCAL_ACC_R:
        fprintf(fp, "LOCAL_R ");
        break;
      case CONST_ACC_R:
        fprintf(fp, "CONST   ");
        break;
      case TEXTURE_ACC_R:
        fprintf(fp, "TEXTURE ");
        break;
      case GLOBAL_ACC_W:
        fprintf(fp, "GLOBAL_W");
        break;
      case LOCAL_ACC_W:
        fprintf(fp, "LOCAL_W ");
        break;
      case L2_WRBK_ACC:
        fprintf(fp, "L2_WRBK ");
        break;
      case INST_ACC_R:
        fprintf(fp, "INST    ");
        break;
      case L1_WRBK_ACC:
        fprintf(fp, "L1_WRBK ");
        break;
      default:
        fprintf(fp, "unknown ");
        break;
    }
  }

  gpgpu_context *gpgpu_ctx;

 private:
  void init(gpgpu_context *ctx);

  u32 m_uid;
  new_addr_type m_addr;  // request address
  bool m_write;
  u32 m_req_size;  // bytes
  mem_access_type m_type;
  active_mask_t m_warp_mask;
  mem_access_byte_mask_t m_byte_mask;
  mem_access_sector_mask_t m_sector_mask;
};

class mem_fetch;

class mem_fetch_interface {
  friend class baseline_cache;
  public:
    mem_fetch_interface() {
      m_if_name     = "mem_fetch_interface";
      m_push_q_name = "xxx";
    }
    virtual bool full(u32 size, bool write) const = 0;
    virtual void push(mem_fetch *mf) = 0;
    std::string get_if_name() { return m_if_name; }
    std::string get_push_q_name() { return m_push_q_name; }
  protected:
    void set_if_name(std::string if_name) {
      m_if_name = if_name;
    }
  private:
    std::string m_if_name;
    std::string m_push_q_name;
};

class mem_fetch_allocator {
 public:
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           u32 size, bool wr, u64 cycle,
                           u64 streamID) const = 0;
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           u64 cycle) const = 0;
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           u32 size, bool wr, u64 cycle,
                           u32 wid, u32 sid, u32 tpc,
                           mem_fetch *original_mf,
                           u64 streamID) const = 0;
};

// the maximum number of destination, source, or address uarch operands in a
// instruction
#define MAX_REG_OPERANDS 32

struct dram_callback_t {
  dram_callback_t() {
    function = NULL;
    instruction = NULL;
    thread = NULL;
  }
  void (*function)(const class inst_t *, class ptx_thread_info *);

  const class inst_t *instruction;
  class ptx_thread_info *thread;
};

class inst_t {
 public:
  inst_t() {
    m_decoded = false;
    pc = (address_type) - 1;
    trace_opcode = "xxxx";
    reconvergence_pc = (address_type) - 1;
    op = NO_OP;    
    bar_type = NOT_BAR;
    red_type = NOT_RED;
    bar_id = (unsigned)-1;
    bar_count = (unsigned)-1;
    oprnd_type = UN_OP;
    sp_op = OTHER_OP;
    op_pipe = UNKOWN_OP;
    mem_op = NOT_TEX;
    const_cache_operand = 0;
    num_operands = 0;
    num_regs = 0;
    memset(out, 0, sizeof(unsigned));
    memset(in, 0, sizeof(unsigned));
    is_vectorin = 0;
    is_vectorout = 0;
    space = memory_space_t();
    cache_op = CACHE_UNDEFINED;
    latency = 1;
    initiation_interval = 1;
    for (u32 i = 0; i < MAX_REG_OPERANDS; i++) {
      arch_reg.src[i] = -1;
      arch_reg.dst[i] = -1;
    }
    isize = 0;
  }
  bool valid() const { return m_decoded; }
  virtual void print_insn(FILE *fp) const {
    fprintf(fp, " [inst @ pc=0x%04llx] ", pc);
  }
  bool is_load() const {
    return (op == LOAD_OP || op == TENSOR_CORE_LOAD_OP ||
            memory_op == memory_load);
  }
  bool is_store() const {
    return (op == STORE_OP || op == TENSOR_CORE_STORE_OP ||
            memory_op == memory_store);
  }

  bool is_fp() const { return ((sp_op == FP__OP)); }  // VIJAY
  bool is_fpdiv() const { return ((sp_op == FP_DIV_OP)); }
  bool is_fpmul() const { return ((sp_op == FP_MUL_OP)); }
  bool is_dp() const { return ((sp_op == DP___OP)); }
  bool is_dpdiv() const { return ((sp_op == DP_DIV_OP)); }
  bool is_dpmul() const { return ((sp_op == DP_MUL_OP)); }
  bool is_imul() const { return ((sp_op == INT_MUL_OP)); }
  bool is_imul24() const { return ((sp_op == INT_MUL24_OP)); }
  bool is_imul32() const { return ((sp_op == INT_MUL32_OP)); }
  bool is_idiv() const { return ((sp_op == INT_DIV_OP)); }
  bool is_sfu() const {
    return ((sp_op == FP_SQRT_OP) || (sp_op == FP_LG_OP) ||
            (sp_op == FP_SIN_OP) || (sp_op == FP_EXP_OP) ||
            (sp_op == TENSOR__OP));
  }
  bool is_alu() const { return (sp_op == INT__OP); }

  u32 get_num_operands() const { return num_operands; }
  u32 get_num_regs() const { return num_regs; }

  // never used member functions
  // 1. {num_regs, num_operands} are both set inside 
  // trace_warp_inst_t::parse_from_trace_struct(...)
  // 2. And "num_operands = num_regs", and hence 
  // loop upper-bound of "i < (cu->get_num_operands() - cu->get_num_regs())" 
  // is never met in opndcoll_rfu_t::dispatch_ready_cu()
  void set_num_regs(u32 num) { num_regs = num; }
  void set_num_operands(u32 num) { num_operands = num; }

  void set_bar_id(u32 id) { bar_id = id; }
  void set_bar_count(u32 count) { bar_count = count; }

  address_type pc;  // program counter address of instruction
  std::string trace_opcode; // trace.opcode (e.g., "LDG.E.SYS")
  u32 isize;  // size of instruction in bytes
  op_type op; // opcode (uarch visible)

  barrier_type bar_type;
  reduction_type red_type;
  u32 bar_id;
  u32 bar_count;

  types_of_operands oprnd_type;  // code (uarch visible) identify if the
                                 // operation is an interger or a floating point
  special_ops sp_op; // code (uarch visible) identify if int_alu, fp_alu, int_mul ....
  operation_pipeline op_pipe;  // code (uarch visible) identify the pipeline of
                               // the operation (SP, SFU or MEM)
  mem_operation mem_op;        // code (uarch visible) identify memory type
  bool const_cache_operand;    // has a load from constant memory as an operand
  _memory_op_t memory_op;      // memory_op used by ptxplus
  u32 num_operands;
  u32 num_regs;  // count vector operand as one register operand

  address_type reconvergence_pc;  // -1 => not a branch, -2 => use function
                                  // return address

  u32 out[8];
  u32 outcount;
  u32 in[24];
  u32 incount;
  unsigned char is_vectorin;
  unsigned char is_vectorout;
  int pred;  // predicate register number
  int ar1, ar2;
  // register number for bank conflict evaluation
  struct {
    int dst[MAX_REG_OPERANDS];
    int src[MAX_REG_OPERANDS];
  } arch_reg;
  // int arch_reg[MAX_REG_OPERANDS]; // register number for bank conflict
  // evaluation
  u32 latency;  // operation latency
  u32 initiation_interval;

  u32 data_size;  // what is the size of the word being operated on?
  memory_space_t space;
  cache_operator_type cache_op;

 protected:
  bool m_decoded;
  virtual void pre_decode() {}
};

enum divergence_support_t { POST_DOMINATOR = 1, NUM_SIMD_MODEL };

const u32 max_accesses_per_insn_per_tid = 8;

class warp_inst_t : public inst_t {
 public:
  friend class data_cache;
  // constructors
  warp_inst_t() {
    m_uid = 0;
    m_mem_req_uid = (u32) - 1;
    m_streamID = (u64) - 1;
    m_empty = true;
    m_config = NULL;

    // Ni:
    m_is_ldgsts = false;
    m_is_ldgdepbar = false;
    m_is_depbar = false;

    m_depbar_group_no = 0;
  }
  warp_inst_t(const core_config *config) {
    m_uid = 0;
    m_mem_req_uid = (u32) - 1;
    m_streamID = (u64) - 1;
    assert(config->warp_size <= MAX_WARP_SIZE);
    m_config = config;
    m_empty = true;
    m_isatomic = false;
    m_per_scalar_thread_valid = false;
    m_mem_accesses_created = false;
    m_cache_hit = false;
    m_is_printf = false;
    m_is_cdp = 0;
    should_do_atomic = true;

    // Ni:
    m_is_ldgsts = false;
    m_is_ldgdepbar = false;
    m_is_depbar = false;

    m_depbar_group_no = 0;
  }
  virtual ~warp_inst_t() {}

  // modifiers
  void broadcast_barrier_reduction(const active_mask_t &access_mask);
  void do_atomic(bool forceDo = false);
  void do_atomic(const active_mask_t &access_mask, bool forceDo = false);
  void clear() { m_empty = true; }

  void issue(const active_mask_t &mask, u32 warp_id,
             u64 cycle, int dynamic_warp_id, int sch_id,
             u64 streamID);

  const active_mask_t &get_active_mask() const { return m_warp_active_mask; }
  void completed(u64 cycle)
      const;  // stat collection: called when the instruction is completed

  void set_addr(u32 n, new_addr_type addr) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);
      m_per_scalar_thread_valid = true;
    }
    m_per_scalar_thread[n].memreqaddr[0] = addr;
  }
  void set_addr(u32 n, new_addr_type *addr, u32 num_addrs) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);
      m_per_scalar_thread_valid = true;
    }
    assert(num_addrs <= max_accesses_per_insn_per_tid);
    for (u32 i = 0; i < num_addrs; i++) {
      m_per_scalar_thread[n].memreqaddr[i] = addr[i];
    }      
  }
  void print_m_accessq() {
    if (accessq_empty())
      return;
    else {
      printf("Printing mem access generated\n");
      std::list<mem_access_t>::iterator it;
      for (it = m_accessq.begin(); it != m_accessq.end(); ++it) {
        printf("MEM_TXN_GEN:%s:%llx, Size:%d \n",
               mem_access_type_str(it->get_type()), it->get_addr(),
               it->get_size());
      }
    }
  }
  struct transaction_info {
    std::bitset<4> chunks;  // bitmask: 32-byte chunks accessed
    mem_access_byte_mask_t bytes;
    active_mask_t active;  // threads in this transaction

    bool test_bytes(u32 start_bit, u32 end_bit) {
      for (u32 i = start_bit; i <= end_bit; i++) {
        if (bytes.test(i)) {
          return true;
        }
      }
      return false;
    }
  };

  void parse_mem_access_type(
    const memory_space_t& space, const bool& is_write,
    mem_access_type& access_type);

  void generate_mem_accesses(u64 cycle = (u64) - 1);
  // "legacy" indicates arch before VOLTA
  void modify_sector_size_for_legacy_arch(bool& sector_segment_size, u32& segment_size);
  void memory_coalescing_arch(bool is_write, mem_access_type access_type);
  void memory_coalescing_arch_atomic(bool is_write,
                                     mem_access_type access_type);
  void memory_coalescing_arch_reduce_and_send(bool is_write,
                                              mem_access_type access_type,
                                              const transaction_info &info,
                                              new_addr_type addr,
                                              u32 segment_size);

  void add_callback(u32 lane_id,
                    void (*function)(const class inst_t *,
                                     class ptx_thread_info *),
                    const inst_t *inst, class ptx_thread_info *thread,
                    bool atomic) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);
      m_per_scalar_thread_valid = true;
      if (atomic) {
        m_isatomic = true;
      }
    }
    m_per_scalar_thread[lane_id].callback.function = function;
    m_per_scalar_thread[lane_id].callback.instruction = inst;
    m_per_scalar_thread[lane_id].callback.thread = thread;
  }
  void set_active(const active_mask_t &active);
  void clear_active(const active_mask_t &inactive);
  void set_not_active(u32 lane_id);

  // accessors
  virtual void print_insn(FILE *fp) const {
    fprintf(fp, " [inst @ pc=0x%04llx] ", pc);
    for (int i = (int)m_config->warp_size - 1; i >= 0; i--)
      fprintf(fp, "%c", ((m_warp_active_mask[i]) ? '1' : '0'));
  }
  bool active(u32 thread) const { return m_warp_active_mask.test(thread); }
  u32 active_count() const { return m_warp_active_mask.count(); }
  u32 issued_count() const {
    assert(m_empty == false);
    return m_warp_issued_mask.count();
  }  // for instruction counting
  bool empty() const { return m_empty; }
  u32 get_warp_id() const {
    assert(!m_empty);
    return m_warp_id;
  }
  u32 warp_id_func() const  // to be used in functional simulations only
  {
    return m_warp_id;
  }
  u32 dynamic_warp_id() const {
    assert(!m_empty);
    return m_dynamic_warp_id;
  }
  bool has_callback(u32 n) const {
    return m_warp_active_mask[n] && m_per_scalar_thread_valid &&
           (m_per_scalar_thread[n].callback.function != NULL);
  }
  new_addr_type get_addr(u32 n) const {
    assert(m_per_scalar_thread_valid);
    return m_per_scalar_thread[n].memreqaddr[0];
  }

  bool isatomic() const { return m_isatomic; }

  u32 warp_size() const { return m_config->warp_size; }

  bool accessq_empty() const { return m_accessq.empty(); }
  u32 accessq_count() const { return m_accessq.size(); }
  const mem_access_t &accessq_back() { return m_accessq.back(); }
  void accessq_pop_back() { m_accessq.pop_back(); }

  bool dispatch_delay() {
    if (cycles > 0) cycles--;
    return cycles > 0;
  }

  bool has_dispatch_delay() { return cycles > 0; }

  void print(FILE *fout) const;
  u32 get_uid() const { return m_uid; }
  u64 get_streamID() const { return m_streamID; }
  u32 get_schd_id() const { return m_scheduler_id; }
  active_mask_t get_warp_active_mask() const { return m_warp_active_mask; }

  const core_config* get_config() const {
    return m_config;
  }
  std::string get_inst_info(u32 core_id, u32 mem_req_uid = (u32) - 1) const;
  void set_mem_req_uid(u32 mem_req_uid);
  u32 get_mem_req_uid();

 protected:
  u32 m_uid;
  u32 m_mem_req_uid;
  u64 m_streamID;
  bool m_empty;
  bool m_cache_hit;
  u64 issue_cycle;
  u32 cycles;  // used for implementing initiation interval delay
  bool m_isatomic;
  bool should_do_atomic;
  bool m_is_printf;
  u32 m_warp_id;
  u32 m_dynamic_warp_id;
  const core_config *m_config;

  // dynamic active mask for timing model (after predication)  
  active_mask_t m_warp_active_mask;  
                                   
  // active mask at issue (prior to predication test)
  // -- for instruction counting
  active_mask_t m_warp_issued_mask;

  struct per_thread_info {
    per_thread_info() {
      for (u32 i = 0; i < max_accesses_per_insn_per_tid; i++) {
        memreqaddr[i] = 0;
      }        
    }
    dram_callback_t callback;
    // effective address, upto 8 different requests 
    // (to support 32B access in 8 chunks of 4B each)    
    new_addr_type memreqaddr[max_accesses_per_insn_per_tid];
  };
  bool m_per_scalar_thread_valid;
  std::vector<per_thread_info> m_per_scalar_thread;
  bool m_mem_accesses_created;
  std::list<mem_access_t> m_accessq;

  u32 m_scheduler_id;  // the scheduler that issues this inst

  // Jin: cdp support
 public:
  int m_is_cdp;

  // Ni: add boolean to indicate whether the instruction is ldgsts
  bool m_is_ldgsts;
  bool m_is_ldgdepbar;
  bool m_is_depbar;
  u32 m_depbar_group_no;

  std::list<mem_access_t> get_access_q() {
    return m_accessq;
  }
};

void move_warp(warp_inst_t *&dst, warp_inst_t *&src);

size_t get_kernel_code_size(class function_info *entry);
class checkpoint {
 public:
  checkpoint();
  ~checkpoint() { printf("clasfsfss destructed\n"); }

  void load_global_mem(class memory_space *temp_mem, char *f1name);
  void store_global_mem(class memory_space *mem, char *fname, char *format);
  u32 radnom;
};
/*
 * This abstract class used as a base for functional and performance and
 * simulation, it has basic functional simulation data structures and
 * procedures.
 */
class core_t {
 public:
  core_t(gpgpu_sim *gpu, kernel_info_t *kernel, u32 warp_size,
         u32 threads_per_shader)
      : m_gpu(gpu),
        m_kernel(kernel),
        m_simt_stack(NULL),
        m_thread(NULL),
        m_warp_size(warp_size) {
    m_warp_count = threads_per_shader / m_warp_size;
    // Handle the case where the number of threads is not a
    // multiple of the warp size
    if (threads_per_shader % m_warp_size != 0) {
      m_warp_count += 1;
    }
    assert(m_warp_count * m_warp_size > 0);
    m_thread = (ptx_thread_info **)calloc(m_warp_count * m_warp_size,
                                          sizeof(ptx_thread_info *));
    initilizeSIMTStack(m_warp_count, m_warp_size);

    for (u32 i = 0; i < MAX_CTA_PER_SHADER; i++) {
      for (u32 j = 0; j < MAX_BARRIERS_PER_CTA; j++) {
        reduction_storage[i][j] = 0;
      }
    }
  }
  virtual ~core_t() { free(m_thread); }
  virtual void warp_exit(u32 warp_id) = 0;
  virtual bool warp_waiting_at_barrier(u32 warp_id) const = 0;
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, u32 t,
                                             u32 tid) = 0;
  class gpgpu_sim *get_gpu() {
    return m_gpu;
  }
  void execute_warp_inst_t(warp_inst_t &inst, u32 warpId = (unsigned)-1);
  bool ptx_thread_done(u32 hw_thread_id) const;
  virtual void updateSIMTStack(u32 warpId, warp_inst_t *inst);
  void initilizeSIMTStack(u32 warp_count, u32 warps_size);
  void deleteSIMTStack();
  warp_inst_t getExecuteWarp(u32 warpId);
  void get_pdom_stack_top_info(u32 warpId, u32 *pc,
                               u32 *rpc) const;
  kernel_info_t *get_kernel_info() { return m_kernel; }
  class ptx_thread_info **get_thread_info() {
    return m_thread;
  }
  u32 get_warp_size() const { return m_warp_size; }
  void and_reduction(u32 ctaid, u32 barid, bool value) {
    reduction_storage[ctaid][barid] &= value;
  }
  void or_reduction(u32 ctaid, u32 barid, bool value) {
    reduction_storage[ctaid][barid] |= value;
  }
  void popc_reduction(u32 ctaid, u32 barid, bool value) {
    reduction_storage[ctaid][barid] += value;
  }
  u32 get_reduction_value(u32 ctaid, u32 barid) {
    return reduction_storage[ctaid][barid];
  }

 protected:
  class gpgpu_sim *m_gpu;
  kernel_info_t *m_kernel;
  simt_stack **m_simt_stack;  // pdom based reconvergence context for each warp
  class ptx_thread_info **m_thread;
  u32 m_warp_size;
  u32 m_warp_count;
  u32 reduction_storage[MAX_CTA_PER_SHADER][MAX_BARRIERS_PER_CTA];
};

// register that can hold multiple instructions.
class register_set {
 public:
  register_set(u32 num, const char *name) {
    for (u32 i = 0; i < num; i++) {
      regs.push_back(new warp_inst_t());
    }
    m_name = name;
  }
  u32 regs_size() { return regs.size(); }
  const char *get_name() { return m_name; }
  bool has_free() {
    for (u32 i = 0; i < regs.size(); i++) {
      if (regs[i]->empty()) {
        return true;
      }
    }
    return false;
  }
  bool has_free(bool sub_core_model, u32 reg_id) {
    // in subcore model, each sched has a one specific reg to use (based on
    // sched id)
    if (!sub_core_model) {
      return has_free();
    }
    if (reg_id >= regs.size()) {
      printf("%s has only %u regs, but reg_id:%u is queried\n", 
        m_name, (unsigned)regs.size(), reg_id);
    }

    assert(reg_id < regs.size());
    return regs[reg_id]->empty();
  }
  bool has_ready() {
    for (u32 i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        return true;
      }
    }
    return false;
  }
  bool has_ready(bool sub_core_model, u32 reg_id) {
    if (!sub_core_model) return has_ready();
    assert(reg_id < regs.size());
    return (not regs[reg_id]->empty());
  }

  u32 get_ready_reg_id() {
    // for sub core model we need to figure which reg_id has the ready warp
    // this function should only be called if has_ready() was true
    assert(has_ready());
    warp_inst_t **ready;
    ready = NULL;
    u32 reg_id = 0;
    for (u32 i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
          // ready is oldest
        } else {
          ready = &regs[i];
          reg_id = i;
        }
      }
    }
    return reg_id;
  }
  u32 get_schd_id(u32 reg_id) {
    assert(not regs[reg_id]->empty());
    return regs[reg_id]->get_schd_id();
  }
  void move_in(warp_inst_t *&src) {
    warp_inst_t **free = get_free();
    move_warp(*free, src);
  }
  // void copy_in( warp_inst_t* src ){
  //   src->copy_contents_to(*get_free());
  //}
  void move_in(bool sub_core_model, u32 reg_id, warp_inst_t *&src) {
    warp_inst_t **free;
    if (!sub_core_model) {
      free = get_free();
    } else {
      assert(reg_id < regs.size());
      free = get_free(sub_core_model, reg_id);
    }
    move_warp(*free, src);
  }

  void move_out_to(warp_inst_t *&dest) {
    warp_inst_t **ready = get_ready();
    move_warp(dest, *ready);
  }
  void move_out_to(bool sub_core_model, u32 reg_id, warp_inst_t *&dest) {
    if (!sub_core_model) {
      return move_out_to(dest);
    }
    warp_inst_t **ready = get_ready(sub_core_model, reg_id);
    assert(ready != NULL);
    move_warp(dest, *ready);
  }

  warp_inst_t **get_ready() {
    warp_inst_t **ready;
    ready = NULL;
    for (u32 i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
          // ready is oldest
        } else {
          ready = &regs[i];
        }
      }
    }
    return ready;
  }
  warp_inst_t **get_ready(bool sub_core_model, u32 reg_id) {
    if (!sub_core_model) return get_ready();
    warp_inst_t **ready;
    ready = NULL;
    if (reg_id >= regs.size()) {
      printf("%s requires reg_id:%u, which is however, exceeds total %u reserved\n", 
        m_name, reg_id, (unsigned)regs.size()); // 2/4
    }

    assert(reg_id < regs.size());
    if (not regs[reg_id]->empty()) {
      ready = &regs[reg_id];
    }
    return ready;
  }

  void print(FILE *fp) const {
    fprintf(fp, "%s : @%p\n", m_name, this);
    for (u32 i = 0; i < regs.size(); i++) {
      fprintf(fp, "     ");
      regs[i]->print(fp);
      fprintf(fp, "\n");
    }
  }

  warp_inst_t **get_free() {
    for (u32 i = 0; i < regs.size(); i++) {
      if (regs[i]->empty()) {
        return &regs[i];
      }
    }
    assert(0 && "No free registers found");
    return NULL;
  }

  warp_inst_t **get_free(bool sub_core_model, u32 reg_id) {
    // in subcore model, each sched has a one specific reg to use (based on
    // sched id)
    if (!sub_core_model) {
      return get_free();
    }

    assert(reg_id < regs.size());
    if (regs[reg_id]->empty()) {
      return &regs[reg_id];
    }
    assert(0 && "No free register found");
    return NULL;
  }

  u32 get_size() { return regs.size(); }

 private:
  std::vector<warp_inst_t *> regs;
  const char *m_name;
};

#endif  // #ifdef __cplusplus

#endif  // #ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
