// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington,
// Vijay Kandiah, Nikos Hardavellas, Mahmoud Khairy, Junrui Pan,
// Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
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

#include "gpu-cache.h"
#include <assert.h>
#include "gpu-sim.h"
#include "hashing.h"
#include "stat-tool.h"
#include "shader_trace.h"
#include "../../libcuda/gpgpu_context.h"

static inline std::pair<std::bitset<128>, std::bitset<128>> to_u128_pair(const std::bitset<256>& bs) {
  std::bitset<128> hi_bits;
  std::bitset<128> lo_bits;
  for (int i = 0; i < 128; ++i) {
    hi_bits[i] = bs[i + 128];
    lo_bits[i] = bs[i];
  }
  return {hi_bits, lo_bits}; // Attention: return high 128 bits in first
}
static inline std::pair<uint64_t,uint64_t> to_u64_pair(const std::bitset<128>& bs) {
  uint64_t lo = 0, hi = 0;
  for (int i = 0; i < 64; ++i) {
    if (bs.test(i))      lo |= (1ull << i);
    if (bs.test(i+64))   hi |= (1ull << (i));
  }
  return {hi, lo}; // 注意：返回时把高 64 位放在 first
}
void print_hex_128_for_bitset(const std::bitset<128>& bs) {
  std::pair<uint64_t,uint64_t> hilo = to_u64_pair(bs);
  uint64_t hi = hilo.first;
  uint64_t lo = hilo.second;
  std::cout << "0x"
            << std::hex << std::uppercase
            << std::setw(16) << std::setfill('0') << hi
            << std::setw(16) << std::setfill('0') << lo
            << std::dec << '\n';
}

// used to allocate memory that is large enough to adapt the changes in cache
// size across kernels
const char *cache_request_status_str(enum cache_request_status status) {
  static const char *static_cache_request_status_str[] = {
    "HIT",
    "VC_HIT",
    "HIT_RESERVED", 
    "MISS",
    "RESERVATION_FAIL",
    "SECTOR_MISS",
    "MSHR_HIT",
    "BYPASS_ACTIVATED",
    "BYPASS_DEACTIVATED",
};

  assert(sizeof(static_cache_request_status_str) / sizeof(const char *) ==
         NUM_CACHE_REQUEST_STATUS);
  assert(status < NUM_CACHE_REQUEST_STATUS);

  return static_cache_request_status_str[status];
}

const char *replacement_policy_str(enum replacement_policy_t rp) {
  static const char *staic_replacement_policy_str[] = {
    "LRU",
    "FIFO", 
    "SRRIP"
  };
  assert(sizeof(staic_replacement_policy_str) / sizeof(const char *) ==
         NUM_REPLACEMENT_POLICY);
  assert(rp < NUM_REPLACEMENT_POLICY);
  return staic_replacement_policy_str[rp];
}

const char *srrip_update_policy_str(enum srrip_update_policy_t up) {
  static const char *static_srrip_update_policy_str[] = {
    "HP",
    "FP"
  };
  assert(sizeof(static_srrip_update_policy_str) / sizeof(const char *) ==
         NUM_SRRIP_UPDATE_POLICY);
  assert(up < NUM_SRRIP_UPDATE_POLICY);
  return static_srrip_update_policy_str[up];
}

const char *mshr_config_t_str(enum mshr_config_t type) {
  static const char *static_mshr_config_t_str[] = {
    "TEX_FIFO",
    "ASSOC",
    "SECTOR_TEX_FIFO",
    "SECTOR_ASSOC"
  };

  assert(sizeof(static_mshr_config_t_str) / sizeof(const char *) ==
         NUM_MSHR_CONFIGS);
  assert(type < NUM_MSHR_CONFIGS);

  return static_mshr_config_t_str[type];
}

//
const char *mf_request_type_str(enum mf_type type) {
  static const char *static_mf_request_type_str[] = {
    "READ_REQUEST",
    "WRITE_REQUEST",
    "READ_REPLY",
    "WRITE_ACK"
};

  assert(sizeof(static_mf_request_type_str) / sizeof(const char *) ==
         NUM_MF_TYPE);
  assert(type < NUM_MF_TYPE);

  return static_mf_request_type_str[type];
}

const char* cache_block_state_str(enum cache_block_state state) {
  static const char *static_cache_block_state_str[] = {
      "INVALID", "RESERVED", "VALID", "MODIFIED"};

  assert(sizeof(static_cache_block_state_str) / sizeof(const char *) ==
         NUM_CACHE_BLOCK_STATES);
  assert(state < NUM_CACHE_BLOCK_STATES);

  return static_cache_block_state_str[state];
}

const char* write_policy_str(enum write_policy_t wp) {
  static const char* static_write_policy_str[] = {
      "READ_ONLY", "WRITE_BACK", "WRITE_THROUGH", "WRITE_EVICT",
      "LOCAL_WB_GLOBAL_WE"};

  assert(sizeof(static_write_policy_str) / sizeof(const char*) ==
         NUM_WRITE_POLICIES);
  assert(wp < NUM_WRITE_POLICIES);

  return static_write_policy_str[wp];
}

const char* write_allocate_policy_str(enum write_allocate_policy_t wap) {
  static const char* static_write_allocate_policy_str[] = {
      "NO_WRITE_ALLOCATE", "WRITE_ALLOCATE", "FETCH_ON_WRITE",
      "LAZY_FETCH_ON_READ"};

  assert(sizeof(static_write_allocate_policy_str) / sizeof(const char*) ==
         NUM_WRITE_ALLOCATE_POLICIES);
  assert(wap < NUM_WRITE_ALLOCATE_POLICIES);

  return static_write_allocate_policy_str[wap];
}

const char *cache_fail_status_str(enum cache_reservation_fail_reason status) {
  static const char *static_cache_reservation_fail_reason_str[] = {
    "LINE_ALLOC_FAIL",
    "MSHR_ENTRY_FAIL",
    "MISS_QUEUE_FULL",
    "MSHR_MERGE_ENTRY_FAIL",
    "MSHR_RW_PENDING"
    };
  
  assert(sizeof(static_cache_reservation_fail_reason_str) /
             sizeof(const char *) ==
         NUM_CACHE_RESERVATION_FAIL_STATUS);
  assert(status < NUM_CACHE_RESERVATION_FAIL_STATUS);

  return static_cache_reservation_fail_reason_str[status];
}

const char *line_alloc_fail_driver_str(enum line_alloc_fail_driver driver) {
  static const char *static_line_alloc_fail_driver_str[] = {
    "LINE_ALLOC_FAIL__RD_ONLY_MISS",
    "LINE_ALLOC_FAIL__RD_PROBE_MISS",
    "LINE_ALLOC_FAIL__WR_PROBE_MISS" };    

  assert(sizeof(static_line_alloc_fail_driver_str) / sizeof(const char *) ==
         NUM_LINE_ALLOC_FAIL_DRIVER);
  assert(driver < NUM_LINE_ALLOC_FAIL_DRIVER);

  return static_line_alloc_fail_driver_str[driver];
}

const char *mshr_entry_fail_driver_str(enum mshr_entry_fail_driver driver) {
  static const char *static_mshr_entry_fail_driver_str[] = {
  "MSHR_ENTRY_FAIL__RD_MISS",
  "MSHR_ENTRY_FAIL__WR_ALLOC_MISS",
  "MSHR_ENTRY_FAIL__WR_ALLOC_MISS_FETCH_ON_WR"};

  assert(sizeof(static_mshr_entry_fail_driver_str) / sizeof(const char *) ==
         NUM_MSHR_ENTRY_FAIL_DRIVER);
  assert(driver < NUM_MSHR_ENTRY_FAIL_DRIVER);

  return static_mshr_entry_fail_driver_str[driver];
}

const char *miss_queue_full_driver_str(enum miss_queue_full_driver driver) {
  static const char *static_mshr_queue_full_driver_str[] = {
    "WR_THROUGH_HIT",
    "WR_EVICT_HIT",
    "WR_ALLOC_MISS",
    "WR_ALLOC_MISS_FETCH_ON_WR_WHOLE_LINE",
    "WR_ALLOC_MISS_FETCH_ON_WR_PARTIAL_LINE",
    "WR_ALLOC_MISS_LAZY_FETCH_ON_RD",
    "WR_MISS_NO_WR_ALLOC",
    "RD_MISS",
    "RD_ONLY_MISS"};

  assert(sizeof(static_mshr_queue_full_driver_str) / sizeof(const char *) ==
         NUM_MISS_QUEUE_FULL_DRIVER);
  assert(driver < NUM_MISS_QUEUE_FULL_DRIVER);

  return static_mshr_queue_full_driver_str[driver];
}

const char *mshr_merge_entry_fail_driver_str(enum mshr_merge_entry_fail_driver driver) {
  static const char *static_mshr_merge_entry_fail_driver_str[] = {
    "RD_MISS",
    "WR_ALLOC_MISS",
    "WR_ALLOC_MISS_FETCH_ON_WR_PARTIAL_LINE"};

  assert(sizeof(static_mshr_merge_entry_fail_driver_str) / sizeof(const char *) ==
         NUM_MSHR_MERGE_ENTRY_FAIL_DRIVER);
  assert(driver < NUM_MSHR_MERGE_ENTRY_FAIL_DRIVER);

  return static_mshr_merge_entry_fail_driver_str[driver];
}

unsigned l1d_cache_config::set_bank(new_addr_type addr) const {
  // For sector cache, we select one sector per bank (sector interleaving)
  // This is what was found in Volta (one sector per bank, sector interleaving)
  // otherwise, line interleaving
  return cache_config::hash_function(addr, l1_banks,
                                     l1_banks_byte_interleaving_log2,
                                     l1_banks_log2, l1_banks_hashing_function);
}

void l1d_cache_config::extra_config(
  char* bypass_config, char* victim_cache_config) {
  // <victim_cache_enable>
  [[maybe_unused]] int ntok_victim_cache = 
    sscanf(victim_cache_config, "%c,%u", &m_victim_cache_enable, &m_victim_cache_entries);
  fprintf(Trace::out, "----------- L1D victim_cache config is below -----------\n"
    "m_victim_cache_enable = %c m_victim_cache_entries = %u\n",
    m_victim_cache_enable, m_victim_cache_entries);

  // <bypass_enable>,<bypass_mode>,<infinite_bypasses>,<total_evictions_aware>,
  // <max_bypasses>,<max_evictions_bound>,<trash_conf_cnt_bound>
  [[maybe_unused]] int ntok_bypass = 
    sscanf(bypass_config, "%c,%u,%c,%c,%u,%u,%u", 
      &m_bypass_enable, &m_bypass_mode, &m_infinite_bypasses, &m_total_evictions_aware,
      &m_max_bypasses, &m_max_evictions_bound, &m_trash_conf_cnt_bound);
  fprintf(Trace::out, 
    "----------- %s bypass_config is below -----------\n "
    "m_bypass_enable = %c m_bypass_mode = %u m_infinite_bypasses = %c m_total_evictions_aware = %c "
    "m_max_bypasses = %u m_max_evictions_bound = %u m_trash_conf_cnt_bound = %u\n",
    m_cache_name, 
    m_bypass_enable, m_bypass_mode, m_infinite_bypasses, m_total_evictions_aware, 
    m_max_bypasses, m_max_evictions_bound, m_trash_conf_cnt_bound);
}

unsigned l1d_cache_config::get_max_cache_multiplier() const {
  // set * assoc * cacheline size. Then convert Byte to KB
  // gpgpu_unified_cache_size is in KB while original_sz is in B
  if (m_unified_cache_size > 0) {
    unsigned original_size = m_nset * original_m_assoc * m_line_sz / 1024;
    assert(m_unified_cache_size % original_size == 0);
    return m_unified_cache_size / original_size;
  } else {
    return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER;
  }
}

unsigned cache_config::recalc_orig_addr(new_addr_type tag, unsigned set_index) const {
  new_addr_type orig_addr = 
    (tag << (m_line_sz_log2 + m_nset_log2)) | (set_index << m_line_sz_log2);
  return orig_addr;
}

unsigned cache_config::set_index(new_addr_type addr, unsigned warp_id) const {
  return cache_config::hash_function(addr, m_nset, m_line_sz_log2, m_nset_log2,
                                     m_set_index_function, warp_id);
}

std::pair<unsigned, unsigned> 
cache_config::set_index_pairs(new_addr_type addr, unsigned warp_id) const {
  unsigned ln_set_index = cache_config::hash_function(
    addr, m_nset, m_line_sz_log2, m_nset_log2, LINEAR_SET_FUNCTION);

  unsigned hashed_set_index = cache_config::hash_function(
    addr, m_nset, m_line_sz_log2, m_nset_log2, m_set_index_function, warp_id);
  
  return std::pair<unsigned, unsigned>(ln_set_index, hashed_set_index);
}

unsigned cache_config::hash_function(new_addr_type addr, unsigned m_nset,
                                     unsigned m_line_sz_log2,
                                     unsigned m_nset_log2,
                                     unsigned m_index_function,
                                     unsigned warp_id) const {
  unsigned set_index = 0;

  switch (m_index_function) {
    case FERMI_HASH_SET_FUNCTION: {
      /*
       * Set Indexing function from "A Detailed GPU Cache Model Based on Reuse
       * Distance Theory" Cedric Nugteren et al. HPCA 2014
       */
      unsigned lower_xor = 0;
      unsigned upper_xor = 0;

      if (m_nset == 32 || m_nset == 64) {
        // Lower xor value is bits 7-11
        lower_xor = (addr >> m_line_sz_log2) & 0x1F;

        // Upper xor value is bits 13, 14, 15, 17, and 19
        upper_xor = (addr & 0xE000) >> 13;    // Bits 13, 14, 15
        upper_xor |= (addr & 0x20000) >> 14;  // Bit 17
        upper_xor |= (addr & 0x80000) >> 15;  // Bit 19

        set_index = (lower_xor ^ upper_xor);

        // 48KB cache prepends the set_index with bit 12
        if (m_nset == 64) set_index |= (addr & 0x1000) >> 7;

      } else { /* Else incorrect number of sets for the hashing function */
        assert(
            "\nGPGPU-Sim cache configuration error: The number of sets should "
            "be "
            "32 or 64 for the hashing set index function.\n" &&
            0);
      }
      break;
    }
    case BITWISE_XORING_FUNCTION: {
      new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2);
      unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1);
      set_index = bitwise_hash_function(higher_bits, index, m_nset);
      break;
    }
    case HASH_IPOLY_FUNCTION: {
      new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2);
      unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1);
      set_index = ipoly_hash_function(higher_bits, index, m_nset);
      break;
    }
    case CUSTOM_SET_FUNCTION: {
      /* No custom set function implemented */
      break;
    }
    case LINEAR_SET_FUNCTION: {
      set_index = (addr >> m_line_sz_log2) & (m_nset - 1);
      break;
    }
    case WARP_CORR_FUNCTION: {
      const unsigned b5 = (warp_id >> 5) & 0x1;
      const unsigned b4 = (warp_id >> 4) & 0x1;
      const unsigned b3 = (warp_id >> 3) & 0x1;
      const unsigned b2 = (warp_id >> 2) & 0x1;
      const unsigned b1 = (warp_id >> 1) & 0x1;
      const unsigned b0 = (warp_id >> 0) & 0x1;
      unsigned r3 = b5 ^ b3;
      unsigned r2 = b4 ^ b2;
      unsigned r1 = b1;
      unsigned r0 = b0;
      unsigned xor_warp_id = ((r3 << 3) | (r2 << 2) | (r1 << 1) | r0) & (m_nset - 1);
      unsigned ln_set_idx  = (addr >> m_line_sz_log2) & (m_nset - 1);
      set_index = (ln_set_idx ^ xor_warp_id) & (m_nset - 1);
      break;
    }
    default: {
      assert("\nUndefined set index function.\n" && 0);
      break;
    }
  }

  // Linear function selected or custom set index function not implemented
  assert((set_index < m_nset) &&
         "\nError: Set index out of bounds. This is caused by "
         "an incorrect or unimplemented custom set index function.\n");

  return set_index;
}

void l2_cache_config::init(linear_to_raw_address_translation *address_mapping) {
  cache_config::init(
    m_config_string, m_mshr_config_string, m_rrpv_config_string, m_rep_enhance_string,
    FuncCachePreferNone, "L2");
  m_address_mapping = address_mapping;
}

unsigned l2_cache_config::set_index(new_addr_type addr, unsigned warp_id) const {
  new_addr_type part_addr = addr;

  if (m_address_mapping) {
    // Calculate set index without memory partition bits to reduce set camping
    part_addr = m_address_mapping->partition_address(addr);
  }

  return cache_config::set_index(part_addr, warp_id);
}

tag_array::~tag_array() {
  unsigned cache_lines_num = m_config.get_max_num_lines();
  for (unsigned i = 0; i < cache_lines_num; ++i) delete m_lines[i];
  delete[] m_lines;
}

tag_array::tag_array(cache_config &config, int core_id, int type_id,
                     cache_block_t **new_lines)
    : m_config(config), m_lines(new_lines) {
  init(core_id, type_id);
}

void tag_array::update_cache_parameters(cache_config &config) {
  m_config = config;
}

tag_array::tag_array(gpgpu_sim *gpu, cache_config &config, int core_id, int type_id)
    : m_gpu(gpu), m_config(config) {
  
  m_l1d_max_evicts.resize(gpu->m_shader_config->num_shader(), 0);
  m_l1d_avg_evicts.resize(gpu->m_shader_config->num_shader(), 0);  
  m_l1d_rd_byp_activated_times.clear();
  m_l1d_rd_byp_deactivated_times.clear();
  m_l1d_rd_fill_time.clear();
  m_l1d_evict_time.clear();
  m_l1d_rd_fill_to_evict_gap.clear();
  m_l1d_evictions.clear();
  m_l1d_occupied.clear();
  m_l1d_rd_bypass_confidence.clear();
  m_l1d_rd_bypass_activated.clear();
  m_avg_l1d_rd_fill_to_evict_gap.clear();
  m_l1d_fill_to_evict_lines.clear();
  m_l1d_trashed_lines.clear();  
  m_l1d_mpki = 0.0f;
  // m_l1d_evictions_bound = 5; // 124.722 (+3.535%) drops compared with 10 below
  // m_l1d_evictions_bound = 8;
  // m_l1d_evictions_bound = 3;
  // m_l1d_evictions_bound = 9;
  // m_l1d_evictions_bound = 10; // dead-lock under "lrr" warp-sched scheme but 2nd-highest IPC under "gto"
  // m_l1d_evictions_bound = 13; // worse
  // m_l1d_evictions_bound = 20;
  // m_trash_conf_cnt_bound = 2; // Drops
  // m_trash_conf_cnt_bound = 3; // Highest
  // m_trash_conf_cnt_bound = 4;
  // m_trash_conf_cnt_bound = 10; // ok (confirmed again with only inc/dec conf cnt inside tag_array::probe)
  // m_trash_conf_cnt_bound = 5; // ok (activate: 2; deactivate: 4)
  // m_trash_conf_cnt_bound = 7; // ok (activate: 2; deactivate: 4)

  m_reref_gap.resize(gpu->m_shader_config->num_shader());
  m_avg_reref_gap.resize(gpu->m_shader_config->num_shader(), 0);
  for (unsigned i = 0; i < m_avg_reref_gap.size(); i++)
  {
    m_reref_gap[i].clear();
  }
  unsigned cache_lines_num = config.get_max_num_lines();
  m_lines = new cache_block_t *[cache_lines_num];
  if (config.m_cache_type == NORMAL) {
    for (unsigned i = 0; i < cache_lines_num; ++i) {
      m_lines[i] = new line_cache_block();
      unsigned max_rrpv_val = (1 << m_config.m_rrpv_bits) - 1;
      m_lines[i]->set_rrpv(max_rrpv_val); // for SRRIP
      m_lines[i]->set_max_rrpv(max_rrpv_val); // for SRRIP
    }      
  } else if (config.m_cache_type == SECTOR) {
    for (unsigned i = 0; i < cache_lines_num; ++i) {
      m_lines[i] = new sector_cache_block();
      unsigned max_rrpv_val = (1 << m_config.m_rrpv_bits) - 1;
      m_lines[i]->set_rrpv(max_rrpv_val); // for SRRIP
      m_lines[i]->set_max_rrpv(max_rrpv_val); // for SRRIP
    }      
  } else {
    assert(0);
  }    

  init(core_id, type_id);

  m_trashed_reqs.clear();
  m_incoming_bypasses.clear();
  m_victim_bypasses.clear();
}

void tag_array::init(int core_id, int type_id) {  
  m_access = 0;
  m_miss = 0;
  m_pending_hit = 0;
  m_res_fail = 0;
  m_sector_miss = 0;
  // initialize snapshot counters for visualizer
  m_prev_snapshot_access = 0;
  m_prev_snapshot_miss = 0;
  m_prev_snapshot_pending_hit = 0;
  m_core_id = core_id;
  m_type_id = type_id;
  is_used = false;
  m_dirty = 0;
  m_total_records_in_mshr = 0;
  // m_mshr_recorded_block_addresses.clear();
}

void tag_array::add_pending_line(mem_fetch *mf) {
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i == pending_lines.end()) {
    pending_lines[addr] = mf->get_inst().get_uid();
  }
}

void tag_array::remove_pending_line(mem_fetch *mf) {
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i != pending_lines.end()) {
    pending_lines.erase(addr);
  }
}

enum cache_request_status tag_array::probe(
  const std::string& caller,
  baseline_cache* cache,
  new_addr_type raw_addr,
  new_addr_type addr /* block_addr */, unsigned &idx,
  mem_fetch *mf, bool is_write,
  u64 time,
  bool& inter_warp_has_interference, 
  WARP_INTERFERE_RECORD& inter_warp_interfere_record,
  bool probe_mode) {

  mem_access_sector_mask_t mask = mf->get_access_sector_mask();
  std::string final_caller = caller + "-> tag_array::probe";
  
  if (DTRACE(PROBE_L2_TAG)) {
    if (m_is_l2 && mf) {
      fprintf(Trace::out, "%llu caller:%s tag_array::probe(4th in-arg *mf) "
        "probed L2 for mf [warp:%u][sid:%u][addr:%#llx]\n",
        time, caller.c_str(), mf->get_wid(), mf->get_sid(), mf->get_addr());
    } else if (m_is_l2) {
      // Never met
      fprintf(Trace::out, "%llu caller:%s tag_array::probe(4th in-arg *mf) "
        "probed L2, but !mf\n", time, caller.c_str());      
    }
  }

  return probe(
    final_caller.c_str(), cache, raw_addr, addr, idx, mask, is_write, time, 
    probe_mode, inter_warp_has_interference, inter_warp_interfere_record, mf);
}

void tag_array::gather_rep_candidates(
  u64 time,
  mem_fetch *mf, cache_block_t* line, const unsigned& set_index, const unsigned& index,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates) {

  assert(line->is_valid_line());
  if (mf && m_is_l1d) {
    m_gpu->get_shader_stats()->m_l1d_repl_cands[mf->get_sid()]++; 
  }

  LINE_RECENCY recency(
    line->is_valid_line(),
    line->was_recorded_in_mshr(),
    line->get_reref_gap(),
    line->get_last_access_time(), 
    line->get_last_fill_time(),
    line->get_last_evict_time(),
    line->get_evict_gap(),
    line->get_rrpv(),
    line->get_max_rrpv(),
    line->get_total_hits(),
    line->get_total_evictions(), 
    line->get_total_accesses(),
    line->get_last_warp_id(),
    line->get_last_core_id()
  );  
  hybrid_rep_candidates.push_back(std::pair<unsigned, LINE_RECENCY>(index, recency));

  if (m_config.m_mshr_corr_repl == 'T') {
    assert(m_config.m_mshr_disable == 'F');
    if (DTRACE(MSHR_AWARED_REPL)) {
      fprintf(Trace::out, "%llu %s lines[index:%u] was_recorded_in_mshr = %u\n",
        time, m_config.get_cache_name(), index, line->was_recorded_in_mshr());
    }
    if (!line->was_recorded_in_mshr()) {
      hybrid_rep_candidates_no_record_in_mshr.push_back(std::pair<unsigned, LINE_RECENCY>(index, recency));
    } else {
      hybrid_rep_candidates_recorded_in_mshr.push_back(std::pair<unsigned, LINE_RECENCY>(index, recency));
    }
  }
}

void tag_array::lru_pick(
  cache_block_t* line, u64& valid_timestamp, 
  unsigned& valid_line, const unsigned& index, 
  unsigned& warp_id, unsigned& core_id,
  bool& lru_has_picked) {

  if (line->get_last_access_time() < valid_timestamp) {
    valid_timestamp = line->get_last_access_time();
    valid_line      = index;
    warp_id         = line->get_last_warp_id();
    core_id         = line->get_last_core_id();
    lru_has_picked  = true;
  }
}
void tag_array::fill_time_pick(
  cache_block_t* line, u64& valid_timestamp, 
  unsigned& valid_line, const unsigned& index) {

  if (line->get_last_fill_time() < valid_timestamp) {
    valid_timestamp = line->get_last_fill_time();
    valid_line      = index;
  }
}

void tag_array::warp_interfere_aware_pick(
  u64 time,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  bool& has_interfered,
  unsigned& valid_line /* victim index */,
  unsigned& warp_id, unsigned& core_id,
  mem_fetch *mf
) {
  [[maybe_unused]] bool orig_has_interfered = has_interfered;

  std::vector<std::pair<unsigned, LINE_RECENCY>> hybrid_rep_candidates_in_use;
  if (m_config.m_mshr_corr_repl == 'T') {
    // for debug 2/27
    if (m_is_l2) {
      assert(0);
    }

    if (hybrid_rep_candidates_no_record_in_mshr.size()) {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_no_record_in_mshr;
    } else {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_recorded_in_mshr;
    }
  } else {
    hybrid_rep_candidates_in_use = hybrid_rep_candidates;
  }
  assert(hybrid_rep_candidates_in_use.size());

  assert(mf); // not sure if mf != nullptr
  std::vector<std::pair<unsigned, LINE_RECENCY>> target_warp_candidates;
  for (unsigned i = 0; i < hybrid_rep_candidates_in_use.size(); i++)
  {
    if (hybrid_rep_candidates_in_use[i].second.warp_id == mf->get_wid()) {
      target_warp_candidates.push_back(hybrid_rep_candidates_in_use[i]);
    }
  }
  if (!target_warp_candidates.empty()) {
    std::sort(target_warp_candidates.begin(), target_warp_candidates.end(), cmpForSmallerTimestamp);
    auto& lru_pos = target_warp_candidates[0];
    if (m_config.m_fill_time_ascend == 'T') {
      std::sort(target_warp_candidates.begin(), target_warp_candidates.end(), cmpForSmallerFillTime);
      if (target_warp_candidates[0].second.last_fill_time == lru_pos.second.last_access_time) {
        valid_line = target_warp_candidates[0].first;
        warp_id    = target_warp_candidates[0].second.warp_id;
        core_id    = target_warp_candidates[0].second.core_id;
      } else {
        valid_line = lru_pos.first;
        warp_id    = lru_pos.second.warp_id;
        core_id    = lru_pos.second.core_id;
      }
    } else {
      valid_line = lru_pos.first;
      warp_id    = lru_pos.second.warp_id;
      core_id    = lru_pos.second.core_id;
    }
  } else {
    std::sort(hybrid_rep_candidates_in_use.begin(), hybrid_rep_candidates_in_use.end(), cmpForSmallerTimestamp);
    auto& lru_pos = hybrid_rep_candidates_in_use[0];
    valid_line = lru_pos.first;
    warp_id    = lru_pos.second.warp_id;
    core_id    = lru_pos.second.core_id;
    if (warp_id == ((unsigned) - 1)) { // possible, i.e., mf has not bee filled
      has_interfered = false;
    } else {
      has_interfered = true;
    }
  }

  if (valid_line == 189) {
    if (DTRACE(WARP_CACHE_INTERFERE)) {
      for (unsigned i = 0; i < hybrid_rep_candidates_in_use.size(); i++)
      {
        fprintf(Trace::out, "hybrid_rep_candidates_in_use[%u] = "
          "{idx:%u is_line_invalid:%u warp_id:%u core_id:%u last_access_time:%llu}\n", 
          i, hybrid_rep_candidates_in_use[i].first, 
          hybrid_rep_candidates_in_use[i].second.is_valid,
          hybrid_rep_candidates_in_use[i].second.warp_id, 
          hybrid_rep_candidates_in_use[i].second.core_id, 
          hybrid_rep_candidates_in_use[i].second.last_access_time);
      }
    }
  }
  // printf("valid_line = %u warp_id = %u core_id = %u. has_interfered = %u->%u\n", 
  //   valid_line, warp_id, core_id, orig_has_interfered, has_interfered);
}

void tag_array::pick_with_lru(
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  unsigned& valid_line,
  unsigned& warp_id, unsigned& core_id,
  u64& smallest_access_time,
  unsigned& lru_picked_total_hits,
  u64& lru_picked_avg_evict_interval) {

  if (m_config.m_mshr_corr_repl == 'T') {
    std::vector<std::pair<unsigned, LINE_RECENCY>> hybrid_rep_candidates_in_use;
    if (hybrid_rep_candidates_no_record_in_mshr.size()) {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_no_record_in_mshr;
    } else {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_recorded_in_mshr;
    }
    assert(hybrid_rep_candidates_in_use.size());
    std::sort(hybrid_rep_candidates_in_use.begin(), hybrid_rep_candidates_in_use.end(), cmpForSmallerTimestamp);
    valid_line                    = hybrid_rep_candidates_in_use[0].first; // update valid_line 
    smallest_access_time          = hybrid_rep_candidates_in_use[0].second.last_access_time;
    lru_picked_total_hits         = hybrid_rep_candidates_in_use[0].second.total_hits;
    warp_id                       = hybrid_rep_candidates_in_use[0].second.warp_id;
    core_id                       = hybrid_rep_candidates_in_use[0].second.core_id;
  } else {
    assert(hybrid_rep_candidates.size());

    std::sort(hybrid_rep_candidates.begin(), hybrid_rep_candidates.end(), cmpForSmallerTimestamp);
    valid_line                    = hybrid_rep_candidates[0].first; // update valid_line 
    smallest_access_time          = hybrid_rep_candidates[0].second.last_access_time;
    lru_picked_total_hits         = hybrid_rep_candidates[0].second.total_hits;
    warp_id                       = hybrid_rep_candidates[0].second.warp_id;
    core_id                       = hybrid_rep_candidates[0].second.core_id;
 
    // srad_v2 with mshr_en_but_no_aware_all_lru.config would appear below:
    // valid_line:309 != hybrid_rep_candidates[0].first:321
    // hybrid_rep_candidates[0] = {idx:321 timestamp:15367}
    // hybrid_rep_candidates[1] = {idx:309 timestamp:15367}  
    // lru_pick() gen valid_line:309
    assert(hybrid_rep_candidates.size());
    if (hybrid_rep_candidates.size() > 1) {
      if (hybrid_rep_candidates[0].second.last_access_time != 
          hybrid_rep_candidates[1].second.last_access_time) {
        assert(valid_line == hybrid_rep_candidates[0].first);
      }
    } else if (hybrid_rep_candidates.size() == 1) {
      assert(valid_line == hybrid_rep_candidates[0].first);
    }
  }
}

void tag_array::pick_modified_by_total_hits_ascend(
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  unsigned& valid_line, const unsigned& lru_picked_total_hits) {

  if (m_config.m_mshr_corr_repl == 'T') {
    std::vector<std::pair<unsigned, LINE_RECENCY>> hybrid_rep_candidates_in_use;
    if (hybrid_rep_candidates_no_record_in_mshr.size()) {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_no_record_in_mshr;
    } else {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_recorded_in_mshr;
    }
    
    std::sort(hybrid_rep_candidates_in_use.begin(), hybrid_rep_candidates_in_use.end(), cmpForSmallerTotalHits);
    if (hybrid_rep_candidates_in_use[0].second.total_hits < lru_picked_total_hits) {
      valid_line = hybrid_rep_candidates_in_use[0].first;
    }
  } else {
    std::sort(hybrid_rep_candidates.begin(), hybrid_rep_candidates.end(), cmpForSmallerTotalHits);
    if (hybrid_rep_candidates[0].second.total_hits < lru_picked_total_hits) {
      valid_line = hybrid_rep_candidates[0].first;
    }
  }
}

void tag_array::fill_time_awared_modification_for_srrip(
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  unsigned& valid_line) {

  if (m_config.m_mshr_corr_repl == 'T') {
    std::vector<std::pair<unsigned, LINE_RECENCY>> hybrid_rep_candidates_in_use;
    if (hybrid_rep_candidates_no_record_in_mshr.size()) {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_no_record_in_mshr;
    } else {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_recorded_in_mshr;
    }    
    std::sort(hybrid_rep_candidates_in_use.begin(), hybrid_rep_candidates_in_use.end(), cmpForSmallerFillTime);
    if (hybrid_rep_candidates_in_use[0].second.rrpv == hybrid_rep_candidates_in_use[0].second.max_rrpv) {
      valid_line = hybrid_rep_candidates_in_use[0].first;
    }
  } else {
    std::sort(hybrid_rep_candidates.begin(), hybrid_rep_candidates.end(), cmpForSmallerFillTime);
    if (hybrid_rep_candidates[0].second.rrpv == hybrid_rep_candidates[0].second.max_rrpv) {
      valid_line = hybrid_rep_candidates[0].first;
    }
  }
}

void tag_array::fill_time_awared_modification_for_lru(
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  unsigned& valid_line,
  const u64& smallest_last_access_time) {

  if (m_config.m_mshr_corr_repl == 'T') {
    std::vector<std::pair<unsigned, LINE_RECENCY>> hybrid_rep_candidates_in_use;
    if (hybrid_rep_candidates_no_record_in_mshr.size()) {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_no_record_in_mshr;
    } else {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_recorded_in_mshr;
    }    
    std::sort(hybrid_rep_candidates_in_use.begin(), hybrid_rep_candidates_in_use.end(), cmpForSmallerFillTime);
    if (hybrid_rep_candidates_in_use[0].second.last_access_time <= smallest_last_access_time) {
      valid_line = hybrid_rep_candidates_in_use[0].first;
    }
  } else {
    std::sort(hybrid_rep_candidates.begin(), hybrid_rep_candidates.end(), cmpForSmallerFillTime);
    if (hybrid_rep_candidates[0].second.last_access_time <= smallest_last_access_time) {
      valid_line = hybrid_rep_candidates[0].first;
    }
  }
}

void tag_array::mshr_awared_modification_for_srrip(
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  unsigned& valid_line) {
  
  if (m_config.m_mshr_corr_repl == 'T') {
    std::vector<std::pair<unsigned, LINE_RECENCY>> hybrid_rep_candidates_in_use;
    if (hybrid_rep_candidates_no_record_in_mshr.size()) {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_no_record_in_mshr;
    } else {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_recorded_in_mshr;
    } 
    for (size_t i = 0; i < hybrid_rep_candidates_in_use.size(); i++)
    {      
      if (hybrid_rep_candidates_in_use[i].second.rrpv == hybrid_rep_candidates_in_use[i].second.max_rrpv) {
        valid_line = hybrid_rep_candidates_in_use[i].first;
        break;
      }
    }    
  }
}

void tag_array::inc_conf_cnt(int& conf, const int upper_bound, const int step) {
  conf += step;
  conf = (conf >= upper_bound) ? upper_bound : conf;
}
void tag_array::dec_conf_cnt(int& conf, const int lower_bound, const int step) {
  conf -= step;
  conf = (conf < lower_bound) ? lower_bound : conf;
}

enum cache_request_status tag_array::probe(
  const std::string& caller,
  baseline_cache* cache,
  new_addr_type raw_addr,
  new_addr_type addr /* block_addr*/, unsigned &idx,
  mem_access_sector_mask_t mask,
  bool is_write, 
  u64 time,
  bool probe_mode,
  bool& inter_warp_has_interference, 
  WARP_INTERFERE_RECORD& inter_warp_interfere_record,
  mem_fetch *mf) {

  if (DTRACE(PROBE_L2_TAG)) {
    if (m_is_l2 && mf) {
      fprintf(Trace::out, "%llu caller:%s tag_array::probe(4th in-arg mask) "
        "probed L2 for mf [warp:%u][sid:%u][addr:%#llx]\n",
        time, caller.c_str(), mf->get_wid(), mf->get_sid(), mf->get_addr());
    } else if (m_is_l2) {
      fprintf(Trace::out, "%llu caller:%s tag_array::probe(4th in-arg mask) "
        "probed L2, but !mf\n", time, caller.c_str());
    }
  }

  const shader_core_config *shader_cfg = m_gpu->getShaderCoreConfig();
  struct cache_sub_stats total_css;
  struct cache_sub_stats css;
  for (unsigned i = 0; i < shader_cfg->n_simt_clusters; i++) {    
    m_gpu->m_cluster[i]->get_L1D_sub_stats(css);
    total_css += css;

    if (DTRACE(GATHER_L1D_STATS_AFTER_PROBE)) {
      fprintf(Trace::out, "%llu GATHER_L1D_STATS_AFTER_PROBE "
        "kernel:%u total_css.rd_misses:%llu css.rd_misses:%llu\n", 
        time, m_gpu->m_kernel_id, total_css.rd_misses, css.rd_misses
      );
    }    
  }

  u64 tot_insns = m_gpu->gpu_tot_sim_insn + m_gpu->gpu_sim_insn;
  [[maybe_unused]] float last_l1d_mpki = m_l1d_mpki;
  m_l1d_mpki = 1000 * (total_css.misses / (float)tot_insns);
  
  if (DTRACE(TAG_PROBE)) {
    fprintf(Trace::out, "%llu %s called tag_array::probe(3rd in-arg mask) addr:%#llx\n", 
      time, caller.c_str(), addr);
  }

  std::string str_cache_name = m_config.get_cache_name();
  if (!strcmp(m_config.get_cache_name(), "L2") && mf) {
    str_cache_name += "_sub[";
    str_cache_name += std::to_string(mf->get_sub_partition());
    str_cache_name += "]";
  } else if (m_is_l1d && mf) {
    // printf("tag probe for L1D");
  }

  unsigned set_index = 0;
  new_addr_type tag = m_config.tag(addr);
  new_addr_type sector_addr = m_config.mshr_addr(raw_addr);
  // const shader_core_config *shader_cfg = m_gpu->getShaderCoreConfig();
  const bool warp_ctx_valid = mf && (mf->get_wid() < shader_cfg->max_warps_per_shader);
  if (m_is_l1d && warp_ctx_valid) {
    std::pair<unsigned, unsigned> set_index_pairs = m_config.set_index_pairs(addr, mf->get_wid());
    set_index = set_index_pairs.second; 
  // if (m_is_l1d && warp_ctx_valid) {
  //   std::pair<unsigned, unsigned> set_index_pairs = m_config.set_index_pairs(addr, mf->get_wid());
  //   unsigned candidate_set_1 = set_index_pairs.first;
  //   unsigned candidate_set_2 = set_index_pairs.second;

  //   if (candidate_set_1 == candidate_set_2) {
  //     set_index = candidate_set_1;
  //   } else {
  //     bool hit_in_set_1 = false;
  //     bool hit_in_set_2 = false;
  //     bool has_unreserved_1 = false;
  //     bool has_unreserved_2 = false;
  //     unsigned n_reserved_lines_1 = 0;
  //     unsigned n_reserved_lines_2 = 0;
  //     unsigned n_invalid_lines_1 = 0;
  //     unsigned n_invalid_lines_2 = 0;

  //     for (unsigned way = 0; way < m_config.m_assoc; way++) {
  //       unsigned index_1 = candidate_set_1 * m_config.m_assoc + way;
  //       cache_block_t *line_1 = m_lines[index_1];

  //       if (line_1->m_tag == tag && !line_1->is_invalid_line()) {
  //         hit_in_set_1 = true;
  //       }
  //       if (line_1->is_reserved_line()) {
  //         n_reserved_lines_1++;
  //       } else {
  //         has_unreserved_1 = true;
  //       }
  //       if (line_1->is_invalid_line()) {
  //         n_invalid_lines_1++;
  //       }

  //       unsigned index_2 = candidate_set_2 * m_config.m_assoc + way;
  //       cache_block_t *line_2 = m_lines[index_2];

  //       if (line_2->m_tag == tag && !line_2->is_invalid_line()) {
  //         hit_in_set_2 = true;
  //       }
  //       if (line_2->is_reserved_line()) {
  //         n_reserved_lines_2++;
  //       } else {
  //         has_unreserved_2 = true;
  //       }
  //       if (line_2->is_invalid_line()) {
  //         n_invalid_lines_2++;
  //       }
  //     }

  //     if (hit_in_set_1 && !hit_in_set_2) {
  //       set_index = candidate_set_1;
  //     } else if (hit_in_set_2) {
  //       set_index = candidate_set_2;
  //     } else if (has_unreserved_1 != has_unreserved_2) {
  //       set_index = has_unreserved_1 ? candidate_set_1 : candidate_set_2;
  //     } else if (n_invalid_lines_1 != n_invalid_lines_2) {
  //       set_index = (n_invalid_lines_1 > n_invalid_lines_2) ? candidate_set_1 : candidate_set_2;
  //     } else if (n_reserved_lines_1 != n_reserved_lines_2) {
  //       set_index = (n_reserved_lines_1 < n_reserved_lines_2) ? candidate_set_1 : candidate_set_2;
  //     } else {
  //       set_index = candidate_set_2;
  //     }
  //   }

    // set_index = m_config.set_index(addr, mf->get_wid()); // dead lock
  } else {
    set_index = m_config.set_index(addr);
  }

  // Just for checking if (tag === block_addr) (It seems not) // 1-14
  // [[maybe_unused]] new_addr_type block_addr = m_config.block_addr(addr); 

  // for debug
  if (m_config.get_sif() != 'L') {
    assert(1);
    // assert(addr == m_config.block_addr(addr));
    // assert(tag == m_config.block_addr(addr));
  }

  unsigned invalid_line = (unsigned) - 1;
  unsigned valid_line = (unsigned) - 1;
  u64 valid_timestamp = (unsigned) - 1;
  
  bool srrip_has_picked = false;

  bool all_reserved = true;
  bool all_miss = true;
  bool all_sector_valid = true;

  bool lru_has_picked = false;
  unsigned lru_picked_total_hits = (unsigned) - 1;
  u64 lru_picked_avg_evict_interval = (u64) - 1;
  unsigned last_warp_id = (unsigned) - 1;
  unsigned last_core_id = (unsigned) - 1;

  std::vector<std::pair<unsigned /* unfolded index */, LINE_RECENCY>> hybrid_rep_candidates;
  std::vector<std::pair<unsigned /* unfolded index */, LINE_RECENCY>> hybrid_rep_candidates_no_record_in_mshr;
  std::vector<std::pair<unsigned /* unfolded index */, LINE_RECENCY>> hybrid_rep_candidates_recorded_in_mshr;

  bool cache_hit = false;
  [[maybe_unused]] bool has_unreserved_line = false;

  if (m_is_l1d && mf) {
    auto& reref_gap_set = m_reref_gap[mf->get_sid()];
    for (auto& item : reref_gap_set) {
      if (item.first == addr) {
        if (item.second > 0) {
          u64 last_avg_reref_gap = m_avg_reref_gap[mf->get_sid()];
          m_avg_reref_gap[mf->get_sid()] = (last_avg_reref_gap + item.second) / 2;
          u64 new_reref_gap = time - item.second;
          if (DTRACE(REREF_GAP)) {
            fprintf(Trace::out, "%llu %s "
              "Updated m_avg_reref_gap[sid:%u] = %llu = "
              "(last:%llu + new_gap:%llu) / 2. "
              "Updated m_reref_gap[sid:%u][addr:%#llx] = %llu\n",
              time, m_config.get_cache_name(), 
              mf->get_sid(), m_avg_reref_gap[mf->get_sid()], 
              last_avg_reref_gap, item.second,
              mf->get_sid(), addr, new_reref_gap
            );
          }
        } else {
          // Has not been accessed since last eviction (nothing)
        }
        item.second = time - item.second; // update gap
        break;
      }
    }

    // m_l1d_unique_lines[mf->get_sid()].insert(addr);
    // assert(m_l1d_unique_lines.size() <= m_gpu->m_shader_config->num_shader());
  }
  
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];

    if (line->m_tag == tag) {
      line->inc_total_hits();
      cache_hit = true;
      lines_locality[addr]++;

      if (m_config.m_replacement_policy == SRRIP) {
        if (m_config.m_srrip_update_policy == srrip_update_policy_t::HP) {
          line->set_rrpv(0);
        } else if (m_config.m_srrip_update_policy == srrip_update_policy_t::FP) {
          if (line->get_rrpv() >= 1) {
            line->dec_rrpv();
            assert(line->get_rrpv() != ((unsigned) - 1));
          }
        } else {
          assert(0);
        }
      }

      all_miss = false;
      if (line->get_status(mask) == RESERVED) {
        idx = index;
        return HIT_RESERVED;
      } else if (line->get_status(mask) == VALID) {
        idx = index;
        return HIT;
      } else if (line->get_status(mask) == MODIFIED) {
        if ((!is_write && line->is_readable(mask)) || is_write) {        
          idx = index;
          return HIT;
        } else {
          if (DTRACE(TAG_PROBE)) {              
            fprintf(Trace::out, "%llu %s tag_array::probe "
              "is_write:%u is_readable:%u "
              "SECTOR_MISS for addr:%#llx\n", 
              time, m_config.get_cache_name(), 
              is_write, line->is_readable(mask), addr);
          }
          idx = index;
          return SECTOR_MISS;
        }
      } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
        idx = index;

        if (DTRACE(TAG_PROBE)) {              
          fprintf(Trace::out, "%llu %s tag_array::probe "
            "is_valid_line && line->get_status(mask) == INVALID "
            "SECTOR_MISS for addr:%#llx\n", 
            time, m_config.get_cache_name(), addr);
        }

        return SECTOR_MISS;
      } else {
        assert(line->get_status(mask) == INVALID);
        all_sector_valid = false;
      }
    } // cacheline hit
    line->m_n_rereferenced++;

    if (!line->is_reserved_line()) {
      has_unreserved_line = true;
      // percentage of dirty lines in the cache
      // number of dirty lines / total lines in the cache
      float dirty_line_percentage =
          ((float)m_dirty / (m_config.m_nset * m_config.m_assoc)) * 100;
      // If the cacheline is from a load op (not modified),
      // or the total dirty cacheline is above a specific value,
      // Then this cacheline is eligible to be considered for replacement
      // candidate i.e. Only evict clean cachelines until total dirty cachelines
      // reach the limit.
      if (!line->is_modified_line() ||
          dirty_line_percentage >= m_config.m_wr_percent) {
        all_reserved = false;
        if (line->is_invalid_line()) {
          invalid_line = index;
          if (DTRACE(CHECK_INVALID_LINE)) {
            fprintf(Trace::out, "%llu %s %s Found invalid_line = index = %u for addr:%#llx\n",
              time, caller.c_str(), m_config.get_cache_name(), index, addr);
          }
        } else {
          gather_rep_candidates(
            time, mf, line, set_index, index /* unfolded */, 
            hybrid_rep_candidates_no_record_in_mshr,
            hybrid_rep_candidates_recorded_in_mshr,
            hybrid_rep_candidates);

          if (m_config.m_warp_interfere_aware == 'T') {
            // assert(m_is_l1d);
          } else if (m_config.m_replacement_policy == SRRIP) {
            if (!srrip_has_picked) {
              if (line->get_rrpv() == line->get_max_rrpv()) {
                valid_line       = index;
                srrip_has_picked = true;
              } else {
                if (line->get_rrpv() < line->get_max_rrpv()) {
                  line->inc_rrpv();
                }
              }
            }
          } else if (m_config.m_replacement_policy == LRU) {
            lru_pick(
              line, valid_timestamp, 
              valid_line, index, last_warp_id, last_core_id, lru_has_picked);
          } else if (m_config.m_replacement_policy == FIFO) {
            if (line->get_alloc_time() < valid_timestamp) {
              valid_timestamp = line->get_alloc_time();
              valid_line = index;
            } 
          }
        } // valid line
      } // evict conditions (clean || too many dirty lines)
    } // if (!line->is_reserved_line())
  } // for (unsigned way = 0; way < m_config.m_assoc; way++)

  if (cache_hit) {
    assert(all_sector_valid == false);
  }

  if (all_reserved) {
    assert(m_config.m_alloc_policy == ON_MISS);
    return RESERVATION_FAIL;  // miss and not enough space in cache to allocate on miss
  }
  
  assert(all_miss == all_sector_valid);
  
  if (invalid_line != (unsigned) - 1) {
    // m_is_l2 && (invalid_line == (unsigned) - 1) can also happen (srad_v2-rodinia-2.0-ft)
    idx = invalid_line;
    if (DTRACE(L2_EVICTION) || DTRACE(CACHE_REPLACE)) {
      fprintf(Trace::out, "%llu %s %s victim = invalid_line = %u for addr:%#llx\n",
        time, caller.c_str(), m_config.get_cache_name(), idx, addr);
    }
  } else if (m_config.m_warp_interfere_aware == 'T') {
    if (DTRACE(CHECK_L2_CFG) && m_is_l2) {
      fprintf(Trace::out, "%llu %s m_warp_interfere_aware = T", 
        time, m_config.get_cache_name());
    }
    assert(valid_line == (unsigned) - 1);
    bool has_interfered = false;
    warp_interfere_aware_pick(
      time,
      hybrid_rep_candidates,
      hybrid_rep_candidates_no_record_in_mshr,
      hybrid_rep_candidates_recorded_in_mshr,
      has_interfered,
      valid_line /* later used */, 
      last_warp_id, last_core_id, mf);
    idx = valid_line;
    inter_warp_has_interference = has_interfered;
    if (DTRACE(L2_EVICTION) && m_is_l2) {
      fprintf(Trace::out, "%llu %s victim = valid_line = %u for addr:%#llx\n",
        time, m_config.get_cache_name(), idx, addr);
    } 
    if (has_interfered) {
      assert(last_core_id == mf->get_sid());
      inter_warp_interfere_record.last_warp_id = last_warp_id;
      inter_warp_interfere_record.curr_warp_id = mf->get_wid();
      if (DTRACE(WARP_INTERFERE)) {
        if (has_interfered) {
          fprintf(Trace::out, "%llu warp:%u evicted idx:%#x hit by warp:%u last time\n",
            time, mf->get_wid(), idx, last_warp_id);
        }
      }      
    }
  } else {
    if (valid_line != (unsigned) - 1) {
      if (m_config.m_replacement_policy == LRU) {
        assert(lru_has_picked);
        assert(hybrid_rep_candidates.size());

        // Possible re-pick from "hybrid_rep_candidates_in_use"
        pick_with_lru(
          hybrid_rep_candidates,
          hybrid_rep_candidates_no_record_in_mshr,
          hybrid_rep_candidates_recorded_in_mshr,
          valid_line, last_warp_id, last_core_id, valid_timestamp,
          lru_picked_total_hits, lru_picked_avg_evict_interval);

        if (m_config.m_fill_time_ascend == 'T') {
          fill_time_awared_modification_for_lru(
            hybrid_rep_candidates,
            hybrid_rep_candidates_no_record_in_mshr,
            hybrid_rep_candidates_recorded_in_mshr,
            valid_line,
            valid_timestamp);
        }
      } // LRU 
      else if (m_config.m_replacement_policy == SRRIP) {
        assert(srrip_has_picked);
        mshr_awared_modification_for_srrip(
          hybrid_rep_candidates,
          hybrid_rep_candidates_no_record_in_mshr,
          hybrid_rep_candidates_recorded_in_mshr,
          valid_line);
        if (m_config.m_fill_time_ascend == 'T') {
          fill_time_awared_modification_for_srrip(
            hybrid_rep_candidates,
            hybrid_rep_candidates_no_record_in_mshr,
            hybrid_rep_candidates_recorded_in_mshr,
            valid_line);
        }
      } // SRRIP
      // if (mf) {
      //   mf->set_victim_avg_evict_interval(m_lines[valid_line]->get_avg_evict_interval());
      // }
      idx = valid_line;
      if (DTRACE(L2_EVICTION) && m_is_l2) {
        fprintf(Trace::out, "%llu %s victim = valid_line = %u for addr:%#llx\n",
          time, m_config.get_cache_name(), idx, addr);
      }
    } else if (valid_line == (unsigned) - 1) {
      if (m_config.m_replacement_policy == LRU) {
        assert(0);
      } else if (m_config.m_replacement_policy == SRRIP) {
        assert(!srrip_has_picked);

        pick_with_lru(
          hybrid_rep_candidates,
          hybrid_rep_candidates_no_record_in_mshr,
          hybrid_rep_candidates_recorded_in_mshr,
          idx, last_warp_id, last_core_id, valid_timestamp,
          lru_picked_total_hits, lru_picked_avg_evict_interval);
        if (m_config.m_fill_time_ascend == 'T') {
          fill_time_awared_modification_for_lru(
            hybrid_rep_candidates,
            hybrid_rep_candidates_no_record_in_mshr,
            hybrid_rep_candidates_recorded_in_mshr,
            idx,
            valid_timestamp);
        }
        if (DTRACE(L2_EVICTION) && m_is_l2) {
          fprintf(Trace::out, "%llu %s !valid_line. victim = idx = %u for addr:%#llx\n",
            time, m_config.get_cache_name(), idx, addr);
        }
      } // SRRIP
    } // else if (valid_line == (unsigned) - 1) 
  } // else if (m_config.m_warp_interfere_aware == 'F')

  assert(idx != (unsigned) - 1);

  ////////////////////////////////////// Begin of L1D VC //////////////////////////////////////
  if (m_is_l1d && mf) {
    cache_block_t *victim = m_lines[idx];
    const new_addr_type swap_out_addr = victim->m_block_addr;  
    const new_addr_type victim_addr = victim->m_block_addr;
    BYPASS_KEY victim_key(victim->m_stream_id, victim->m_kernel, victim_addr);  
    LOCALITY_KEY loc_key(victim->m_stream_id, victim->m_kernel);

    [[maybe_unused]] const int max_evictions_bound = 5;
    // const bool swap_out_valid = victim->is_valid_line() || victim->is_modified_line();
    const bool swap_out_valid = victim->is_valid_line() && !victim->is_modified_line();
    // Begin of probing L1D victim cache
    if (m_config.m_victim_cache_enable == 'T') {
      // VC hit in-coming addr 
      if (cache->m_victim_cache.find(addr) != cache->m_victim_cache.end()) {
        // Swap in between L1D and VC
        if (swap_out_valid) {
          // L1D side:
          victim->allocate(
            m_config.tag(addr), m_config.block_addr(addr), mf->get_streamID(),
            m_gpu->m_kernel_id, time, mf->get_access_sector_mask());          
          fill(cache, idx, time, mf);
          // VC side:
          cache->m_victim_cache.erase(addr);
          cache->m_victim_cache.insert(swap_out_addr);        
        }
        cache->m_stats.inc_l1d_vc_hits(loc_key);
        cache->m_stats.inc_l1d_vc_accesses(loc_key);
        return VC_HIT;
      }
      else { // VC missed in-coming addr as well
        if (swap_out_valid) {
          if (cache->m_victim_cache.size() < m_config.m_victim_cache_entries) {
            cache->m_victim_cache.insert(swap_out_addr);
          }
        }
        cache->m_stats.inc_l1d_vc_misses(loc_key);
        cache->m_stats.inc_l1d_vc_accesses(loc_key);
      }
    } // End of probing L1D victim cache
  }
  ////////////////////////////////////// Begin of L1D VC //////////////////////////////////////

  if (m_is_l1d) {
    cache_block_t *victim = m_lines[idx];
    victim->inc_total_evictions();
    m_gpu->get_shader_stats()->m_l1d_victims[mf->get_sid()]++;
    assert(victim->is_reserved_line() == false);
  }

  if (m_config.m_warp_interfere_aware == 'F') {
    if (valid_line != (unsigned) - 1) {
      if (last_warp_id == (unsigned) - 1) {
        assert(last_warp_id == last_core_id);
      } else {
        if (last_warp_id != mf->get_wid()) {
          inter_warp_has_interference = true;
          inter_warp_interfere_record.last_warp_id = last_warp_id;
          inter_warp_interfere_record.curr_warp_id = mf->get_wid();
        } else {
          m_gpu->get_shader_stats()->intra_warp_interfere[mf->get_sid()][last_warp_id]++;
          if (DTRACE(INTRA_WARP_INTERFERE)) {
            fprintf(Trace::out, "%llu %s intra-warp-interfere found "
              "for warp:%u on addr:%#llx\n",
              time, m_config.get_cache_name(), last_warp_id, addr
            );
          }
        }
      }
    }
  }

  if (idx == (unsigned) - 1) {
    fprintf(stderr, "tag_array::probe - Error: No victim found for addr %#llx in %s\n",
            addr, m_config.get_cache_name());
    abort();
  }

  if (m_is_l1d && mf) {
    cache_block_t *victim = m_lines[idx];
    const new_addr_type victim_addr = victim->m_block_addr;
    BYPASS_KEY victim_key(victim->m_stream_id, victim->m_kernel, victim_addr);  
    m_l1d_evictions[victim_key]++;
  }  
  
  // Switch on/off L1D bypass
  if (m_is_l1d && m_config.m_bypass_enable == 'T' && mf && !mf->is_write() && !mf->isatomic()) {
    assert(addr == m_config.block_addr(addr));
    cache_block_t *victim = m_lines[idx];
    const new_addr_type swap_out_addr = victim->m_block_addr;  
    const new_addr_type victim_addr = victim->m_block_addr;
    BYPASS_KEY incoming_key(mf->get_streamID(), m_gpu->m_kernel_id, addr);
    BYPASS_KEY victim_key(victim->m_stream_id, victim->m_kernel, victim_addr);  
    LOCALITY_KEY loc_key(victim->m_stream_id, victim->m_kernel);
    auto it_f2e_line = m_l1d_fill_to_evict_lines.find(loc_key);
    if (it_f2e_line == m_l1d_fill_to_evict_lines.end()) {
      std::set<new_addr_type> addr_set;
      addr_set.insert(addr);
      m_l1d_fill_to_evict_lines[loc_key] = addr_set;
    } else {
      m_l1d_fill_to_evict_lines[loc_key].insert(addr);
    }

    bool income_key_hit = false;
    bool victim_key_hit = false;

    bool new_byp_cand   = false;
    bool new_incom_byp  = false;
    bool new_victim_byp = false;

    if (m_l1d_occupied.find(incoming_key) != m_l1d_occupied.end() && m_l1d_occupied[incoming_key]) {
      assert(0); // bypassed L1D item should never be probed again
      if (DTRACE(REFILL_LFB_BYP_FILL_TAG)) {
        fprintf(Trace::out, "%llu L1D bypassed item was refilled, "
          "and now probed again for key(streamID:%llu, kernel:%u, sector_addr:%#llx)\n", 
          time, mf->get_streamID(), m_gpu->m_kernel_id, sector_addr);
      }
    }

    auto it_income_key = m_l1d_rd_fill_to_evict_gap.find(incoming_key);
    auto it_victim_key = m_l1d_rd_fill_to_evict_gap.find(victim_key);
    if (it_income_key != m_l1d_rd_fill_to_evict_gap.end()) {
      income_key_hit = true;
    }
    if (it_victim_key != m_l1d_rd_fill_to_evict_gap.end()) {
      victim_key_hit = true;
    }
    if (!income_key_hit && !victim_key_hit) {
      for (auto& record : m_l1d_rd_fill_to_evict_gap) {
        bool new_incom_byp = 
          record.first.stream_id == mf->get_streamID() &&
          record.first.kernel == m_gpu->m_kernel_id;
        bool new_victim_byp = 
          record.first.stream_id == victim->m_stream_id &&
          record.first.kernel == victim->m_kernel;
        if (new_incom_byp || new_victim_byp) {
          new_byp_cand = true;
        }
      }
    }

    set_l1d_rd_fill_to_evict_gap(victim_key, time - get_l1d_rd_fill_time(victim_key));

    [[maybe_unused]] const float incoming_byp_ratio = 0.0;
    // [[maybe_unused]] const float incoming_byp_ratio = 0.2; // 120.354 (-0.091%)	l1d_byp_T_F_T_40_3_3_lrr_srad_v2
    // [[maybe_unused]] const float incoming_byp_ratio = 0.3; // 120.620 (+0.130%)	l1d_byp_T_F_T_40_3_3_incoming_030_lrr_srad_v2	
    // [[maybe_unused]] const float incoming_byp_ratio = 0.7;
    // [[maybe_unused]] const float incoming_byp_ratio = 0.5;
    // [[maybe_unused]] const float incoming_byp_ratio = 1.0;
    if (new_byp_cand) { // Just update, and nothing to do with bypass decision
      set_l1d_evict_time(victim_key, time);
      average_l1d_rd_fill_to_evict_gap(victim_key);      
      m_l1d_rd_bypass_confidence[victim_key] = 0;
    } 
    else if (income_key_hit) {
      set_l1d_evict_time(victim_key, time);
      average_l1d_rd_fill_to_evict_gap(victim_key);

      // Begin of total-evictions-aware scheme
      if (m_config.m_total_evictions_aware == 'T') {
        if (get_l1d_evictions(incoming_key) > m_config.m_max_evictions_bound) {
          if (m_config.m_infinite_bypasses == 'T') {
            m_incoming_bypasses.insert(incoming_key);
            m_trashed_reqs.insert(incoming_key);
            mf->set_l1d_rd_byp_activated();
            m_l1d_rd_byp_activated_times[incoming_key]++;
            assert(hit_l1d_bypassed_item(incoming_key, mf));        
          } else {
            if (m_incoming_bypasses.size() < (incoming_byp_ratio * m_config.m_max_bypasses) &&
                m_trashed_reqs.size() < m_config.m_max_bypasses) {
              m_incoming_bypasses.insert(incoming_key); // -gpgpu_cache:l1d_bypass T,F,T,40,3,3 reaches here
              m_trashed_reqs.insert(incoming_key);            
              mf->set_l1d_rd_byp_activated();
              m_l1d_rd_byp_activated_times[incoming_key]++;
              assert(hit_l1d_bypassed_item(incoming_key, mf));
            }
          }
        }
      } // End of total-evictions-aware scheme
    } else if (victim_key_hit) {
      set_l1d_evict_time(victim_key, time);
      average_l1d_rd_fill_to_evict_gap(victim_key);  
      // Begin of total-evictions-aware scheme
      const char* insert_src = "xx";
      if (m_config.m_total_evictions_aware == 'T') {
        if (get_l1d_evictions(victim_key) > m_config.m_max_evictions_bound) {
          if (m_config.m_infinite_bypasses == 'T') {
            m_victim_bypasses.insert(victim_key);
            m_trashed_reqs.insert(victim_key);       
            mf->set_l1d_rd_byp_activated(); // m_l1d_rd_byp_change = 2 = 2'b10
            m_l1d_rd_byp_activated_times[victim_key]++;
            assert(hit_l1d_bypassed_item(victim_key, mf));
            if (DTRACE(INSERTED_BYP_ITEM)) {
              insert_src = "scheme1 infinite_bypasses";
              fprintf(Trace::out, "%llu caller:%s %s INSERTED_BYP_ITEM "
                "<streamID:%llu, kernel:%u, block_addr:%#llx> victim->m_allocated = %u\n", 
                time, caller.c_str(), insert_src, victim_key.stream_id, victim_key.kernel, victim_key.sector_addr,
                victim->m_allocated);
            }
          } else {
            if (m_victim_bypasses.size() < ((1 - incoming_byp_ratio) * m_config.m_max_bypasses) &&
                m_trashed_reqs.size() < m_config.m_max_bypasses) {

              if (m_config.m_bypass_mode == 0) {
                // bool swap_out_valid = victim->is_valid_line() || victim->is_modified_line();
                bool swap_out_valid = victim->is_valid_line() && !victim->is_modified_line();
                const new_addr_type swap_out_addr = victim->m_block_addr;          
                const new_addr_type victim_addr = victim->m_block_addr;
                if (swap_out_valid) {                
                  ////////////////////////// Look-up VB //////////////////////////
                  if (cache->m_victim_buffer.find(addr) != cache->m_victim_buffer.end()) {
                    /////////////////////////// VB ///////////////////////////
                    // L1D side:
                    // 1) Allocate the line that is to be filled
                    // victim->m_block_addr would be updated inside below
                    victim->allocate(
                        m_config.tag(addr), m_config.block_addr(addr), mf->get_streamID(),
                        m_gpu->m_kernel_id, time, mf->get_access_sector_mask());          
                    // 2) Fill the line (sector) with new timestamp and status       
                    fill(cache, idx, time, mf);

                    // VB side:
                    // 1) Remove the hit address that has been swapped into L1D
                    cache->m_victim_buffer.erase(addr);
                    // 2) Insert the swapped out address into VC
                    cache->m_victim_buffer.insert(swap_out_addr); // logic
                    m_l1d_evictions[victim_key]++; // stats

                    if (DTRACE(L1D_VICTIM_BUFFER)) {
                      fprintf(Trace::out, "%llu TPC:%u SM:%u L1D VB hit. "
                        "L1D victim' swap_out_addr:%#llx "
                        "swapped with VB's addr:%#llx\n", 
                        time, mf->get_tpc(), mf->get_sid(), swap_out_addr, addr);
                    }

                    return VC_HIT;
                    ///////////////////////////////////////////////////////////////
                  } else if (cache->m_victim_buffer.size() < m_config.m_victim_cache_entries) {
                    // Just insert new entry into VB
                    cache->m_victim_buffer.insert(swap_out_addr);
                    if (DTRACE(L1D_VICTIM_BUFFER)) {
                      fprintf(Trace::out, "%llu TPC:%u SM:%u L1D VB missed in-coming addr:%#llx "
                        "and inserted victim's swap_out_addr:%#llx (->size:%lu). "
                        "victim->is_modified_line:%u\n", 
                        time, mf->get_tpc(), mf->get_sid(), 
                        addr, swap_out_addr, 
                        cache->m_victim_buffer.size(),
                        victim->is_modified_line());
                    }
                  }     
                } // swap condition satisfies
              }
              
              m_victim_bypasses.insert(victim_key);
              m_trashed_reqs.insert(victim_key); 
              mf->set_l1d_rd_byp_activated(); // m_l1d_rd_byp_change = 2 = 2'b10
              m_l1d_rd_byp_activated_times[victim_key]++;
              assert(hit_l1d_bypassed_item(victim_key, mf));
              if (DTRACE(INSERTED_BYP_ITEM)) {
                insert_src = "scheme1 finite_bypasses";
                fprintf(Trace::out, "%llu caller:%s %s INSERTED_BYP_ITEM "
                  "<streamID:%llu, kernel:%u, block_addr:%#llx> victim->m_allocated = %u\n", 
                  time, caller.c_str(), insert_src, victim_key.stream_id, victim_key.kernel, victim_key.sector_addr,
                  victim->m_allocated);
              }                
            }
          }
        } // Bypass is activated (1st scheme)
      } // End of total-evictions-aware scheme

      // Begin of locality-aware scheme
      // Inc confidence && possible insert into trash set
      if (get_l1d_rd_fill_to_evict_gap(victim_key) < get_avg_l1d_rd_fill_to_evict_gap(victim_key)) {
        inc_conf_cnt(m_l1d_rd_bypass_confidence[victim_key], m_config.m_trash_conf_cnt_bound, 1);
        if (m_l1d_rd_bypass_confidence[victim_key] == m_config.m_trash_conf_cnt_bound) {
          if (m_config.m_infinite_bypasses == 'T') {
            m_trashed_reqs.insert(victim_key);
            mf->set_l1d_rd_byp_activated();
            m_l1d_rd_byp_activated_times[victim_key]++;
            assert(hit_l1d_bypassed_item(victim_key, mf));
            if (DTRACE(INSERTED_BYP_ITEM)) {
              insert_src = "scheme2 infinite_bypasses";
              fprintf(Trace::out, "%llu caller:%s %s INSERTED_BYP_ITEM "
                "<streamID:%llu, kernel:%u, block_addr:%#llx> victim->m_allocated = %u\n", 
                time, caller.c_str(), insert_src, victim_key.stream_id, victim_key.kernel, victim_key.sector_addr,
                victim->m_allocated);
            }            
          } else {
            if (m_victim_bypasses.size() < ((1 - incoming_byp_ratio) * m_config.m_max_bypasses) &&
                m_trashed_reqs.size() < m_config.m_max_bypasses) {
              m_victim_bypasses.insert(victim_key);        
              m_trashed_reqs.insert(victim_key);
              mf->set_l1d_rd_byp_activated(); // m_l1d_rd_byp_change = 2 = 2'b10
              m_l1d_rd_byp_activated_times[victim_key]++;
              assert(hit_l1d_bypassed_item(victim_key, mf));
              if (DTRACE(INSERTED_BYP_ITEM)) {
                insert_src = "scheme2 finite_bypasses";
                fprintf(Trace::out, "%llu caller:%s %s INSERTED_BYP_ITEM "
                  "<streamID:%llu, kernel:%u, block_addr:%#llx> victim->m_allocated = %u\n", 
                  time, caller.c_str(), insert_src, victim_key.stream_id, victim_key.kernel, victim_key.sector_addr,
                  victim->m_allocated);
              }              
            }
          }
        }
      } // End of locality-aware scheme
    } // victim_key_hit = True

    // m_l1d_trashed_lines.insert(addr);
    // m_gpu->get_shader_stats()->m_n_l1d_trashed_lines[mf->get_sid()] = m_l1d_trashed_lines.size();    

    // m_l1d_max_evicts[mf->get_sid()] = std::max(m_l1d_max_evicts[mf->get_sid()], m_l1d_max_evictions[addr]);
    // m_gpu->get_shader_stats()->m_l1d_max_evicts[mf->get_sid()] = m_l1d_max_evicts[mf->get_sid()];

    // m_l1d_avg_evicts[mf->get_sid()] = (m_l1d_avg_evicts[mf->get_sid()] + m_l1d_max_evictions[addr]) >> 1;
    // m_gpu->get_shader_stats()->m_l1d_avg_evicts[mf->get_sid()] = m_l1d_avg_evicts[mf->get_sid()];
  } // if (m_is_l1d && mf && !mf->is_write() && !mf->isatomic()) {

  // if (m_is_l1d && mf) {
  //   cache_block_t *victim = m_lines[idx];
  //   const new_addr_type victim_addr = victim->m_block_addr;
  //   BYPASS_KEY victim_key(victim->m_stream_id, victim->m_kernel, victim_addr);  
  //   m_l1d_evictions[victim_key]++;
  // }  

  // 2/25 Reset MSHR record
  // reset_record_in_mshr(idx);
  if (DTRACE(MSHR_AWARED_REPL)) {
    fprintf(Trace::out, "%llu %s reset_record_in_mshr(index:%u)\n",
      time, m_config.get_cache_name(), idx);
  }

  if (m_is_l1d && (idx != (unsigned) - 1)) {
    // Eviction is the start point of each reref_gap record
    // Regardless if there has already been a valid (> 0) reref_gap,
    // reset the gap to zero as the start point of a new record.
    auto& reref_gap_set = m_reref_gap[mf->get_sid()];
    const unsigned last_size = reref_gap_set.size();
    bool found = false;
    for (auto& item : reref_gap_set)
    {
      if (item.first == addr && item.second > 0) { 
        item.second = 0;
        if (DTRACE(REREF_GAP)) {
          fprintf(Trace::out, "%llu %s Reset reref_gap[sid:%u][addr:%#llx] to 0\n",
            time, m_config.get_cache_name(), mf->get_sid(), addr);
        }
        found = true;
        break;
      }
    }
    if (!found) {
      m_reref_gap[mf->get_sid()].push_back(std::make_pair(addr, 0));
      if (DTRACE(REREF_GAP)) {
        fprintf(Trace::out, "%llu %s Added reref_gap[sid:%u][addr:%#llx] (size:%u->%lu)\n",
          time, m_config.get_cache_name(), mf->get_sid(), addr,
          last_size, m_reref_gap[mf->get_sid()].size()
        );
      }  
    }
  }

  return MISS;
}

void tag_array::inc_rrpv_for_one_set(unsigned set_index) {
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];
    if (line->get_rrpv() == line->get_max_rrpv()) {
      // do nothing
      if (DTRACE(DEBUG_SRRIP)) {
        fprintf(Trace::out, "m_lines[index:%#x]->get_rrpv = 3 "
          "inside inc_rrpv_for_one_set\n", index);
      }
    } else {
      line->inc_rrpv();
      if (DTRACE(SRRIP_INC_RRPV)) {
        fprintf(Trace::out, "%s Inside tag_array::inc_rrpv_for_one_set, "
          "m_lines[index:%#x] inc_rrpv. rrpv = %u\n",
          m_config.m_cache_name, index, line->get_rrpv()
        );
      }
    }
    assert(line->get_rrpv() <= line->get_max_rrpv());
  }
}
bool tag_array::already_has_max_rrpv_in_one_set(unsigned set_index) {
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];
    if (line->get_rrpv() == line->get_max_rrpv()) {
      return true;
    }    
  }
  return false;
}

enum cache_request_status tag_array::access(
  baseline_cache *cache,
  new_addr_type raw_addr,
  new_addr_type addr /* block_addr */, 
  u64 time,
  unsigned &idx, mem_fetch *mf,
  bool bypass_2nd_probe) {
  bool wb = false;
  evicted_block_info evicted;
  
  enum cache_request_status result = 
    access(cache, raw_addr, addr, time, idx, wb, evicted, mf, bypass_2nd_probe);
  assert(!wb);
  return result;
}

enum cache_request_status tag_array::access(
  baseline_cache *cache,
  new_addr_type raw_addr,
  new_addr_type addr /* block_addr */, 
  u64 time,
  unsigned &idx, bool &wb,
  evicted_block_info &evicted,
  mem_fetch *mf,
  bool bypass_2nd_probe) {

  m_access++;
  is_used = true;
  shader_cache_access_log(m_core_id, m_type_id, 0);  // log accesses to cache

  bool inter_warp_has_interference = false;
  WARP_INTERFERE_RECORD inter_warp_interfere_record((unsigned )- 1, (unsigned) - 1);

  enum cache_request_status status = MISS;
  if (bypass_2nd_probe) {
    status = VC_HIT;
  } else {
    status = probe("tag_array::access", cache, raw_addr, addr, idx, mf, mf->is_write(), time, 
        inter_warp_has_interference, inter_warp_interfere_record);    
  }
  // enum cache_request_status status = 
  //   probe("tag_array::access", cache, raw_addr, addr, idx, mf, mf->is_write(), time, 
  //     inter_warp_has_interference, inter_warp_interfere_record);

  switch (status) {
    case HIT_RESERVED:
      m_pending_hit++;
    case HIT:
      mf->get_access_type() == GLOBAL_ACC_W ? m_writes++ : m_reads++;
      m_lines[idx]->set_last_access_time(time, mf->get_access_sector_mask());
      break;
    case VC_HIT:
      mf->get_access_type() == GLOBAL_ACC_W ? m_writes++ : m_reads++;
      m_lines[idx]->set_last_access_time(time, mf->get_access_sector_mask());
      break;
    case MISS:
      mf->get_access_type() == GLOBAL_ACC_W ? m_writes++ : m_reads++;
      mf->get_access_type() == GLOBAL_ACC_W ? m_wr_miss++ : m_rd_miss++;
      m_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      if (m_config.m_alloc_policy == ON_MISS) {
        if (m_lines[idx]->is_modified_line()) {
          wb = true;
          evicted.set_info(
            m_lines[idx]->m_stream_id,
            m_lines[idx]->m_kernel,
            m_lines[idx]->m_block_addr,
            m_lines[idx]->get_modified_size(),
            m_lines[idx]->get_dirty_byte_mask(),
            m_lines[idx]->get_dirty_sector_mask());
          m_dirty--;
        }
        m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr),
                               mf->get_streamID(), m_gpu->m_kernel_id,
                               time, mf->get_access_sector_mask());
      }
      break;
    case SECTOR_MISS:
      assert(m_config.m_cache_type == SECTOR);
      mf->get_access_type() == GLOBAL_ACC_W ? m_wr_sector_miss++ : m_rd_sector_miss++;
      mf->get_access_type() == GLOBAL_ACC_W ? m_writes++ : m_reads++;
      m_sector_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      if (m_config.m_alloc_policy == ON_MISS) {
        bool before = m_lines[idx]->is_modified_line();
        ((sector_cache_block *)m_lines[idx])
            ->allocate_sector(time, mf->get_access_sector_mask());
        if (before && !m_lines[idx]->is_modified_line()) {
          m_dirty--;
        }
      }
      break;
    case RESERVATION_FAIL:
      m_res_fail++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      break;
    default:
      fprintf(stderr,
              "tag_array::access - Error: Unknown"
              "cache_request_status %d\n",
              status);
      abort();
  }
  return status;
}

void tag_array::fill(
  baseline_cache *cache,
  new_addr_type addr, u64 time, mem_fetch *mf, bool is_write) {
  fill(cache, addr, time, mf->get_access_sector_mask(), mf->get_access_byte_mask(), is_write, mf);
}

// This function will be called in following two scenarios:
// 1) m_config.m_alloc_policy == ON_FILL inside
//    baseline_cache::fill(mem_fetch *mf, u64 time)
// 2) L2 memcpy
void tag_array::fill(
  baseline_cache *cache,
  new_addr_type addr, u64 time,
  mem_access_sector_mask_t mask,
  mem_access_byte_mask_t byte_mask, bool is_write,
  mem_fetch *mf) {

  if (DTRACE(TAG_FILL)) {
    if (mf) {
      fprintf(Trace::out, "%llu %s tag_array::fill mf "
        "[sid:%u][warp:%u][addr:%#llx]\n",
        time, m_config.get_cache_name(), 
        mf->get_sid(), mf->get_wid(), mf->get_addr());
    }
  }
  
  unsigned idx;

  bool inter_warp_has_interference = false;
  WARP_INTERFERE_RECORD inter_warp_interfere_record((unsigned )- 1, (unsigned) - 1);

  enum cache_request_status status = 
    probe("tag_array::fill", cache, addr, m_config.block_addr(addr), 
      idx, mask, is_write, time, false /* probe_mode */,
      inter_warp_has_interference, inter_warp_interfere_record, mf);

  if (status == RESERVATION_FAIL) {
    return;
  }

  bool before = m_lines[idx]->is_modified_line();
  // assert(status==MISS||status==SECTOR_MISS); // MSHR should have prevented
  // redundant memory request
  if (status == MISS) {
    if (mf) {
      m_lines[idx]->allocate(
        m_config.tag(addr), m_config.block_addr(addr), 
        mf->get_streamID(), m_gpu->m_kernel_id, time, mask); 
    } else {
      m_lines[idx]->allocate(
        m_config.tag(addr), m_config.block_addr(addr), 
        (u64) - 1, m_gpu->m_kernel_id, time, mask);
    }
  } else if (status == SECTOR_MISS) {
    assert(m_config.m_cache_type == SECTOR);
    ((sector_cache_block *)m_lines[idx])->allocate_sector(time, mask);
  }
  if (before && !m_lines[idx]->is_modified_line()) {
    m_dirty--;
  }
  before = m_lines[idx]->is_modified_line();
  m_lines[idx]->fill(time, mask, byte_mask);
  if (m_lines[idx]->is_modified_line() && !before) {
    m_dirty++;
  }
}

void tag_array::fill(
  baseline_cache *cache,
  unsigned index, u64 time, mem_fetch *mf) {

  if (mf && mf->get_inst().is_load()) {
    if (DTRACE(LOAD_PIPE)) {      
      auto warp_inst = mf->get_inst();      
      fprintf(Trace::out, "%llu tag_array::fill %s for inst %s\n",
        time, m_config.get_cache_name(),
        warp_inst.get_inst_info(mf->get_sid(), mf->get_request_uid()).c_str());
    }
  } 

  if (DTRACE(TAG_FILL)) {
    if (mf) {
      fprintf(Trace::out, "%llu %s tag_array::fill mf "
        "[sid:%u][warp:%u][addr:%#llx]\n",
        time, m_config.get_cache_name(), 
        mf->get_sid(), mf->get_wid(), mf->get_addr());
    }
  }

  assert(m_config.m_alloc_policy == ON_MISS);
  bool before = m_lines[index]->is_modified_line();

  const shader_core_config *shader_cfg = m_gpu->getShaderCoreConfig();
  // m_lines[index]->print_status();  
  if (DTRACE(LINE_STATUS_CHANGE)) {
    fprintf(Trace::out, "%llu %s tag_array::fill addr:%#llx "
      "status = {%s %s %s %s}\n",
      time, m_config.get_cache_name(),
      m_lines[index]->m_block_addr,
      m_lines[index]->get_sector_status(0).c_str(),
      m_lines[index]->get_sector_status(1).c_str(),
      m_lines[index]->get_sector_status(2).c_str(),
      m_lines[index]->get_sector_status(3).c_str()
    );
  }
  [[maybe_unused]] bool reserved = m_lines[index]->is_reserved_line();
  [[maybe_unused]] bool modified = m_lines[index]->is_modified_line();
  // 3/5 Temporarily commented for no MSHR cfg
  // assert(reserved | modified);
  if (m_is_l1d && mf && (mf->get_wid() < shader_cfg->max_warps_per_shader)) {
    // {set_last_warp_id, set_last_core_id} has already been done inside func below.
    m_lines[index]->fill(time, mf->get_access_sector_mask(), mf->get_access_byte_mask(), mf);
    assert(m_lines[index]->get_last_core_id() != ((unsigned) - 1));
    assert(m_lines[index]->get_last_warp_id() != ((unsigned) - 1));
    if (DTRACE(WARP_CACHE_INTERFERE)) {
      fprintf(Trace::out, "%llu %s tag_array::fill "
        "m_lines[index:%u]->set_last_warp_id:%u set_last_core_id:%u for addr:%#llx\n",
        time, m_config.get_cache_name(), index, 
        mf->get_wid(), mf->get_sid(), m_lines[index]->m_block_addr);
    }
  } else {
    m_lines[index]->fill(time, mf->get_access_sector_mask(), mf->get_access_byte_mask());  
  }
  // m_lines[index]->fill(time, mf->get_access_sector_mask(), mf->get_access_byte_mask());
  if (m_lines[index]->is_modified_line() && !before) {
    m_dirty++;
  }
}

void tag_array::reset_record_in_mshr(unsigned index) {
  m_lines[index]->m_was_recorded_in_mshr = false;
}

void tag_array::set_recorded_in_mshr(unsigned index) {
  m_lines[index]->m_was_recorded_in_mshr = true;
}

// TODO: we need write back the flushed data to the upper level
void tag_array::flush() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    if (m_lines[i]->is_modified_line()) {
      for (unsigned j = 0; j < SECTOR_CHUNK_SIZE; j++) {
        m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j), "tag_array::flush()");
      }
    }

  m_dirty = 0;
  is_used = false;
}

void tag_array::invalidate() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    for (unsigned j = 0; j < SECTOR_CHUNK_SIZE; j++)
      m_lines[i]->set_status(
        INVALID, mem_access_sector_mask_t().set(j), 
        "tag_array::invalidate() " + std::string(m_config.get_cache_name()));

  m_dirty = 0;
  is_used = false;
}

float tag_array::windowed_miss_rate() const {
  unsigned n_access = m_access - m_prev_snapshot_access;
  unsigned n_miss = (m_miss + m_sector_miss) - m_prev_snapshot_miss;
  // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

  float missrate = 0.0f;
  if (n_access != 0) missrate = (float)(n_miss + m_sector_miss) / n_access;
  return missrate;
}

void tag_array::new_window() {
  m_prev_snapshot_access = m_access;
  m_prev_snapshot_miss = m_miss;
  m_prev_snapshot_miss = m_miss + m_sector_miss;
  m_prev_snapshot_pending_hit = m_pending_hit;
}

void tag_array::print(FILE *stream, unsigned &total_access,
                      unsigned &total_misses) const {
  m_config.print(stream);
  fprintf(stream,
          "\t\tAccess = %d, Miss = %d, Sector_Miss = %d, Total_Miss = %d "
          "(%.3g), PendingHit = %d (%.3g)\n",
          m_access, m_miss, m_sector_miss, (m_miss + m_sector_miss),
          (float)(m_miss + m_sector_miss) / m_access, m_pending_hit,
          (float)m_pending_hit / m_access);
  total_misses += (m_miss + m_sector_miss);
  total_access += m_access;
}

void tag_array::get_stats(unsigned &total_access, unsigned &total_misses,
                          unsigned &total_hit_res,
                          unsigned &total_res_fail) const {
  // Update statistics from the tag array
  total_access = m_access;
  total_misses = (m_miss + m_sector_miss);
  total_hit_res = m_pending_hit;
  total_res_fail = m_res_fail;
}

bool was_write_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_REQUEST_SENT) return true;
  }
  return false;
}

bool was_writeback_sent(const std::list<cache_event> &events,
                        cache_event &wb_event) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_BACK_REQUEST_SENT) {
      wb_event = *e;
      return true;
    }
  }
  return false;
}

bool was_read_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == READ_REQUEST_SENT) return true;
  }
  return false;
}

bool was_writeallocate_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_ALLOCATE_SENT) return true;
  }
  return false;
}
/****************************************************************** MSHR
 * ******************************************************************/

/// Checks if there is a pending request to the lower memory level already
bool mshr_table::probe(new_addr_type block_addr) const {
  table::const_iterator a = m_data.find(block_addr);
  return a != m_data.end();
}

/// Checks if there is space for tracking a new memory access
bool mshr_table::full(new_addr_type block_addr) const {
  table::const_iterator i = m_data.find(block_addr);
  if (i != m_data.end())
    return i->second.m_list.size() >= m_max_merged;
  else
    return m_data.size() >= m_num_entries;
}

/// Add or merge this access
void mshr_table::add(new_addr_type mshr_addr, mem_fetch *mf, bool& is_new_entry, const char* cache_type) {
  is_new_entry = !m_data.count(mshr_addr) ? true : false; // for debug
  [[maybe_unused]] const unsigned prev_size = m_data.size(); // for debug

  m_data[mshr_addr].m_list.push_back(mf);

  assert(m_data.size() <= m_num_entries);
  assert(m_data[mshr_addr].m_list.size() <= m_max_merged);
  // indicate that this MSHR entry contains an atomic operation
  if (mf->isatomic()) {
    m_data[mshr_addr].m_has_atomic = true;
  }
}

unsigned mshr_table::occupied_slots(new_addr_type mshr_addr) {
  return m_data[mshr_addr].m_list.size();
}

/// check is_read_after_write_pending
bool mshr_table::is_read_after_write_pending(new_addr_type block_addr) {
  std::list<mem_fetch *> my_list = m_data[block_addr].m_list;
  bool write_found = false;
  for (std::list<mem_fetch *>::iterator it = my_list.begin();
       it != my_list.end(); ++it) {
    if ((*it)->is_write())  // Pending Write Request
      write_found = true;
    else if (write_found)  // Pending Read Request and we found previous Write
      return true;
  }

  return false;
}

/// Accept a new cache fill response: mark entry ready for processing
void mshr_table::mark_ready(const char* cache_name, new_addr_type block_addr, bool &has_atomic, u64 cycle) {
  assert(!busy());
  table::iterator a = m_data.find(block_addr);
  assert(a != m_data.end());
  m_lfb.push_back(block_addr);
  has_atomic = a->second.m_has_atomic;
  assert(m_lfb.size() <= m_data.size());

  if (DTRACE(REFILL_MSHR)) {
    mem_fetch* front_mf = a->second.m_list.front();
    fprintf(Trace::out, "%llu %s_sub[%d] m_lfb added %#llx\n", 
      cycle, cache_name, front_mf->get_sub_partition(), block_addr);
  }
}

/// Returns next ready access
mem_fetch* mshr_table::next_access(const char* cache_type, u64 cycle) {
  assert(access_ready());

  [[maybe_unused]] const unsigned last_occupied_entries = static_cast<unsigned>(m_data.size());
  [[maybe_unused]] const unsigned last_merged_slots = m_data[m_lfb.front()].m_list.size();

  new_addr_type block_addr = m_lfb.front();
  assert(!m_data[block_addr].m_list.empty());
  mem_fetch *result = m_data[block_addr].m_list.front();

  if (DTRACE(DUMP_MSHR)) {
    assert(last_occupied_entries == static_cast<unsigned>(m_data.size()));
    fprintf(Trace::out, "%llu Before TPC:%u SM:%u WARP:%u "
      "%s MSHR releasing entry for block_addr: 0x%llx "
      "(occupied:%u free:%u occupancy:%f)\n",
      cycle, 
      result->get_tpc(), result->get_sid(), result->get_wid(),
      cache_type, (u64)block_addr,
      static_cast<unsigned>(m_data.size()),
      m_num_entries - static_cast<unsigned>(m_data.size()),
      m_data.size() / (float)m_num_entries
    );      
    display(Trace::out, cache_type);
  }

  m_data[block_addr].m_list.pop_front();
  if (m_data[block_addr].m_list.empty()) {
    // release entry  
    m_data.erase(block_addr);
    assert(last_occupied_entries == (m_data.size() + 1));

    if (DTRACE(RELEASE_MSHR)) {
      assert(result->get_addr() == block_addr);

      // for debug backprop
      if (cycle == 6294) {
        fprintf(Trace::out, 
          "cache_type = %s is_l2 = %u m_lfb.size() = %lu\n", 
          cache_type, !strcmp(cache_type, "L2"), m_lfb.size());
      }
      if (cycle == 6294 && !strcmp(cache_type, "L2")) {
        fprintf(Trace::out, "Ready display_resp_q\n");
        display_resp_q(Trace::out, cache_type);
        fprintf(Trace::out, "Finished display_resp_q\n");
      }

      std::string whole_cache_name = cache_type;
      if (!strcmp(cache_type, "L2")) {
        whole_cache_name += " sub ";
        whole_cache_name += std::to_string(result->get_sub_partition());
      }
      fprintf(Trace::out, "%llu %s_sub[%d] m_lfb popped %#llx\n", 
        cycle, cache_type, result->get_sub_partition(), block_addr
      );   
    }

    m_lfb.pop_front();
    if (DTRACE(DUMP_MSHR)) {
      fprintf(Trace::out, "%llu After TPC:%u SM:%u WARP:%u "
        "%s MSHR releasing entry for block_addr: 0x%llx "
        "(occupied:%u free:%u occupancy:%f)\n",
        cycle,
        result->get_tpc(), result->get_sid(), result->get_wid(),
        cache_type,
        (u64)block_addr,
        static_cast<unsigned>(m_data.size()), 
        m_num_entries - static_cast<unsigned>(m_data.size()),
        static_cast<unsigned>(m_data.size()) / (float)m_num_entries
      );      
      display(Trace::out, cache_type);
    } 
  }
  return result;
}

void mshr_table::display(FILE *fp, const char* cache_type) const {
  fprintf(fp, "%s MSHR contents\n", cache_type);
  for (table::const_iterator e = m_data.begin(); e != m_data.end(); ++e) {
    u64 block_addr = e->first;
    // fprintf(fp, "%s MSHR: tag:0x%06x, atomic:%d n_merged:%zu entries: ", 
    fprintf(fp, "%s MSHR: tag:%#llx, atomic:%d n_merged:%zu entries: ", 
      cache_type, block_addr, e->second.m_has_atomic, e->second.m_list.size());
    if (!e->second.m_list.empty()) {
      mem_fetch *mf = e->second.m_list.front();
      fprintf(fp, "%p :", mf);
      mf->print(fp);
    } else {
      fprintf(fp, " no memory requests???\n");
    }
  }
}

void mshr_table::display_resp_q(FILE *fp, const char* cache_type) const {
  fprintf(fp, "%s m_lfb is:\n", cache_type);
  int index = 0;
  for (std::list<new_addr_type>::const_iterator iter = m_lfb.begin(); 
    iter != m_lfb.end(); ++iter, ++index) {
    fprintf(fp, "m_lfb[%u] = %#llx\n", index, *iter);
  }
}

/***************************************************************** Caches
 * *****************************************************************/
cache_stats::cache_stats() {
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;  
  m_l2_miss_q_pops = 0;
}

void cache_stats::clear() {
  ///
  /// Zero out all current cache statistics
  ///
  m_stats.clear();  
  
  m_overall_avg_l1d_rd_fill_to_evict_gap.clear();
  m_l1d_rd_miss_served_cycles.clear();  
  m_l1d_wr_miss_served_cycles.clear();  

  m_l1d_accesses.clear();
  m_l1d_reads.clear();
  m_l1d_writes.clear();

  m_l1d_misses.clear();
  m_l1d_rd_misses.clear();
  m_l1d_wr_misses.clear();
  m_l1d_vc_hits.clear();
  m_l1d_vc_misses.clear();
  m_l1d_vc_accesses.clear();
  m_l1d_max_evictions.clear();
  m_l1d_avg_evictions.clear();

  m_n_l1d_fill_to_evict_lines.clear();
  m_l1d_rd_byp_activates.clear();
  m_l1d_rd_byp_deactivates.clear();

  m_l2_sub_miss_served_cycles.clear();
  m_l2_sub_misses.clear();

  m_stats_pw.clear();
  m_fail_stats.clear();
  m_line_alloc_fail.clear();
  m_mshr_entry_fail.clear();
  m_miss_q_full.clear();
  m_mshr_merge_entry_fail.clear();
  m_fail_stats_total.clear();
  m_mshr_occupancy_stats.clear();
  m_accu_l2_dram_queue_size.clear();
  m_accu_l2_icnt_queue_size.clear();
  m_l2_dram_q_accesses.clear();
  m_l2_icnt_q_accesses.clear();
  m_l2_mshr_slots_fills.clear();

  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;  
  m_l2_miss_q_pops = 0;
}

void cache_stats::clear_pw() {
  ///
  /// Zero out per-window cache statistics
  ///
  m_stats_pw.clear();
}

unsigned cache_stats::get_mshr_merge_dist_cnt(
  u64 streamID, unsigned sm_id, unsigned warp_id) {
  return m_mshr_occupancy_stats[streamID][sm_id][warp_id];
}

void cache_stats::inc_mshr_stats(
  u64 streamID, unsigned sm_id, unsigned warp_id) {

  const unsigned sms = 4; // gpgpu_n_cores_per_cluster
  const unsigned max_warps_per_sm = 64; // m_config.max_warps_per_sm

  if (m_mshr_occupancy_stats.find(streamID) == m_mshr_occupancy_stats.end()) {
    std::vector<std::vector<unsigned>> new_val;
    new_val.resize(sms);
    for (unsigned sm = 0; sm < sms; ++sm) {
      new_val[sm].resize(max_warps_per_sm, 0);
    }
    m_mshr_occupancy_stats.insert(std::pair<u64,
        std::vector<std::vector<unsigned>>>(streamID, new_val));
  }
  m_mshr_occupancy_stats.at(streamID)[sm_id][warp_id]++;
}

void cache_stats::inc_accu_l2_dram_queue_size(
  u64 streamID, unsigned l2_sub, unsigned size) {

  if (m_accu_l2_dram_queue_size.find(streamID) == m_accu_l2_dram_queue_size.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(get_sub_partitions());
    m_accu_l2_dram_queue_size.insert(std::pair<u64,
        std::vector<unsigned>>(streamID, new_val));
  }
  m_accu_l2_dram_queue_size.at(streamID)[l2_sub] += size;
}
void cache_stats::inc_accu_l2_icnt_queue_size(
  u64 streamID, unsigned l2_sub, unsigned size) {

  if (m_accu_l2_icnt_queue_size.find(streamID) == m_accu_l2_icnt_queue_size.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(get_sub_partitions());
    m_accu_l2_icnt_queue_size.insert(std::pair<u64,
        std::vector<unsigned>>(streamID, new_val));
  }
  m_accu_l2_icnt_queue_size.at(streamID)[l2_sub] += size;
}
void cache_stats::inc_l2_dram_q_accesses(u64 streamID, unsigned l2_sub) {

  if (m_l2_dram_q_accesses.find(streamID) == m_l2_dram_q_accesses.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(get_sub_partitions());
    m_l2_dram_q_accesses.insert(std::pair<u64,
        std::vector<unsigned>>(streamID, new_val));
  }
  m_l2_dram_q_accesses.at(streamID)[l2_sub]++;
}
void cache_stats::inc_l2_icnt_q_accesses(u64 streamID, unsigned l2_sub) {

  if (m_l2_icnt_q_accesses.find(streamID) == m_l2_icnt_q_accesses.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(get_sub_partitions());
    m_l2_icnt_q_accesses.insert(std::pair<u64,
        std::vector<unsigned>>(streamID, new_val));
  }
  m_l2_icnt_q_accesses.at(streamID)[l2_sub]++;
}

void cache_stats::inc_l2_mshr_slots_fills(u64 streamID, unsigned l2_sub) {

  if (m_l2_mshr_slots_fills.find(streamID) == m_l2_mshr_slots_fills.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(get_sub_partitions());
    m_l2_mshr_slots_fills.insert(std::pair<u64,
        std::vector<unsigned>>(streamID, new_val));
  }
  m_l2_mshr_slots_fills.at(streamID)[l2_sub]++;
}

void cache_stats::overall_average_l1d_rd_fill_to_evict_gap(
  u64 streamID, u64 served_cycles) {
  if (m_overall_avg_l1d_rd_fill_to_evict_gap.find(streamID) == m_overall_avg_l1d_rd_fill_to_evict_gap.end()) {
    u64 new_val;
    m_overall_avg_l1d_rd_fill_to_evict_gap.insert(
      std::pair<u64, u64>(streamID, new_val));    
  }
  [[maybe_unused]] u64 last_avg_gap = 
    m_overall_avg_l1d_rd_fill_to_evict_gap.at(streamID); // for debug

  m_overall_avg_l1d_rd_fill_to_evict_gap.at(streamID) = 
    (m_overall_avg_l1d_rd_fill_to_evict_gap.at(streamID) + served_cycles) >> 1;  
}
void cache_stats::avg_l1d_rd_miss_served_cycles(u64 streamID, u64 served_cycles) {
  if (m_l1d_rd_miss_served_cycles.find(streamID) == m_l1d_rd_miss_served_cycles.end()) {
    m_l1d_rd_miss_served_cycles[streamID] = served_cycles;    
  } else {
    m_l1d_rd_miss_served_cycles[streamID] = 
      (m_l1d_rd_miss_served_cycles[streamID] + served_cycles) >> 1;
  }
}
void cache_stats::avg_l1d_wr_miss_served_cycles(u64 streamID, u64 served_cycles) {
  if (m_l1d_wr_miss_served_cycles.find(streamID) == m_l1d_wr_miss_served_cycles.end()) {
    m_l1d_wr_miss_served_cycles[streamID] = served_cycles;
  } else {
    m_l1d_wr_miss_served_cycles[streamID] = 
      (m_l1d_wr_miss_served_cycles[streamID] + served_cycles) >> 1;
  }
}

void cache_stats::inc_l1d_accesses(u64 streamID, u32 kernel) {
  m_l1d_accesses[streamID][kernel]++;
}
void cache_stats::inc_l1d_reads(u64 streamID, u32 kernel) {
  m_l1d_reads[streamID][kernel]++;
}
void cache_stats::inc_l1d_rd_misses(u64 streamID, u32 kernel) {
  m_l1d_rd_misses[streamID][kernel]++;
}
void cache_stats::inc_l1d_writes(u64 streamID, u32 kernel) {
  m_l1d_writes[streamID][kernel]++;
}
void cache_stats::inc_l1d_wr_misses(u64 streamID, u32 kernel) {
  m_l1d_wr_misses[streamID][kernel]++;
}
void cache_stats::inc_l1d_vc_hits(LOCALITY_KEY loc_key) {
  m_l1d_vc_hits[loc_key]++;
}
void cache_stats::inc_l1d_vc_misses(LOCALITY_KEY loc_key) {
  m_l1d_vc_misses[loc_key]++;
}
void cache_stats::inc_l1d_vc_accesses(LOCALITY_KEY loc_key) {
  m_l1d_vc_accesses[loc_key]++;
}

void cache_stats::update_l1d_max_evictions(const LOCALITY_KEY& loc_key, u32 n_evictions) {
  u32 prev_max_evictions = m_l1d_max_evictions[loc_key];
  m_l1d_max_evictions[loc_key] = std::max(prev_max_evictions, n_evictions);
  if (DTRACE(DEBUG_L1D_MAX_EVICTIONS)) {
    fprintf(Trace::out, "update_l1d_max_evictions for loc_key:<streamID:%llu, kernel:%u> "
      "n_evictions:%u, prev_max_evictions:%u, new_max_evictions:%u\n",
      loc_key.stream_id, loc_key.kernel, n_evictions, prev_max_evictions, m_l1d_max_evictions[loc_key]);
  }  
}
void cache_stats::update_l1d_avg_evictions(const LOCALITY_KEY& loc_key, u32 n_evictions) {
  u32 prev_avg_evictions = m_l1d_avg_evictions[loc_key];
  m_l1d_avg_evictions[loc_key] = (prev_avg_evictions + n_evictions) >> 1;
  if (DTRACE(DEBUG_L1D_AVG_EVICTIONS)) {
    fprintf(Trace::out, "update_l1d_avg_evictions for loc_key:<streamID:%llu, kernel:%u> "
      "n_evictions:%u, prev_avg_evictions:%u, new_avg_evictions:%u\n",
      loc_key.stream_id, loc_key.kernel, n_evictions, prev_avg_evictions, m_l1d_avg_evictions[loc_key]);
  }
}

void cache_stats::update_n_l1d_fill_to_evict(const LOCALITY_KEY& loc_key, u32 n_lines) {
  m_n_l1d_fill_to_evict_lines[loc_key] = n_lines;
}

void cache_stats::inc_l2_sub_miss_served_cycles(
  u64 streamID, unsigned l2_sub, u64 served_cycles) {
  if (m_l2_sub_miss_served_cycles.find(streamID) == m_l2_sub_miss_served_cycles.end()) {
    std::vector<u64> new_val;
    new_val.resize(get_sub_partitions());
    m_l2_sub_miss_served_cycles.insert(
      std::pair<u64, std::vector<u64>>(streamID, new_val));    
  }
  m_l2_sub_miss_served_cycles.at(streamID)[l2_sub] += served_cycles;
}
void cache_stats::inc_l2_sub_misses(u64 streamID, unsigned l2_sub) {
  if (m_l2_sub_misses.find(streamID) == m_l2_sub_misses.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(get_sub_partitions());
    m_l2_sub_misses.insert(std::pair<u64, std::vector<unsigned>>(streamID, new_val));    
  }
  m_l2_sub_misses.at(streamID)[l2_sub]++;
}

void cache_stats::inc_l2_miss_q_pops() {
  m_l2_miss_q_pops++;
}

void cache_stats::gather_lines_stats(u64 streamID, unsigned unfolded_index) {
  if (m_lines_evictons.find(streamID) == m_lines_evictons.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(1); 
  }
}

void cache_stats::update_l1d_rd_byp_act(
  bool en, u64 block_addr, u32 activates, u64 streamID) {
  if (en) {
    if (m_l1d_rd_byp_activates.find(streamID) == m_l1d_rd_byp_activates.end()) {
      std::map<u64, u32> new_record;
      new_record[block_addr] = activates;
      m_l1d_rd_byp_activates[streamID] = new_record;
    } else {
      std::map<u64, u32>& record = m_l1d_rd_byp_activates.at(streamID);
      record[block_addr] = activates;
    }
  }
}
void cache_stats::update_l1d_rd_byp_deact(
  bool en, u64 block_addr, u32 deactivates, u64 streamID) {
  if (en) {
    if (m_l1d_rd_byp_deactivates.find(streamID) == m_l1d_rd_byp_deactivates.end()) {
      std::map<u64, u32> new_record;
      new_record[block_addr] = deactivates;
      m_l1d_rd_byp_deactivates[streamID] = new_record;
    } else {
      std::map<u64, u32>& record = m_l1d_rd_byp_deactivates.at(streamID);
      if (record.find(block_addr) == record.end()) {
        record[block_addr] = deactivates;
      } else {
        record[block_addr] += deactivates;
      }
    }
  }
}

void cache_stats::inc_stats(int access_type, int access_outcome, u64 streamID) {
  ///
  /// Increment the stat corresponding to (access_type, access_outcome) by 1.
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");

  if (m_stats.find(streamID) == m_stats.end()) {
    std::vector<std::vector<u64>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0);
    }
    m_stats.insert(std::pair<u64, std::vector<std::vector<u64>>>(streamID, new_val));
  }
  m_stats.at(streamID)[access_type][access_outcome]++;
}

void cache_stats::update_evict_stats(u64 streamID, u64 victim_avg_evict_interval) {
  
  if (m_evict_stats.find(streamID) == m_evict_stats.end()) {
    u32 new_val = 0;
    m_evict_stats.insert(std::pair<u64, u32>(streamID, new_val));
  }

  m_evict_stats.at(streamID) = 
  (m_evict_stats.at(streamID) + victim_avg_evict_interval) >> 1;
}
// void cache_stats::update_bypass_stats(
//     u64 streamID, 
//     u64 new_bypass) {
  
//   if (m_evict_stats.find(streamID) == m_evict_stats.end()) {
//     unsigned new_val;
//     m_evict_stats.insert(std::pair<u64, unsigned>(streamID, new_val));
//   }

//   m_evict_stats.at(streamID) = 
//   (m_evict_stats.at(streamID) + victim_avg_evict_interval) >> 1;
// }

void cache_stats::inc_stats_pw(int access_type, int access_outcome,
                               u64 streamID) {
  ///
  /// Increment the corresponding per-window cache stat
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");

  if (m_stats_pw.find(streamID) == m_stats_pw.end()) {
    std::vector<std::vector<u64>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0);
    }
    m_stats_pw.insert(std::pair<u64,
                                std::vector<std::vector<u64>>>(
        streamID, new_val));
  }
  m_stats_pw.at(streamID)[access_type][access_outcome]++;
}

void cache_stats::inc_fail_stats(
  int access_type, int fail_outcome, 
  u64 streamID, int fail_driver) {

  if (!check_fail_valid(access_type, fail_outcome))
    assert(0 && "Unknown cache access type or access fail");

  if (m_fail_stats.find(streamID) == m_fail_stats.end()) {
    std::vector<std::vector<u64>> new_val;    
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_RESERVATION_FAIL_STATUS, 0);
    }
        
    m_fail_stats.insert(std::pair<u64,
                        std::vector<std::vector<u64>>>(
                        streamID, new_val));
    m_fail_stats_total.insert(
      std::pair<u64, std::vector<u64>>(
        streamID, std::vector<u64>(NUM_MEM_ACCESS_TYPE, 0)));
  } // if (m_fail_stats.find(streamID) == m_fail_stats.end()) { ---> Create new entry
  m_fail_stats.at(streamID)[access_type][fail_outcome]++;
  m_fail_stats_total.at(streamID)[access_type]++;

  if (m_line_alloc_fail.find(streamID) == m_line_alloc_fail.end()) {
    std::vector<std::vector<u64>> new_line_alloc_fail_driver;
    new_line_alloc_fail_driver.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_line_alloc_fail_driver[j].resize(NUM_LINE_ALLOC_FAIL_DRIVER, 0);
    }    
    m_line_alloc_fail.insert(std::pair<u64,
      std::vector<std::vector<u64>>>(streamID, new_line_alloc_fail_driver));
  }
  if (m_mshr_entry_fail.find(streamID) == m_mshr_entry_fail.end()) {
    std::vector<std::vector<u64>> new_mshr_entry_fail_driver;
    new_mshr_entry_fail_driver.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_mshr_entry_fail_driver[j].resize(NUM_MSHR_ENTRY_FAIL_DRIVER, 0);
    }    
    m_mshr_entry_fail.insert(std::pair<u64,
      std::vector<std::vector<u64>>>(streamID, new_mshr_entry_fail_driver));
  }
  if (m_miss_q_full.find(streamID) == m_miss_q_full.end()) {
    std::vector<std::vector<u64>> new_miss_q_full_driver;
    new_miss_q_full_driver.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_miss_q_full_driver[j].resize(NUM_MISS_QUEUE_FULL_DRIVER, 0);
    }    
    m_miss_q_full.insert(std::pair<u64,
      std::vector<std::vector<u64>>>(streamID, new_miss_q_full_driver));
  }
  if (m_mshr_merge_entry_fail.find(streamID) == m_mshr_merge_entry_fail.end()) {
    std::vector<std::vector<u64>> new_mshr_merge_entry_fail_driver;
    new_mshr_merge_entry_fail_driver.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_mshr_merge_entry_fail_driver[j].resize(NUM_MSHR_MERGE_ENTRY_FAIL_DRIVER, 0);
    }    
    m_mshr_merge_entry_fail.insert(std::pair<u64,
      std::vector<std::vector<u64>>>(streamID, new_mshr_merge_entry_fail_driver));
  }    

  if (fail_driver != -1) {
    if (static_cast<cache_reservation_fail_reason>(fail_outcome) == LINE_ALLOC_FAIL) {
      m_line_alloc_fail.at(streamID)[access_type][fail_driver]++;
      if (DTRACE(LINE_ALLOC_FAIL_DRIVER)) {
        fprintf(Trace::out, "m_line_alloc_fail[%s][LINE_ALLOC_FAIL][%s]++\n",
          mem_access_type_str(static_cast<mem_access_type>(access_type)),
          line_alloc_fail_driver_str(static_cast<line_alloc_fail_driver>(fail_driver)));
      }
    } else if (static_cast<cache_reservation_fail_reason>(fail_outcome) == MSHR_ENTRY_FAIL) {
      m_mshr_entry_fail.at(streamID)[access_type][fail_driver]++;
      if (DTRACE(MSHR_ENTRY_FAIL_DRIVER)) {
        fprintf(Trace::out, "m_mshr_entry_fail[%s][MSHR_ENTRY_FAIL][%s]++\n",
          mem_access_type_str(static_cast<mem_access_type>(access_type)),
          mshr_entry_fail_driver_str(static_cast<mshr_entry_fail_driver>(fail_driver)));
      }      
    } else if (static_cast<cache_reservation_fail_reason>(fail_outcome) == MISS_QUEUE_FULL) {
      m_miss_q_full.at(streamID)[access_type][fail_driver]++;
      if (DTRACE(MISS_QUEUE_FULL_DRIVER)) {
        fprintf(Trace::out, "m_miss_q_full[%s][MISS_QUEUE_FULL][%s]++\n",
          mem_access_type_str(static_cast<mem_access_type>(access_type)),
          miss_queue_full_driver_str(static_cast<miss_queue_full_driver>(fail_driver)));
      }      
    } else if (static_cast<cache_reservation_fail_reason>(fail_outcome) == MSHR_MERGE_ENTRY_FAIL) {
      m_mshr_merge_entry_fail.at(streamID)[access_type][fail_driver]++;
      if (DTRACE(MSHR_MERGE_FAIL_DRIVER)) {
        fprintf(Trace::out, "m_mshr_merge_entry_fail[%s][MSHR_MERGE_ENTRY_FAIL][%s]++\n",
          mem_access_type_str(static_cast<mem_access_type>(access_type)),
          mshr_merge_entry_fail_driver_str(static_cast<mshr_merge_entry_fail_driver>(fail_driver)));
      }
    }

  }
}

enum cache_request_status cache_stats::select_stats_status(
    enum cache_request_status probe, enum cache_request_status access) const {
  ///
  /// This function selects how the cache access outcome should be counted.
  /// HIT_RESERVED is considered as a MISS in the cores, however, it should be
  /// counted as a HIT_RESERVED in the caches.
  ///

  if (probe == HIT_RESERVED && access != RESERVATION_FAIL) {
    return probe;
  } else if ((probe == SECTOR_MISS) && (access == MISS)) {
    return probe;
  } else {
    return access;
  }    
}

u64 &cache_stats::operator()(int access_type, int access_outcome,
                                            bool fail_outcome,
                                            u64 streamID) {
  ///
  /// Simple method to read/modify the stat corresponding to (access_type,
  /// access_outcome) Used overloaded () to avoid the need for separate
  /// read/write member functions
  ///
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome");

    return m_fail_stats.at(streamID)[access_type][access_outcome];
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome");

    return m_stats.at(streamID)[access_type][access_outcome];
  }
}

u64 cache_stats::operator()(
  int access_type, 
  int access_outcome,
  bool is_fail_outcome,  
  int fail_driver, u64 streamID) const {

  assert(is_fail_outcome == true);
  if (static_cast<cache_reservation_fail_reason>(access_outcome) == LINE_ALLOC_FAIL) {
    return m_line_alloc_fail.at(streamID)[access_type][fail_driver];     
  } else if (static_cast<cache_reservation_fail_reason>(access_outcome) == MSHR_ENTRY_FAIL) {
    return m_mshr_entry_fail.at(streamID)[access_type][fail_driver];
  } else if (static_cast<cache_reservation_fail_reason>(access_outcome) == MISS_QUEUE_FULL) {
    return m_miss_q_full.at(streamID)[access_type][fail_driver];     
  } else if (static_cast<cache_reservation_fail_reason>(access_outcome) == MSHR_MERGE_ENTRY_FAIL) {
    return m_mshr_merge_entry_fail.at(streamID)[access_type][fail_driver];
  } else {
    assert(0);
    return (u64) - 1;
  }
}

u64 cache_stats::operator()(int access_type, int access_outcome,
                                           bool fail_outcome,
                                           u64 streamID) const {
  ///
  /// Const accessor into m_stats.
  ///
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome)) {
      assert(0 && "Unknown cache access type or fail outcome");
    }      
    return m_fail_stats.at(streamID)[access_type][access_outcome];
  } else {
    if (!check_valid(access_type, access_outcome)) {
      assert(0 && "Unknown cache access type or access outcome");
    }
    return m_stats.at(streamID)[access_type][access_outcome];
  }
}

u64 cache_stats::operator()(
  unsigned sm, unsigned warp, u64 streamID) const {
  if (!check_valid(sm, warp)) {
    assert(0 && "Unknown sm_id or warp_id");
  }
  return m_mshr_occupancy_stats.at(streamID)[sm][warp];
}

unsigned cache_stats::operator()(
  unsigned l2_sub, u64 streamID) const {
  auto it = m_accu_l2_dram_queue_size.find(streamID);
  if (it == m_accu_l2_dram_queue_size.end()) return 0;
  if (l2_sub >= it->second.size()) return 0;
  return it->second[l2_sub];
}
u64 cache_stats::operator()(
  unsigned l2_sub, u64 streamID, const char* tgt_name) const {
  if (!strcmp(tgt_name, "m_l2_sub_miss_served_cycles")) {
    auto it = m_l2_sub_miss_served_cycles.find(streamID);
    if (it != m_l2_sub_miss_served_cycles.end()) {
      return it->second[l2_sub];
    }
    return 0;
  } else if (!strcmp(tgt_name, "m_l2_sub_misses")) {
    auto it = m_l2_sub_misses.find(streamID);
    if (it != m_l2_sub_misses.end()) {
      return it->second[l2_sub];
    } 
    return 0;    
  } else {
    assert(0);
    return (u64) - 1;
  }
}

u32 cache_stats::getU32(u64 streamID, u32 kernel, const char* tgt_name) const {
  // m_l1d_accesses
  // m_l1d_reads
  // m_l1d_writes
  // m_l1d_misses
  // m_l1d_rd_misses
  // m_l1d_wr_misses
  if (!strcmp(tgt_name, "m_l1d_accesses")) {
    auto it_stream = m_l1d_accesses.find(streamID);
    if (it_stream == m_l1d_accesses.end()) {
      return 0;
    } else {
      auto it_kernel = it_stream->second.find(kernel);
      if (it_kernel == it_stream->second.end()) {
        return 0;
      } else {
        return it_kernel->second;
      }
    }
  }
  else if (!strcmp(tgt_name, "m_l1d_reads")) {
    auto it_stream = m_l1d_reads.find(streamID);
    if (it_stream == m_l1d_reads.end()) {
      return 0;
    } else {
      auto it_kernel = it_stream->second.find(kernel);
      if (it_kernel == it_stream->second.end()) {
        return 0;
      } else {
        return it_kernel->second;
      }
    }
  }  
  else if (!strcmp(tgt_name, "m_l1d_writes")) {
    auto it_stream = m_l1d_writes.find(streamID);
    if (it_stream == m_l1d_writes.end()) {
      return 0;
    } else {
      auto it_kernel = it_stream->second.find(kernel);
      if (it_kernel == it_stream->second.end()) {
        return 0;
      } else {
        return it_kernel->second;
      }
    }
  }    
  else if (!strcmp(tgt_name, "m_l1d_misses")) {
    auto it_stream = m_l1d_misses.find(streamID);
    if (it_stream == m_l1d_misses.end()) {
      return 0;
    } else {
      auto it_kernel = it_stream->second.find(kernel);
      if (it_kernel == it_stream->second.end()) {
        return 0;
      } else {
        return it_kernel->second;
      }
    }
  }   
  else if (!strcmp(tgt_name, "m_l1d_rd_misses")) {
    auto it_stream = m_l1d_rd_misses.find(streamID);
    if (it_stream == m_l1d_rd_misses.end()) {
      return 0;
    } else {
      auto it_kernel = it_stream->second.find(kernel);
      if (it_kernel == it_stream->second.end()) {
        return 0;
      } else {
        return it_kernel->second;
      }
    }
  }
  else if (!strcmp(tgt_name, "m_l1d_wr_misses")) {
    auto it_stream = m_l1d_wr_misses.find(streamID);
    if (it_stream == m_l1d_wr_misses.end()) {
      return 0;
    } else {
      auto it_kernel = it_stream->second.find(kernel);
      if (it_kernel == it_stream->second.end()) {
        return 0;
      } else {
        return it_kernel->second;
      }
    }
  } else {
    assert(0 && "Unknown cache stat name");
    return 0;
  }
}

u64 cache_stats::operator()(u64 streamID, const char* tgt_name) const {
// m_overall_avg_l1d_rd_fill_to_evict_gap
// m_l1d_rd_miss_served_cycles
// m_l1d_wr_miss_served_cycles
// m_l2_sub_miss_served_cycles
  if (!strcmp(tgt_name, "m_overall_avg_l1d_rd_fill_to_evict_gap")) {
    auto it = m_overall_avg_l1d_rd_fill_to_evict_gap.find(streamID);
    if (it == m_overall_avg_l1d_rd_fill_to_evict_gap.end()) {
      return 0;
    } else {
      return it->second;
    }
  }
  else if (!strcmp(tgt_name, "m_l1d_rd_miss_served_cycles")) {
    auto it = m_l1d_rd_miss_served_cycles.find(streamID);
    if (it == m_l1d_rd_miss_served_cycles.end()) {
      return 0;
    } else {
      return it->second;
    }
  }
  else if (!strcmp(tgt_name, "m_l1d_wr_miss_served_cycles")) {
    auto it = m_l1d_wr_miss_served_cycles.find(streamID);
    if (it == m_l1d_wr_miss_served_cycles.end()) {
      return 0;
    } else {
      return it->second;
    }
  }
  assert(0);
  return (u64) - 1;  
}

cache_stats cache_stats::operator+(const cache_stats &cs) {
  ///
  /// Overloaded + operator to allow for simple stat accumulation
  ///
  // 1-1 Init
  cache_stats ret;
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_stats.insert(std::pair<u64,
      std::vector<std::vector<u64>>>(streamID, m_stats.at(streamID)));
  }
  for (auto iter = m_stats_pw.begin(); iter != m_stats_pw.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_stats_pw.insert(std::pair<u64,
      std::vector<std::vector<u64>>>(streamID, m_stats_pw.at(streamID)));
  }

  for (auto iter = m_fail_stats.begin(); iter != m_fail_stats.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_fail_stats.insert(std::pair<u64,
      std::vector<std::vector<u64>>>(streamID, m_fail_stats.at(streamID)));
  }
  for (auto iter = m_line_alloc_fail.begin(); iter != m_line_alloc_fail.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_line_alloc_fail.insert(std::pair<u64,
      std::vector<std::vector<u64>>>(streamID, m_line_alloc_fail.at(streamID)));
  }  
  for (auto iter = m_mshr_entry_fail.begin(); iter != m_mshr_entry_fail.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_mshr_entry_fail.insert(std::pair<u64,
      std::vector<std::vector<u64>>>(streamID, m_mshr_entry_fail.at(streamID)));
  }    
  for (auto iter = m_miss_q_full.begin(); iter != m_miss_q_full.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_miss_q_full.insert(std::pair<u64, 
      std::vector<std::vector<u64>>>(streamID, m_miss_q_full.at(streamID)));
  }  
  for (auto iter = m_mshr_merge_entry_fail.begin(); iter != m_mshr_merge_entry_fail.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_mshr_merge_entry_fail.insert(std::pair<u64,
      std::vector<std::vector<u64>>>(streamID, m_mshr_merge_entry_fail.at(streamID)));
  }  
  for (auto iter = m_fail_stats_total.begin(); iter != m_fail_stats_total.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_fail_stats_total.insert(
      std::pair<u64, std::vector<u64>>(streamID, m_fail_stats_total.at(streamID)));     
  }
  for (auto iter = m_mshr_occupancy_stats.begin(); iter != m_mshr_occupancy_stats.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_mshr_occupancy_stats.insert(
      std::pair<u64,
        std::vector<std::vector<u32>>>(streamID, m_mshr_occupancy_stats.at(streamID)));
  }     
  for (auto iter = m_overall_avg_l1d_rd_fill_to_evict_gap.begin(); 
    iter != m_overall_avg_l1d_rd_fill_to_evict_gap.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_overall_avg_l1d_rd_fill_to_evict_gap.insert(
      std::pair<u64, u64>(
        streamID, m_overall_avg_l1d_rd_fill_to_evict_gap.at(streamID)));
  }
  for (auto iter = m_l1d_rd_miss_served_cycles.begin(); 
    iter != m_l1d_rd_miss_served_cycles.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_l1d_rd_miss_served_cycles.insert(
      std::pair<u64, u64>(streamID, m_l1d_rd_miss_served_cycles.at(streamID)));
  }    
  for (auto iter = m_l1d_wr_miss_served_cycles.begin(); 
    iter != m_l1d_wr_miss_served_cycles.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_l1d_wr_miss_served_cycles.insert(
      std::pair<u64, u64>(streamID, m_l1d_wr_miss_served_cycles.at(streamID)));
  }     

  for (auto iter = m_l1d_rd_byp_activates.begin(); 
    iter != m_l1d_rd_byp_activates.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_l1d_rd_byp_activates[streamID] = m_l1d_rd_byp_activates.at(streamID);
  }  
  for (auto iter = m_l1d_rd_byp_deactivates.begin(); 
    iter != m_l1d_rd_byp_deactivates.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_l1d_rd_byp_deactivates[streamID] = m_l1d_rd_byp_deactivates.at(streamID);
  }

  for (auto iter = m_l1d_misses.begin(); 
    iter != m_l1d_misses.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_l1d_misses[streamID] = m_l1d_misses.at(streamID);
  }
  for (auto iter = m_l1d_reads.begin(); iter != m_l1d_reads.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_l1d_reads[streamID] = m_l1d_reads.at(streamID);
  }    
  for (auto iter = m_l1d_rd_misses.begin(); iter != m_l1d_rd_misses.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_l1d_rd_misses[streamID] = m_l1d_rd_misses.at(streamID);
  }
  for (auto iter = m_l1d_wr_misses.begin(); iter != m_l1d_wr_misses.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_l1d_wr_misses[streamID] = m_l1d_wr_misses.at(streamID);
  }
  for (auto iter = m_l1d_vc_hits.begin(); iter != m_l1d_vc_hits.end(); ++iter) {
    LOCALITY_KEY key = iter->first;
    ret.m_l1d_vc_hits[key] = m_l1d_vc_hits.at(key);
  }
  for (auto iter = m_l1d_vc_misses.begin(); iter != m_l1d_vc_misses.end(); ++iter) {
    LOCALITY_KEY key = iter->first;
    ret.m_l1d_vc_misses[key] = m_l1d_vc_misses.at(key);
  }
  for (auto iter = m_l1d_vc_accesses.begin(); iter != m_l1d_vc_accesses.end(); ++iter) {
    LOCALITY_KEY key = iter->first;
    ret.m_l1d_vc_accesses[key] = m_l1d_vc_accesses.at(key);
  }  
  for (auto iter = m_l1d_max_evictions.begin(); iter != m_l1d_max_evictions.end(); ++iter) {
    LOCALITY_KEY loc_key = iter->first;
    ret.m_l1d_max_evictions[loc_key] = m_l1d_max_evictions.at(loc_key);
  }
  for (auto iter = m_l1d_avg_evictions.begin(); iter != m_l1d_avg_evictions.end(); ++iter) {
    LOCALITY_KEY loc_key = iter->first;
    ret.m_l1d_avg_evictions[loc_key] = m_l1d_avg_evictions.at(loc_key);
  }

  for (auto iter = m_n_l1d_fill_to_evict_lines.begin();
    iter != m_n_l1d_fill_to_evict_lines.end(); ++iter) {
    const LOCALITY_KEY key = iter->first;
    ret.m_n_l1d_fill_to_evict_lines[key] = m_n_l1d_fill_to_evict_lines.at(key);
  }

  for (auto iter = m_l2_sub_miss_served_cycles.begin(); 
    iter != m_l2_sub_miss_served_cycles.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_l2_sub_miss_served_cycles.insert(
      std::pair<u64, std::vector<u64>>(streamID, m_l2_sub_miss_served_cycles.at(streamID)));
  }  
  for (auto iter = m_l2_sub_misses.begin(); 
    iter != m_l2_sub_misses.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_l2_sub_misses.insert(
      std::pair<u64, std::vector<u32>>(streamID, m_l2_sub_misses.at(streamID)));
  }    
  for (auto iter = m_accu_l2_dram_queue_size.begin(); iter != m_accu_l2_dram_queue_size.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_accu_l2_dram_queue_size.insert(
      std::pair<u64, std::vector<u32>>(streamID, m_accu_l2_dram_queue_size.at(streamID)));
  }
  for (auto iter = m_accu_l2_icnt_queue_size.begin(); iter != m_accu_l2_icnt_queue_size.end(); ++iter) {
    u64 streamID = iter->first;
    ret.m_accu_l2_icnt_queue_size.insert(
      std::pair<u64, std::vector<u32>>(streamID, m_accu_l2_icnt_queue_size.at(streamID)));
  }

  // 1-2 Overload "+"
  for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) {
    u64 streamID = iter->first;
    if (ret.m_stats.find(streamID) == ret.m_stats.end()) {
      ret.m_stats.insert(
        std::pair<u64, std::vector<std::vector<u64>>>(streamID, cs.m_stats.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          ret.m_stats.at(streamID)[type][status] += cs(type, status, false, streamID);
        }
      }
    }
  }

  for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) {
    u64 streamID = iter->first;
    if (ret.m_stats.find(streamID) == ret.m_stats.end()) {
      ret.m_stats.insert(
        std::pair<u64, std::vector<std::vector<u64>>>(streamID, cs.m_stats.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          ret.m_stats.at(streamID)[type][status] += cs(type, status, false, streamID);
        }
      }
    }
  }

  for (auto iter = cs.m_l1d_rd_byp_activates.begin(); 
    iter != cs.m_l1d_rd_byp_activates.end(); ++iter) {
    u64 streamID = iter->first;
    if (ret.m_l1d_rd_byp_activates.find(streamID) == ret.m_l1d_rd_byp_activates.end()) {
      ret.m_l1d_rd_byp_activates[streamID] = cs.m_l1d_rd_byp_activates.at(streamID);
    } else {
      std::map<u64, u32>& recorded = m_l1d_rd_byp_activates.at(streamID);
      std::map<u64, u32> in_coming = cs.m_l1d_rd_byp_activates.at(streamID);
      for (auto& iter : in_coming)
      {
        const u64 block_addr = iter.first;
        recorded[block_addr] += iter.second;
      }
    }
  }
  for (auto iter = cs.m_l1d_rd_byp_deactivates.begin(); 
    iter != cs.m_l1d_rd_byp_deactivates.end(); ++iter) {
    u64 streamID = iter->first;
    if (ret.m_l1d_rd_byp_deactivates.find(streamID) == ret.m_l1d_rd_byp_deactivates.end()) {
      ret.m_l1d_rd_byp_deactivates[streamID] = cs.m_l1d_rd_byp_deactivates.at(streamID);
    } else {
      std::map<u64, u32>& recorded = m_l1d_rd_byp_deactivates.at(streamID);
      std::map<u64, u32> in_coming = cs.m_l1d_rd_byp_deactivates.at(streamID);
      for (auto& iter : in_coming)
      {
        const u64 block_addr = iter.first;
        recorded[block_addr] += iter.second;
      }
    }
  }

  for (auto iter = cs.m_stats_pw.begin(); iter != cs.m_stats_pw.end(); ++iter) {
    u64 streamID = iter->first;
    if (ret.m_stats_pw.find(streamID) == ret.m_stats_pw.end()) {
      ret.m_stats_pw.insert(std::pair<u64,
          std::vector<std::vector<u64>>>(streamID, cs.m_stats_pw.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          ret.m_stats_pw.at(streamID)[type][status] += cs(type, status, false, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_fail_stats.begin(); iter != cs.m_fail_stats.end(); ++iter) {
    u64 streamID = iter->first;
    if (ret.m_fail_stats.find(streamID) == ret.m_fail_stats.end()) {
      ret.m_fail_stats.insert(
        std::pair<u64,
            std::vector<std::vector<u64>>>(streamID, cs.m_fail_stats.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS; ++status) {
          ret.m_fail_stats.at(streamID)[type][status] += cs(type, status, true, streamID);
          ret.m_fail_stats_total.at(streamID)[type] += cs(type, status, true, streamID);          
        }        
      }
    }
  }
  for (auto iter = cs.m_mshr_occupancy_stats.begin(); iter != cs.m_mshr_occupancy_stats.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_mshr_occupancy_stats.find(streamID) == ret.m_mshr_occupancy_stats.end()) {
      ret.m_mshr_occupancy_stats.insert(
        std::pair<u64,
        std::vector<std::vector<unsigned>>>(streamID, cs.m_mshr_occupancy_stats.at(streamID)));
    } else {
      const unsigned sms = 4; // gpgpu_n_cores_per_cluster
      const unsigned max_warps_per_sm = 64; // m_config.max_warps_per_sm
      for (unsigned sm = 0; sm < sms; sm++) {
        for (unsigned warp = 0; warp < max_warps_per_sm; warp++) {
          ret.m_mshr_occupancy_stats.at(streamID)[sm][warp] += cs(sm, warp, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_overall_avg_l1d_rd_fill_to_evict_gap.begin(); 
    iter != cs.m_overall_avg_l1d_rd_fill_to_evict_gap.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_overall_avg_l1d_rd_fill_to_evict_gap.find(streamID) == 
      ret.m_overall_avg_l1d_rd_fill_to_evict_gap.end()) {
      ret.m_overall_avg_l1d_rd_fill_to_evict_gap.insert(
        std::pair<u64, u64>(
          streamID, cs.m_overall_avg_l1d_rd_fill_to_evict_gap.at(streamID)));
    } else {
        ret.m_overall_avg_l1d_rd_fill_to_evict_gap.at(streamID) += cs.m_overall_avg_l1d_rd_fill_to_evict_gap.at(streamID);
    }
  }   
  for (auto iter = cs.m_l1d_rd_miss_served_cycles.begin(); 
    iter != cs.m_l1d_rd_miss_served_cycles.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_l1d_rd_miss_served_cycles.find(streamID) == 
      ret.m_l1d_rd_miss_served_cycles.end()) {
      ret.m_l1d_rd_miss_served_cycles.insert(
        std::pair<u64, u64>(
          streamID, cs.m_l1d_rd_miss_served_cycles.at(streamID)));
    } else {
        ret.m_l1d_rd_miss_served_cycles.at(streamID) += cs.m_l1d_rd_miss_served_cycles.at(streamID);      
    }
  }  
  for (auto iter = cs.m_l1d_wr_miss_served_cycles.begin(); 
    iter != cs.m_l1d_wr_miss_served_cycles.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_l1d_wr_miss_served_cycles.find(streamID) == 
      ret.m_l1d_wr_miss_served_cycles.end()) {
      ret.m_l1d_wr_miss_served_cycles[streamID] = cs.m_l1d_wr_miss_served_cycles.at(streamID);
    } else {      
      ret.m_l1d_wr_miss_served_cycles[streamID] += cs.m_l1d_wr_miss_served_cycles.at(streamID);      
    }
  }    
  for (auto iter = cs.m_l1d_misses.begin(); iter != cs.m_l1d_misses.end(); ++iter) {
    u64 streamID = iter->first;
    if (ret.m_l1d_misses.find(streamID) == ret.m_l1d_misses.end()) {
      ret.m_l1d_misses[streamID] = cs.m_l1d_misses.at(streamID);
    } else {
      std::map<u32 /* kernel */, u32> in_coming = cs.m_l1d_misses.at(streamID);
      std::map<u32 /* kernel */, u32>& recorded = ret.m_l1d_misses[streamID];
      for (auto& in_item : in_coming) {
        u32 kernel = in_item.first;
        if (recorded.find(kernel) == recorded.end()) {
          recorded[kernel] = in_item.second;
        } else {
          recorded[kernel] += in_item.second;
        }
      }
    }
  }
  for (auto iter = cs.m_l1d_reads.begin(); iter != cs.m_l1d_reads.end(); ++iter) {
    u64 streamID = iter->first;
    if (ret.m_l1d_reads.find(streamID) == ret.m_l1d_reads.end()) {
      ret.m_l1d_reads[streamID] = cs.m_l1d_reads.at(streamID);
    } else {
      std::map<u32 /* kernel */, u32> in_coming = cs.m_l1d_reads.at(streamID);
      std::map<u32 /* kernel */, u32>& recorded = ret.m_l1d_reads[streamID];
      for (auto& in_item : in_coming) {
        u32 kernel = in_item.first;
        if (recorded.find(kernel) == recorded.end()) {
          recorded[kernel] = in_item.second;
        } else {
          recorded[kernel] += in_item.second;
        }
      }
    }
  }    
  for (auto iter = cs.m_l1d_rd_misses.begin(); iter != cs.m_l1d_rd_misses.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_l1d_rd_misses.find(streamID) == ret.m_l1d_rd_misses.end()) {
      ret.m_l1d_rd_misses[streamID] = cs.m_l1d_rd_misses.at(streamID);
    } else {
      std::map<u32 /* kernel */, u32> in_coming = cs.m_l1d_rd_misses.at(streamID);
      std::map<u32 /* kernel */, u32>& recorded = ret.m_l1d_rd_misses[streamID];
      for (auto& in_item : in_coming) {
        u32 kernel = in_item.first;
        if (recorded.find(kernel) == recorded.end()) {
          recorded[kernel] = in_item.second;
        } else {
          recorded[kernel] += in_item.second;
        }
      }
    }
  }  
  for (auto iter = cs.m_l1d_wr_misses.begin(); iter != cs.m_l1d_wr_misses.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_l1d_wr_misses.find(streamID) == ret.m_l1d_wr_misses.end()) {
      ret.m_l1d_wr_misses[streamID] = cs.m_l1d_wr_misses.at(streamID);
    } else {
      std::map<u32 /* kernel */, u32> in_coming = cs.m_l1d_wr_misses.at(streamID);
      std::map<u32 /* kernel */, u32>& recorded = ret.m_l1d_wr_misses[streamID];
      for (auto& in_item : in_coming) {
        u32 kernel = in_item.first;
        if (recorded.find(kernel) == recorded.end()) {
          recorded[kernel] = in_item.second;
        } else {
          recorded[kernel] += in_item.second;
        }
      }
    }
  }
  for (auto iter = cs.m_l1d_vc_hits.begin(); iter != cs.m_l1d_vc_hits.end(); ++iter) {  
    LOCALITY_KEY key = iter->first;
    if (ret.m_l1d_vc_hits.find(key) == ret.m_l1d_vc_hits.end()) {
      ret.m_l1d_vc_hits[key] = cs.m_l1d_vc_hits.at(key);
    } else {
      ret.m_l1d_vc_hits[key] += cs.m_l1d_vc_hits.at(key);
    }
  }
  for (auto iter = cs.m_l1d_vc_misses.begin(); iter != cs.m_l1d_vc_misses.end(); ++iter) {  
    LOCALITY_KEY key = iter->first;
    if (ret.m_l1d_vc_misses.find(key) == ret.m_l1d_vc_misses.end()) {
      ret.m_l1d_vc_misses[key] = cs.m_l1d_vc_misses.at(key);
    } else {
      ret.m_l1d_vc_misses[key] += cs.m_l1d_vc_misses.at(key);
    }
  }
  for (auto iter = cs.m_l1d_vc_accesses.begin(); iter != cs.m_l1d_vc_accesses.end(); ++iter) {  
    LOCALITY_KEY key = iter->first;
    if (ret.m_l1d_vc_accesses.find(key) == ret.m_l1d_vc_accesses.end()) {
      ret.m_l1d_vc_accesses[key] = cs.m_l1d_vc_accesses.at(key);
    } else {
      ret.m_l1d_vc_accesses[key] += cs.m_l1d_vc_accesses.at(key);
    }
  }  
  for (auto iter = cs.m_l1d_max_evictions.begin(); iter != cs.m_l1d_max_evictions.end(); ++iter) {  
    LOCALITY_KEY loc_key = iter->first;
    if (ret.m_l1d_max_evictions.find(loc_key) == ret.m_l1d_max_evictions.end()) {
      ret.m_l1d_max_evictions[loc_key] = cs.m_l1d_max_evictions.at(loc_key);
    } else {
      ret.m_l1d_max_evictions[loc_key] += cs.m_l1d_max_evictions.at(loc_key);
    }
  }
  for (auto iter = cs.m_l1d_avg_evictions.begin(); iter != cs.m_l1d_avg_evictions.end(); ++iter) {  
    LOCALITY_KEY loc_key = iter->first;
    if (ret.m_l1d_avg_evictions.find(loc_key) == ret.m_l1d_avg_evictions.end()) {
      ret.m_l1d_avg_evictions[loc_key] = cs.m_l1d_avg_evictions.at(loc_key);
    } else {
      ret.m_l1d_avg_evictions[loc_key] += cs.m_l1d_avg_evictions.at(loc_key);
    }
  }

  for (auto iter = cs.m_n_l1d_fill_to_evict_lines.begin(); iter != cs.m_n_l1d_fill_to_evict_lines.end(); ++iter) {  
    const LOCALITY_KEY key = iter->first;
    if (ret.m_n_l1d_fill_to_evict_lines.find(key) == ret.m_n_l1d_fill_to_evict_lines.end()) {
      ret.m_n_l1d_fill_to_evict_lines[key] = cs.m_n_l1d_fill_to_evict_lines.at(key);
    } else {
      ret.m_n_l1d_fill_to_evict_lines[key] += cs.m_n_l1d_fill_to_evict_lines.at(key);
    }
  }

  for (auto iter = cs.m_l2_sub_miss_served_cycles.begin(); 
    iter != cs.m_l2_sub_miss_served_cycles.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_l2_sub_miss_served_cycles.find(streamID) == 
      ret.m_l2_sub_miss_served_cycles.end()) {
      ret.m_l2_sub_miss_served_cycles.insert(
        std::pair<u64, std::vector<u64>>(
          streamID, cs.m_l2_sub_miss_served_cycles.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_l2_sub_miss_served_cycles.at(streamID)[l2_sub] += 
        cs.m_l2_sub_miss_served_cycles.at(streamID)[l2_sub];
      }
    }
  }
  for (auto iter = cs.m_l2_sub_misses.begin(); 
    iter != cs.m_l2_sub_misses.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_l2_sub_misses.find(streamID) == 
      ret.m_l2_sub_misses.end()) {
      ret.m_l2_sub_misses.insert(
        std::pair<u64, std::vector<unsigned>>(
          streamID, cs.m_l2_sub_misses.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_l2_sub_misses.at(streamID)[l2_sub] += 
        cs.m_l2_sub_misses.at(streamID)[l2_sub];
      }
    }
  }

  for (auto iter = cs.m_accu_l2_dram_queue_size.begin(); iter != cs.m_accu_l2_dram_queue_size.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_accu_l2_dram_queue_size.find(streamID) == ret.m_accu_l2_dram_queue_size.end()) {
      ret.m_accu_l2_dram_queue_size.insert(
        std::pair<u64, std::vector<unsigned>>(streamID, cs.m_accu_l2_dram_queue_size.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_accu_l2_dram_queue_size.at(streamID)[l2_sub] += cs.m_accu_l2_dram_queue_size.at(streamID)[l2_sub];
      }      
    }
  }
  for (auto iter = cs.m_accu_l2_icnt_queue_size.begin(); iter != cs.m_accu_l2_icnt_queue_size.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_accu_l2_icnt_queue_size.find(streamID) == ret.m_accu_l2_icnt_queue_size.end()) {
      ret.m_accu_l2_icnt_queue_size.insert(
        std::pair<u64, std::vector<unsigned>>(streamID, cs.m_accu_l2_icnt_queue_size.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_accu_l2_icnt_queue_size.at(streamID)[l2_sub] += cs.m_accu_l2_icnt_queue_size.at(streamID)[l2_sub];
      }
    }
  }
  for (auto iter = cs.m_l2_dram_q_accesses.begin(); iter != cs.m_l2_dram_q_accesses.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_l2_dram_q_accesses.find(streamID) == ret.m_l2_dram_q_accesses.end()) {
      ret.m_l2_dram_q_accesses.insert(
        std::pair<u64, std::vector<unsigned>>(streamID, cs.m_l2_dram_q_accesses.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_l2_dram_q_accesses.at(streamID)[l2_sub] += cs.m_l2_dram_q_accesses.at(streamID)[l2_sub];
      }
    }
  }
  for (auto iter = cs.m_l2_icnt_q_accesses.begin(); iter != cs.m_l2_icnt_q_accesses.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_l2_icnt_q_accesses.find(streamID) == ret.m_l2_icnt_q_accesses.end()) {
      ret.m_l2_icnt_q_accesses.insert(
        std::pair<u64, std::vector<unsigned>>(streamID, cs.m_l2_icnt_q_accesses.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_l2_icnt_q_accesses.at(streamID)[l2_sub] += cs.m_l2_icnt_q_accesses.at(streamID)[l2_sub];
      }
    }
  }  
  for (auto iter = cs.m_l2_mshr_slots_fills.begin(); iter != cs.m_l2_mshr_slots_fills.end(); ++iter) {  
    u64 streamID = iter->first;
    if (ret.m_l2_mshr_slots_fills.find(streamID) == ret.m_l2_mshr_slots_fills.end()) {
      ret.m_l2_mshr_slots_fills.insert(
        std::pair<u64, std::vector<unsigned>>(streamID, cs.m_l2_mshr_slots_fills.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_l2_mshr_slots_fills.at(streamID)[l2_sub] += cs.m_l2_mshr_slots_fills.at(streamID)[l2_sub];
      }
    }
  }

  for (auto iter = cs.m_line_alloc_fail.begin(); iter != cs.m_line_alloc_fail.end(); ++iter) {
    u64 streamID = iter->first;
    if (ret.m_line_alloc_fail.find(streamID) == ret.m_line_alloc_fail.end()) {
      ret.m_line_alloc_fail.insert(std::pair<u64,
        std::vector<std::vector<u64>>>(streamID, cs.m_line_alloc_fail.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned driver = 0; driver < NUM_LINE_ALLOC_FAIL_DRIVER; ++driver) {
          // Should increment here otherwise the final stats would be much less than recorded
          ret.m_line_alloc_fail.at(streamID)[type][driver] += cs(type, LINE_ALLOC_FAIL, true, driver, streamID);
        }        
      }
    }
  }  
  for (auto iter = cs.m_mshr_entry_fail.begin(); iter != cs.m_mshr_entry_fail.end(); ++iter) {
    u64 streamID = iter->first;
    if (ret.m_mshr_entry_fail.find(streamID) == ret.m_mshr_entry_fail.end()) {
      ret.m_mshr_entry_fail.insert(std::pair<u64,
        std::vector<std::vector<u64>>>(streamID, cs.m_mshr_entry_fail.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned driver = 0; driver < NUM_MSHR_ENTRY_FAIL_DRIVER; ++driver) {
          // Should increment here otherwise the final stats would be much less than recorded
          ret.m_mshr_entry_fail.at(streamID)[type][driver] += cs(type, MSHR_ENTRY_FAIL, true, driver, streamID);
        }        
      }
    }
  }  
  for (auto iter = cs.m_miss_q_full.begin(); iter != cs.m_miss_q_full.end(); ++iter) {
    u64 streamID = iter->first;
    if (ret.m_miss_q_full.find(streamID) == ret.m_miss_q_full.end()) {
      ret.m_miss_q_full.insert(std::pair<u64,
        std::vector<std::vector<u64>>>(streamID, cs.m_miss_q_full.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned driver = 0; driver < NUM_MISS_QUEUE_FULL_DRIVER; ++driver) {
          // Should increment here otherwise the final stats would be much less than recorded
          ret.m_miss_q_full.at(streamID)[type][driver] += cs(type, MISS_QUEUE_FULL, true, driver, streamID);
        }        
      }
    }
  }  
  for (auto iter = cs.m_mshr_merge_entry_fail.begin(); iter != cs.m_mshr_merge_entry_fail.end(); ++iter) {
    u64 streamID = iter->first;
    if (ret.m_mshr_merge_entry_fail.find(streamID) == ret.m_mshr_merge_entry_fail.end()) {
      ret.m_mshr_merge_entry_fail.insert(std::pair<u64,
        std::vector<std::vector<u64>>>(streamID, cs.m_mshr_merge_entry_fail.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned driver = 0; driver < NUM_MSHR_MERGE_ENTRY_FAIL_DRIVER; ++driver) {
          // Should increment here otherwise the final stats would be much less than recorded
          ret.m_mshr_merge_entry_fail.at(streamID)[type][driver] += cs(type, MSHR_MERGE_ENTRY_FAIL, true, driver, streamID);
        }        
      }
    }
  }    
  
  ret.m_cache_port_available_cycles =
      m_cache_port_available_cycles + cs.m_cache_port_available_cycles;
  ret.m_cache_data_port_busy_cycles =
      m_cache_data_port_busy_cycles + cs.m_cache_data_port_busy_cycles;
  ret.m_cache_fill_port_busy_cycles =
      m_cache_fill_port_busy_cycles + cs.m_cache_fill_port_busy_cycles;
  ret.m_l2_miss_q_pops += cs.m_l2_miss_q_pops;
  return ret;
}

// m_overall_avg_l1d_rd_fill_to_evict_gap
// m_l1d_rd_miss_served_cycles
// m_l1d_wr_miss_served_cycles
// m_l1d_rd_byp_activates
// m_l1d_rd_byp_deactivates
// m_l1d_accesses
// m_l1d_reads
// m_l1d_writes
// m_l1d_misses
// m_l1d_rd_misses
// m_l1d_wr_misses
// m_n_l1d_fill_to_evict_lines
// m_l2_sub_miss_served_cycles
// m_l2_sub_misses
void cache_stats::accu_single_stat(const char* tgt_item, const cache_stats &cs) {
  if (!strcmp(tgt_item, "m_overall_avg_l1d_rd_fill_to_evict_gap")) {
    for (auto iter = cs.m_overall_avg_l1d_rd_fill_to_evict_gap.begin(); 
      iter != cs.m_overall_avg_l1d_rd_fill_to_evict_gap.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_overall_avg_l1d_rd_fill_to_evict_gap.find(streamID) == 
        m_overall_avg_l1d_rd_fill_to_evict_gap.end()) {
        m_overall_avg_l1d_rd_fill_to_evict_gap.insert(
          std::pair<u64, u64>(streamID, cs.m_overall_avg_l1d_rd_fill_to_evict_gap.at(streamID)));
      } else {
        m_overall_avg_l1d_rd_fill_to_evict_gap.at(streamID) += cs(streamID, tgt_item);
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_rd_miss_served_cycles")) {
    for (auto iter = cs.m_l1d_rd_miss_served_cycles.begin(); iter != cs.m_l1d_rd_miss_served_cycles.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_l1d_rd_miss_served_cycles.find(streamID) == m_l1d_rd_miss_served_cycles.end()) {
        m_l1d_rd_miss_served_cycles.insert(
          std::pair<u64, u64>(streamID, cs.m_l1d_rd_miss_served_cycles.at(streamID)));
      } else {
        m_l1d_rd_miss_served_cycles.at(streamID) += cs(streamID, tgt_item);
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_wr_miss_served_cycles")) {
    for (auto iter = cs.m_l1d_wr_miss_served_cycles.begin(); iter != cs.m_l1d_wr_miss_served_cycles.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_l1d_wr_miss_served_cycles.find(streamID) == m_l1d_wr_miss_served_cycles.end()) {
        m_l1d_wr_miss_served_cycles.insert(
          std::pair<u64, u64>(streamID, cs.m_l1d_wr_miss_served_cycles.at(streamID)));
      } else {
        m_l1d_wr_miss_served_cycles.at(streamID) += cs(streamID, tgt_item);
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_rd_byp_activates")) {
    for (auto iter = cs.m_l1d_rd_byp_activates.begin(); 
      iter != cs.m_l1d_rd_byp_activates.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_l1d_rd_byp_activates.find(streamID) == m_l1d_rd_byp_activates.end()) {
        m_l1d_rd_byp_activates[streamID] = cs.m_l1d_rd_byp_activates.at(streamID);
      } else {
        std::map<u64, u32>& recorded = m_l1d_rd_byp_activates.at(streamID);
        std::map<u64, u32> in_coming = cs.m_l1d_rd_byp_activates.at(streamID);
        for (auto& iter : in_coming) {
          const u64 block_addr = iter.first;
          recorded[block_addr] += iter.second;
        }
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_rd_byp_deactivates")) {
    for (auto iter = cs.m_l1d_rd_byp_deactivates.begin(); 
      iter != cs.m_l1d_rd_byp_deactivates.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_l1d_rd_byp_deactivates.find(streamID) == m_l1d_rd_byp_deactivates.end()) {
        m_l1d_rd_byp_deactivates[streamID] = cs.m_l1d_rd_byp_deactivates.at(streamID);
      } else {
        std::map<u64, u32>& recorded = m_l1d_rd_byp_deactivates.at(streamID);
        std::map<u64, u32> in_coming = cs.m_l1d_rd_byp_deactivates.at(streamID);
        for (auto& iter : in_coming) {
          const u64 block_addr = iter.first;
          recorded[block_addr] += iter.second;
        }
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_accesses")) {
    for (auto iter = cs.m_l1d_accesses.begin(); iter != cs.m_l1d_accesses.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_l1d_accesses.find(streamID) == m_l1d_accesses.end()) {
        m_l1d_accesses[streamID] = cs.m_l1d_accesses.at(streamID);
      } else {
        std::map<u32 /* kernel */, u32>& recorded = m_l1d_accesses.at(streamID);
        std::map<u32 /* kernel */, u32> in_coming = cs.m_l1d_accesses.at(streamID);
        for (auto& iter : in_coming) {
          u32 kernel = iter.first;
          if (recorded.find(kernel) == recorded.end()) {
            recorded[kernel] = iter.second;
          } else {
            recorded[kernel] += iter.second;
          }
        }
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_misses")) {
    for (auto iter = cs.m_l1d_misses.begin(); iter != cs.m_l1d_misses.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_l1d_misses.find(streamID) == m_l1d_misses.end()) {
        m_l1d_misses[streamID] = cs.m_l1d_misses.at(streamID);
      } else {
        std::map<u32 /* kernel */, u32>& recorded = m_l1d_misses.at(streamID);
        std::map<u32 /* kernel */, u32> in_coming = cs.m_l1d_misses.at(streamID);
        for (auto& iter : in_coming) {
          u32 kernel = iter.first;
          if (recorded.find(kernel) == recorded.end()) {
            recorded[kernel] = iter.second;
          } else {
            recorded[kernel] += iter.second;
          }
        }
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_reads")) {
    for (auto iter = cs.m_l1d_reads.begin(); iter != cs.m_l1d_reads.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_l1d_reads.find(streamID) == m_l1d_reads.end()) {
        m_l1d_reads[streamID] = cs.m_l1d_reads.at(streamID);
      } else {
        std::map<u32 /* kernel */, u32>& recorded = m_l1d_reads.at(streamID);
        std::map<u32 /* kernel */, u32> in_coming = cs.m_l1d_reads.at(streamID);
        for (auto& iter : in_coming) {
          const u32 kernel = iter.first;
          [[maybe_unused]] u32 old_value = recorded[kernel];
          recorded[kernel] += iter.second;
          if (DTRACE(GATHER_FINAL_L1D_STATS)) {
            fprintf(Trace::out, "m_l1d_reads[streamID:%llu][kernel:%u] "
              "+= %u (old_val:%u->%u)\n", 
              streamID, kernel, iter.second, old_value, recorded[kernel]);
          }
        }
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_writes")) {
    for (auto iter = cs.m_l1d_writes.begin(); iter != cs.m_l1d_writes.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_l1d_writes.find(streamID) == m_l1d_writes.end()) {
        m_l1d_writes[streamID] = cs.m_l1d_writes.at(streamID);
      } else {
        std::map<u32 /* kernel */, u32>& recorded = m_l1d_writes.at(streamID);
        std::map<u32 /* kernel */, u32> in_coming = cs.m_l1d_writes.at(streamID);
        for (auto& iter : in_coming) {
          const u32 kernel = iter.first;
          recorded[kernel] += iter.second;
        }
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_rd_misses")) {
    for (auto iter = cs.m_l1d_rd_misses.begin(); iter != cs.m_l1d_rd_misses.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_l1d_rd_misses.find(streamID) == m_l1d_rd_misses.end()) {
        m_l1d_rd_misses[streamID] = cs.m_l1d_rd_misses.at(streamID);
      } else {
        std::map<u32 /* kernel */, u32>& recorded = m_l1d_rd_misses.at(streamID);
        std::map<u32 /* kernel */, u32> in_coming = cs.m_l1d_rd_misses.at(streamID);
        for (auto& iter : in_coming) {
          u32 kernel = iter.first;
          if (recorded.find(kernel) == recorded.end()) {
            recorded[kernel] = iter.second;
          } else {
            recorded[kernel] += iter.second;
          }
        }
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_wr_misses")) {
    for (auto iter = cs.m_l1d_wr_misses.begin(); iter != cs.m_l1d_wr_misses.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_l1d_wr_misses.find(streamID) == m_l1d_wr_misses.end()) {
        m_l1d_wr_misses[streamID] = cs.m_l1d_wr_misses.at(streamID);
      } else {
        std::map<u32 /* kernel */, u32>& recorded = m_l1d_wr_misses.at(streamID);
        std::map<u32 /* kernel */, u32> in_coming = cs.m_l1d_wr_misses.at(streamID);
        for (auto& iter : in_coming) {
          u32 kernel = iter.first;
          if (recorded.find(kernel) == recorded.end()) {
            recorded[kernel] = iter.second;
          } else {
            recorded[kernel] += iter.second;
          }
        }
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_vc_hits")) {
    for (auto iter = cs.m_l1d_vc_hits.begin(); iter != cs.m_l1d_vc_hits.end(); ++iter) {
      LOCALITY_KEY key = iter->first;
      if (m_l1d_vc_hits.find(key) == m_l1d_vc_hits.end()) {
        m_l1d_vc_hits[key] = cs.m_l1d_vc_hits.at(key);
      } else {
        m_l1d_vc_hits[key] += cs.m_l1d_vc_hits.at(key);
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_vc_misses")) {
    for (auto iter = cs.m_l1d_vc_misses.begin(); iter != cs.m_l1d_vc_misses.end(); ++iter) {
      LOCALITY_KEY key = iter->first;
      if (m_l1d_vc_misses.find(key) == m_l1d_vc_misses.end()) {
        m_l1d_vc_misses[key] = cs.m_l1d_vc_misses.at(key);
      } else {
        m_l1d_vc_misses[key] += cs.m_l1d_vc_misses.at(key);
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_vc_accesses")) {
    for (auto iter = cs.m_l1d_vc_accesses.begin(); iter != cs.m_l1d_vc_accesses.end(); ++iter) {
      LOCALITY_KEY key = iter->first;
      if (m_l1d_vc_accesses.find(key) == m_l1d_vc_accesses.end()) {
        m_l1d_vc_accesses[key] = cs.m_l1d_vc_accesses.at(key);
      } else {
        m_l1d_vc_accesses[key] += cs.m_l1d_vc_accesses.at(key);
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_max_evictions")) {
    for (auto iter = cs.m_l1d_max_evictions.begin(); iter != cs.m_l1d_max_evictions.end(); ++iter) {
      LOCALITY_KEY loc_key = iter->first;
      if (m_l1d_max_evictions.find(loc_key) == m_l1d_max_evictions.end()) {
        m_l1d_max_evictions[loc_key] = cs.m_l1d_max_evictions.at(loc_key);
      } else {
        m_l1d_max_evictions[loc_key] += cs.m_l1d_max_evictions.at(loc_key);
        if (DTRACE(DEBUG_L1D_MAX_EVICTIONS)) {
          fprintf(Trace::out, "m_l1d_max_evictions[loc_key:<streamID:%llu, kernel:%u>] "
            "+= %u (new_val:%u)\n", 
            loc_key.stream_id, loc_key.kernel, cs.m_l1d_max_evictions.at(loc_key), 
            m_l1d_max_evictions.at(loc_key));
        }
      }
    }
  } else if (!strcmp(tgt_item, "m_l1d_avg_evictions")) {
    for (auto iter = cs.m_l1d_avg_evictions.begin(); iter != cs.m_l1d_avg_evictions.end(); ++iter) {
      LOCALITY_KEY loc_key = iter->first;
      if (m_l1d_avg_evictions.find(loc_key) == m_l1d_avg_evictions.end()) {
        m_l1d_avg_evictions[loc_key] = cs.m_l1d_avg_evictions.at(loc_key);
      } else {
        m_l1d_avg_evictions[loc_key] += cs.m_l1d_avg_evictions.at(loc_key);
        if (DTRACE(DEBUG_L1D_AVG_EVICTIONS)) {
          fprintf(Trace::out, "m_l1d_avg_evictions[loc_key:<streamID:%llu, kernel:%u>] "
            "+= %u (new_val:%u)\n", 
            loc_key.stream_id, loc_key.kernel, cs.m_l1d_avg_evictions.at(loc_key), 
            m_l1d_avg_evictions.at(loc_key));
        }        
      }
    }
  } else if (!strcmp(tgt_item, "m_n_l1d_fill_to_evict_lines")) {
    for (auto iter = cs.m_n_l1d_fill_to_evict_lines.begin(); iter != cs.m_n_l1d_fill_to_evict_lines.end(); ++iter) {
      const LOCALITY_KEY key = iter->first;
      if (m_n_l1d_fill_to_evict_lines.find(key) == m_n_l1d_fill_to_evict_lines.end()) {
        m_n_l1d_fill_to_evict_lines[key] = cs.m_n_l1d_fill_to_evict_lines.at(key);
      } else {
        m_n_l1d_fill_to_evict_lines[key] += cs.m_n_l1d_fill_to_evict_lines.at(key);
      }
    }
  } else if (!strcmp(tgt_item, "m_l2_sub_miss_served_cycles")) {
    for (auto iter = cs.m_l2_sub_miss_served_cycles.begin(); 
      iter != cs.m_l2_sub_miss_served_cycles.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_l2_sub_miss_served_cycles.find(streamID) == m_l2_sub_miss_served_cycles.end()) {
        m_l2_sub_miss_served_cycles.insert(
          std::pair<u64, std::vector<u64>>(
            streamID, cs.m_l2_sub_miss_served_cycles.at(streamID)));
      } else {
        for (u32 l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
          m_l2_sub_miss_served_cycles.at(streamID)[l2_sub] += cs(l2_sub, streamID, tgt_item);
        }      
      }
    }    
  } else if (!strcmp(tgt_item, "m_l2_sub_misses")) {
    for (auto iter = cs.m_l2_sub_misses.begin(); 
      iter != cs.m_l2_sub_misses.end(); ++iter) {
      u64 streamID = iter->first;
      if (m_l2_sub_misses.find(streamID) == m_l2_sub_misses.end()) {
        m_l2_sub_misses.insert(
          std::pair<u64, std::vector<u32>>(
            streamID, cs.m_l2_sub_misses.at(streamID)));
      } else {
        for (u32 l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
          m_l2_sub_misses.at(streamID)[l2_sub] += cs(l2_sub, streamID, tgt_item);
        }      
      }
    }    
  }   

}

cache_stats &cache_stats::operator+=(const cache_stats &cs) {
  ///
  /// Overloaded += operator to allow for simple stat accumulation
  ///
  const char* local_cache_type = get_cache_name();
  std::string l2_prefix = "";
  if (strcmp(local_cache_type, "L2") == 0) {
    l2_prefix = "SG";
    l2_prefix += std::to_string(get_sub_partition());
    l2_prefix += " ";      
  }

  for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_stats.find(streamID) == m_stats.end()) {
      if (DTRACE(M_STATS)) {
        fprintf(Trace::out, "%s %sm_stats.size:%lu m_stats.insert(cs.m_stats.at(streamID:%llu))\n",
          local_cache_type,
          !strcmp(local_cache_type, "L2") ? l2_prefix.c_str() : "", 
          m_stats.size(), streamID
        );
      }

      m_stats.insert(std::pair<u64,
        std::vector<std::vector<u64>>>(streamID, cs.m_stats.at(streamID)));

      if (DTRACE(M_STATS)) {
        for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
          for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {            
            fprintf(Trace::out, "%s %sm_stats.size:%lu "
              "m_stats[streamID:%llu][type:%s][status:%s](%llu)\n",
              local_cache_type,
              !strcmp(local_cache_type, "L2") ? l2_prefix.c_str() : "",
              m_stats.size(),
              streamID, 
              mem_access_type_str(mem_access_type(type)), 
              cache_request_status_str(cache_request_status(status)), 
              m_stats.at(streamID)[type][status]              
            );          
          }
        }
      } // if (DTRACE(M_STATS)) {
    } else {
      if (DTRACE(M_STATS)) {
        fprintf(Trace::out, "%s %sm_stats hit streamID:%llu\n",
          local_cache_type,
          !strcmp(local_cache_type, "L2") ? l2_prefix.c_str() : "", streamID
        );
      }      
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          u64 orig_stats_val = m_stats.at(streamID)[type][status];
          m_stats.at(streamID)[type][status] += cs(type, status, false, streamID);
          if (DTRACE(M_STATS)) {
            fprintf(Trace::out, "%s %sm_stats.size:%lu cache_stats::operator+= "
              "m_stats[streamID:%llu][type:%s][status:%s](%llu->%llu) += cs(%llu)\n",
              local_cache_type,
              !strcmp(local_cache_type, "L2") ? l2_prefix.c_str() : "",
              m_stats.size(),
              streamID, 
              mem_access_type_str(mem_access_type(type)), 
              cache_request_status_str(cache_request_status(status)), 
              orig_stats_val,
              m_stats.at(streamID)[type][status],
              cs(type, status, false, streamID)
            );          
          }
        }
      } // outer-for
    } // m_stats.find(streamID) != m_stats.end()
  } // for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) 
  for (auto iter = cs.m_stats_pw.begin(); iter != cs.m_stats_pw.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_stats_pw.find(streamID) == m_stats_pw.end()) {
      m_stats_pw.insert(std::pair<u64,
                                  std::vector<std::vector<u64>>>(
          streamID, cs.m_stats_pw.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          m_stats_pw.at(streamID)[type][status] += cs(type, status, false, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_fail_stats.begin(); iter != cs.m_fail_stats.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_fail_stats.find(streamID) == m_fail_stats.end()) {
      m_fail_stats.insert(std::pair<u64,
          std::vector<std::vector<u64>>>(streamID, cs.m_fail_stats.at(streamID)));            
      m_fail_stats_total.insert(std::pair<u64,
          std::vector<u64>>(streamID, cs.m_fail_stats_total.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS; ++status) {
          u64 orig_fail_stats = m_fail_stats.at(streamID)[type][status];
          m_fail_stats.at(streamID)[type][status] += cs(type, status, true, streamID);
          m_fail_stats_total.at(streamID)[type] += cs(type, status, true, streamID);
          if (DTRACE(M_STATS)) {
            fprintf(Trace::out, "%s %s"
              "m_fail_stats[streamID:%llu][type:%u][status:%u](%llu->%llu) += cs(%u, %u, true, %llu)\n",
              local_cache_type,
              !strcmp(local_cache_type, "L2") ? l2_prefix.c_str() : "",
              streamID, type, status,
              orig_fail_stats, m_fail_stats.at(streamID)[type][status],
              type, status, streamID
            );
          }          
        } // for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS; ++status) {
      } // for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    }
  } // for (auto iter = cs.m_fail_stats.begin(); iter != cs.m_fail_stats.end(); ++iter) {

  for (auto iter = cs.m_line_alloc_fail.begin(); iter != cs.m_line_alloc_fail.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_line_alloc_fail.find(streamID) == m_line_alloc_fail.end()) {
      m_line_alloc_fail.insert(std::pair<u64,
          std::vector<std::vector<u64>>>(streamID, cs.m_line_alloc_fail.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned driver = 0; driver < NUM_LINE_ALLOC_FAIL_DRIVER; ++driver) {          
          m_line_alloc_fail.at(streamID)[type][driver] += cs(type, LINE_ALLOC_FAIL, true, driver, streamID);
        }
      }
    }
  }  
  for (auto iter = cs.m_mshr_entry_fail.begin(); iter != cs.m_mshr_entry_fail.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_mshr_entry_fail.find(streamID) == m_mshr_entry_fail.end()) {
      m_mshr_entry_fail.insert(std::pair<u64,
          std::vector<std::vector<u64>>>(streamID, cs.m_mshr_entry_fail.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned driver = 0; driver < NUM_MSHR_ENTRY_FAIL_DRIVER; ++driver) {          
          m_mshr_entry_fail.at(streamID)[type][driver] += cs(type, MSHR_ENTRY_FAIL, true, driver, streamID);
        }
      }
    }
  }  
  for (auto iter = cs.m_miss_q_full.begin(); iter != cs.m_miss_q_full.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_miss_q_full.find(streamID) == m_miss_q_full.end()) {
      m_miss_q_full.insert(std::pair<u64,
          std::vector<std::vector<u64>>>(streamID, cs.m_miss_q_full.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned driver = 0; driver < NUM_MISS_QUEUE_FULL_DRIVER; ++driver) {          
          m_miss_q_full.at(streamID)[type][driver] += cs(type, MISS_QUEUE_FULL, true, driver, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_mshr_merge_entry_fail.begin(); iter != cs.m_mshr_merge_entry_fail.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_mshr_merge_entry_fail.find(streamID) == m_mshr_merge_entry_fail.end()) {
      m_mshr_merge_entry_fail.insert(std::pair<u64,
          std::vector<std::vector<u64>>>(streamID, cs.m_mshr_merge_entry_fail.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned driver = 0; driver < NUM_MSHR_MERGE_ENTRY_FAIL_DRIVER; ++driver) {          
          m_mshr_merge_entry_fail.at(streamID)[type][driver] += cs(type, MSHR_MERGE_ENTRY_FAIL, true, driver, streamID);
        }
      }
    }
  }

  for (auto iter = cs.m_mshr_occupancy_stats.begin(); iter != cs.m_mshr_occupancy_stats.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_mshr_occupancy_stats.find(streamID) == m_mshr_occupancy_stats.end()) {
      m_mshr_occupancy_stats.insert(
        std::pair<u64,
        std::vector<std::vector<unsigned>>>(streamID, cs.m_mshr_occupancy_stats.at(streamID)));
    } else {
      const unsigned sms = 4; // gpgpu_n_cores_per_cluster
      const unsigned max_warps_per_sm = 64; // m_config.max_warps_per_sm
      for (unsigned sm = 0; sm < sms; sm++) {
        for (unsigned warp = 0; warp < max_warps_per_sm; warp++) {
          unsigned orig_mshr_occupancy = m_mshr_occupancy_stats.at(streamID)[sm][warp];
          m_mshr_occupancy_stats.at(streamID)[sm][warp] += cs(sm, warp, streamID);
          if (DTRACE(MSHR_STATS) || DTRACE(M_STATS)) {
            fprintf(Trace::out, "%s %s"
              "m_mshr_occupancy_stats[streamID:%llu][sm:%u][warp:%u](%u->%u) += cs(%u, %u, %llu)\n",
              local_cache_type, get_cache_name(),
              streamID, sm, warp,
              orig_mshr_occupancy, m_mshr_occupancy_stats.at(streamID)[sm][warp],
              sm, warp, streamID
            );
          }
        }
      }      
    }
  } // for (auto iter = cs.m_mshr_occupancy_stats.begin(); iter != cs.m_mshr_occupancy_stats.end(); ++iter) {

  accu_single_stat("m_overall_avg_l1d_rd_fill_to_evict_gap", cs);
  accu_single_stat("m_l1d_rd_miss_served_cycles", cs);
  accu_single_stat("m_l1d_wr_miss_served_cycles", cs);
  accu_single_stat("m_l1d_rd_byp_activates", cs);
  accu_single_stat("m_l1d_rd_byp_deactivates", cs);
  accu_single_stat("m_l1d_accesses", cs);
  accu_single_stat("m_l1d_reads", cs);
  accu_single_stat("m_l1d_writes", cs);  
  accu_single_stat("m_l1d_misses", cs);
  accu_single_stat("m_l1d_rd_misses", cs);
  accu_single_stat("m_l1d_wr_misses", cs);
  accu_single_stat("m_l1d_vc_hits", cs);
  accu_single_stat("m_l1d_vc_misses", cs);
  accu_single_stat("m_l1d_vc_accesses", cs);
  accu_single_stat("m_l1d_max_evictions", cs);
  accu_single_stat("m_l1d_avg_evictions", cs);
  accu_single_stat("m_n_l1d_fill_to_evict_lines", cs);
  accu_single_stat("m_l2_sub_miss_served_cycles", cs);
  accu_single_stat("m_l2_sub_misses", cs);

  for (auto iter = cs.m_accu_l2_dram_queue_size.begin(); iter != cs.m_accu_l2_dram_queue_size.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_accu_l2_dram_queue_size.find(streamID) == m_accu_l2_dram_queue_size.end()) {
      m_accu_l2_dram_queue_size.insert(
        std::pair<u64, std::vector<unsigned>>(streamID, cs.m_accu_l2_dram_queue_size.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
        m_accu_l2_dram_queue_size.at(streamID)[l2_sub] += cs(l2_sub, streamID);
      }      
    }
  }
  for (auto iter = cs.m_accu_l2_icnt_queue_size.begin(); iter != cs.m_accu_l2_icnt_queue_size.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_accu_l2_icnt_queue_size.find(streamID) == m_accu_l2_icnt_queue_size.end()) {
      m_accu_l2_icnt_queue_size.insert(
        std::pair<u64, std::vector<unsigned>>(streamID, cs.m_accu_l2_icnt_queue_size.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
        m_accu_l2_icnt_queue_size.at(streamID)[l2_sub] += cs(l2_sub, streamID);
      }      
    }
  }
  for (auto iter = cs.m_l2_dram_q_accesses.begin(); iter != cs.m_l2_dram_q_accesses.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_l2_dram_q_accesses.find(streamID) == m_l2_dram_q_accesses.end()) {
      m_l2_dram_q_accesses.insert(
        std::pair<u64, std::vector<unsigned>>(streamID, cs.m_l2_dram_q_accesses.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
        m_l2_dram_q_accesses.at(streamID)[l2_sub] += cs(l2_sub, streamID);
      }      
    }
  }  
  for (auto iter = cs.m_l2_icnt_q_accesses.begin(); iter != cs.m_l2_icnt_q_accesses.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_l2_icnt_q_accesses.find(streamID) == m_l2_icnt_q_accesses.end()) {
      m_l2_icnt_q_accesses.insert(
        std::pair<u64, std::vector<unsigned>>(streamID, cs.m_l2_icnt_q_accesses.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
        m_l2_icnt_q_accesses.at(streamID)[l2_sub] += cs(l2_sub, streamID);
      }      
    }
  }
  for (auto iter = cs.m_l2_mshr_slots_fills.begin(); iter != cs.m_l2_mshr_slots_fills.end(); ++iter) {
    u64 streamID = iter->first;
    if (m_l2_mshr_slots_fills.find(streamID) == m_l2_mshr_slots_fills.end()) {
      m_l2_mshr_slots_fills.insert(
        std::pair<u64, std::vector<unsigned>>(streamID, cs.m_l2_mshr_slots_fills.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
        m_l2_mshr_slots_fills.at(streamID)[l2_sub] += cs(l2_sub, streamID);
      }      
    }
  }
  m_cache_port_available_cycles += cs.m_cache_port_available_cycles;
  m_cache_data_port_busy_cycles += cs.m_cache_data_port_busy_cycles;
  m_cache_fill_port_busy_cycles += cs.m_cache_fill_port_busy_cycles;
  m_l2_miss_q_pops += cs.m_l2_miss_q_pops;
  return *this;
}

void cache_stats::print_stats(FILE *fout, u64 streamID,
                              const char *cache_info) const {
  ///
  /// For a given CUDA stream, print out each non-zero cache statistic for every
  /// memory access type and status "cache_name" defaults to "Cache_stats" when
  /// no argument is provided, otherwise the provided name is used. The printed
  /// format is
  /// "<cache_info>[<request_type>][<request_status>] = <stat_value>"
  /// Specify streamID to be -1 to print every stream.

  std::vector<unsigned> total_access;
  std::string m_cache_info = cache_info;
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    u64 streamid = iter->first;
    fprintf(fout, "streamID: %llu\n", streamid);
    // when streamID is specified, skip stats for all other streams, otherwise,
    // print stats from all streams
    if ((streamID != ((u64) - 1)) && (streamid != streamID)) { 
      continue;
    }

    total_access.clear();
    total_access.resize(NUM_MEM_ACCESS_TYPE, 0);
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        fprintf(fout, "\t%s[%s][%s] = %llu\n", 
          m_cache_info.c_str(),
          mem_access_type_str((enum mem_access_type)type),
          cache_request_status_str((enum cache_request_status)status),
          m_stats.at(streamid)[type][status]);

        if (status != RESERVATION_FAIL && status != MSHR_HIT) {
          // MSHR_HIT is a special type of SECTOR_MISS
          // so its already included in the SECTOR_MISS
          total_access[type] += m_stats.at(streamid)[type][status];
        }
      }
    }
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      if (total_access[type] > 0)
        fprintf(fout, "\t%s[%s][%s] = %u\n", m_cache_info.c_str(),
                mem_access_type_str((enum mem_access_type)type), "TOTAL_ACCESS",
                total_access[type]);
    }
  }
}

void cache_stats::print_fail_stats(FILE *fout, u64 streamID,
                                   const char *cache_info) const {
  std::string m_cache_info = cache_info;
  for (auto iter = m_fail_stats.begin(); iter != m_fail_stats.end(); ++iter) {
    u64 streamid = iter->first;
    // when streamID is specified, skip stats for all other streams, otherwise,
    // print stats from all streams
    if ((streamID != ((u64) - 1)) && (streamid != streamID)) {
      continue;
    }

    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      // breakdown[GLOBAL_ACC_R] = xxx
      fprintf(fout, "\t%s[%s] = %llu\n", m_cache_info.c_str(), 
        mem_access_type_str((enum mem_access_type)type),
        m_fail_stats_total.at(streamid)[type]);      
      for (unsigned fail = 0; fail < NUM_CACHE_RESERVATION_FAIL_STATUS; ++fail) {
        
        if (m_fail_stats.at(streamid)[type][fail] > 0) {
          // breakdown[GLOBAL_ACC_R][MISS_QUEUE_FULL] = xxx
          fprintf(fout, "\t%s[%s][%s] = %llu\n", m_cache_info.c_str(),
              mem_access_type_str((enum mem_access_type)type),
              cache_fail_status_str((enum cache_reservation_fail_reason)fail),
              m_fail_stats.at(streamid)[type][fail]);

          fprintf(
              fout, "\t%s[%s][%s].dist = %f\n", m_cache_info.c_str(),
              mem_access_type_str((enum mem_access_type)type),
              cache_fail_status_str((enum cache_reservation_fail_reason)fail),
              (float)m_fail_stats.at(streamid)[type][fail] /
                  (float)m_fail_stats_total.at(streamid)[type]);

          if (static_cast<cache_reservation_fail_reason>(fail) == LINE_ALLOC_FAIL) {
            for (unsigned driver = 0; driver < NUM_LINE_ALLOC_FAIL_DRIVER; ++driver) {
              if (m_line_alloc_fail.at(streamid)[type][driver] > 0) {
                fprintf(
                    fout, "\t%s[%s][%s][%s] = %llu\n", m_cache_info.c_str(),
                    mem_access_type_str((enum mem_access_type)type),                    
                    cache_fail_status_str((enum cache_reservation_fail_reason)fail),
                    line_alloc_fail_driver_str((enum line_alloc_fail_driver)driver),
                    m_line_alloc_fail.at(streamid)[type][driver]);

                fprintf(
                    fout, "\t%s[%s][%s][%s].dist = %f\n", m_cache_info.c_str(),
                    mem_access_type_str((enum mem_access_type)type),
                    cache_fail_status_str((enum cache_reservation_fail_reason)fail),
                    line_alloc_fail_driver_str((enum line_alloc_fail_driver)driver),
                    (float)m_line_alloc_fail.at(streamid)[type][driver] /
                        (float)m_fail_stats.at(streamid)[type][cache_reservation_fail_reason::LINE_ALLOC_FAIL]);
              }
            } // for (unsigned driver = 0; driver < NUM_MSHR_MERGE_ENTRY_FAIL_DRIVER; ++driver) {
          } else if (static_cast<cache_reservation_fail_reason>(fail) == MSHR_ENTRY_FAIL) {
            for (unsigned driver = 0; driver < NUM_MSHR_ENTRY_FAIL_DRIVER; ++driver) {
              if (m_mshr_entry_fail.at(streamid)[type][driver] > 0) {
                fprintf(
                    fout, "\t%s[%s][%s][%s] = %llu\n", m_cache_info.c_str(),
                    mem_access_type_str((enum mem_access_type)type),                    
                    cache_fail_status_str((enum cache_reservation_fail_reason)fail),
                    mshr_entry_fail_driver_str((enum mshr_entry_fail_driver)driver),
                    m_mshr_entry_fail.at(streamid)[type][driver]);

                fprintf(
                    fout, "\t%s[%s][%s][%s].dist = %f\n", m_cache_info.c_str(),
                    mem_access_type_str((enum mem_access_type)type),
                    cache_fail_status_str((enum cache_reservation_fail_reason)fail),
                    mshr_entry_fail_driver_str((enum mshr_entry_fail_driver)driver),
                    (float)m_mshr_entry_fail.at(streamid)[type][driver] /
                        (float)m_fail_stats.at(streamid)[type][cache_reservation_fail_reason::MSHR_ENTRY_FAIL]);
              }
            } // for (unsigned driver = 0; driver < NUM_MSHR_ENTRY_FAIL_DRIVER; ++driver) {
          } else if (static_cast<cache_reservation_fail_reason>(fail) == MISS_QUEUE_FULL) {
            for (unsigned driver = 0; driver < NUM_MISS_QUEUE_FULL_DRIVER; ++driver) {
              if (m_miss_q_full.at(streamid)[type][driver] > 0) {
                fprintf(
                    fout, "\t%s[%s][%s][%s] = %llu\n", m_cache_info.c_str(),
                    mem_access_type_str((enum mem_access_type)type),                    
                    cache_fail_status_str((enum cache_reservation_fail_reason)fail),
                    miss_queue_full_driver_str((enum miss_queue_full_driver)driver),
                    m_miss_q_full.at(streamid)[type][driver]);

                fprintf(
                    fout, "\t%s[%s][%s][%s].dist = %f\n", m_cache_info.c_str(),
                    mem_access_type_str((enum mem_access_type)type),
                    cache_fail_status_str((enum cache_reservation_fail_reason)fail),
                    miss_queue_full_driver_str((enum miss_queue_full_driver)driver),
                    (float)m_miss_q_full.at(streamid)[type][driver] /
                        (float)m_fail_stats.at(streamid)[type][cache_reservation_fail_reason::MISS_QUEUE_FULL]);
              }
            } // for (unsigned driver = 0; driver < NUM_MISS_QUEUE_FULL_DRIVER; ++driver) {
          } else if (static_cast<cache_reservation_fail_reason>(fail) == MSHR_MERGE_ENTRY_FAIL) {
            for (unsigned driver = 0; driver < NUM_MSHR_MERGE_ENTRY_FAIL_DRIVER; ++driver) {
              if (m_mshr_merge_entry_fail.at(streamid)[type][driver] > 0) {
                fprintf(
                    fout, "\t%s[%s][%s][%s] = %llu\n", m_cache_info.c_str(),
                    mem_access_type_str((enum mem_access_type)type),                    
                    cache_fail_status_str((enum cache_reservation_fail_reason)fail),
                    mshr_merge_entry_fail_driver_str((enum mshr_merge_entry_fail_driver)driver),
                    m_mshr_merge_entry_fail.at(streamid)[type][driver]);

                fprintf(
                    fout, "\t%s[%s][%s][%s].dist = %f\n", m_cache_info.c_str(),
                    mem_access_type_str((enum mem_access_type)type),
                    cache_fail_status_str((enum cache_reservation_fail_reason)fail),
                    mshr_merge_entry_fail_driver_str((enum mshr_merge_entry_fail_driver)driver),
                    (float)m_mshr_merge_entry_fail.at(streamid)[type][driver] /
                        (float)m_fail_stats.at(streamid)[type][cache_reservation_fail_reason::MSHR_MERGE_ENTRY_FAIL]);
              }
            } // for (unsigned driver = 0; driver < NUM_MSHR_MERGE_ENTRY_FAIL_DRIVER; ++driver) {
          }          
        } // if (m_fail_stats.at(streamid)[type][fail] > 0) {
      } // for (unsigned fail = 0; fail < NUM_CACHE_RESERVATION_FAIL_STATUS; ++fail) {
    } // for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
  }
}

void cache_stats::print_mshr_stats(FILE *fout, u64 streamID,
                                   const char *cache_info) const {
  for (auto iter = m_mshr_occupancy_stats.begin(); iter != m_mshr_occupancy_stats.end(); ++iter) {
    u64 streamid = iter->first;
    // when streamID is specified, skip stats for all other streams, otherwise,
    // print stats from all streams
    if ((streamID != ((u64) - 1)) && (streamid != streamID)) {
      continue;
    }
    const unsigned sms = 4; // gpgpu_n_cores_per_cluster
    const unsigned max_warps_per_sm = 64; // m_config.max_warps_per_sm    
    for (unsigned sm = 0; sm < sms; ++sm) {
      for (unsigned warp = 0; warp < max_warps_per_sm; ++warp) {
        fprintf(fout, "\t%s[streamID:%llu][sm:%u][warp:%u] = %u\n", cache_info,
          streamid, sm, warp, m_mshr_occupancy_stats.at(streamid)[sm][warp]);
      }
    }
  }
  if (!m_mshr_occupancy_stats.size()) {
    fprintf(fout, "\t%s: m_mshr_occupancy_stats is empty\n", cache_info);
  }
}

void cache_stats::print_l2_dram_queue_stats(
  FILE *fout, unsigned l2_dram_q_capacity, u64 streamID, const char *info) const {
  for (auto iter = m_accu_l2_dram_queue_size.begin(); 
    iter != m_accu_l2_dram_queue_size.end(); ++iter) {
    if ((streamID != ((u64) - 1)) && (iter->first != streamID)) {
      continue;
    }
    for (unsigned l2_sub = 0; l2_sub < iter->second.size(); ++l2_sub) {       
      float avg_l2_dram_q_size = 
        (m_accu_l2_dram_queue_size.at(streamID)[l2_sub] / 
        (float)m_l2_dram_q_accesses.at(streamID)[l2_sub]);
      float avg_l2_dram_q_occupancy = avg_l2_dram_q_size / (float)l2_dram_q_capacity;
      fprintf(fout, "\tm_l2_dram_q_accesses[sub:%u] = %u\n", 
        l2_sub, m_l2_dram_q_accesses.at(streamID)[l2_sub]);
      // fprintf(fout, "\tavg_l2_dram_q_size[sub:%u] = %.3f\n", l2_sub, avg_l2_dram_q_size);
      fprintf(fout, "\t%s[sub:%u] = %.3f = (avg_l2_dram_q_size:%.3f / l2_dram_q_capacity:%u)\n", 
        info, l2_sub, avg_l2_dram_q_occupancy, avg_l2_dram_q_size, l2_dram_q_capacity);
    }
  }
}
void cache_stats::print_l2_icnt_queue_stats(
  FILE *fout, unsigned l2_icnt_q_capacity, u64 streamID, const char *info) const {
  for (auto iter = m_accu_l2_icnt_queue_size.begin(); 
    iter != m_accu_l2_icnt_queue_size.end(); ++iter) {
    if ((streamID != ((u64) - 1)) && (iter->first != streamID)) {
      continue;
    }
    for (unsigned l2_sub = 0; l2_sub < iter->second.size(); ++l2_sub) {       
      float avg_l2_icnt_q_size = 
        (m_accu_l2_icnt_queue_size.at(streamID)[l2_sub] / 
        (float)m_l2_icnt_q_accesses.at(streamID)[l2_sub]);
      float avg_l2_icnt_q_occupancy = avg_l2_icnt_q_size / (float)l2_icnt_q_capacity;
      fprintf(fout, "\tavg_l2_icnt_q_size[sub:%u] = %.3f\n", l2_sub, avg_l2_icnt_q_size);
      fprintf(fout, "\t%s[sub:%u] = %.3f = (avg_size:%.3f / l2_icnt_q_capacity:%u)\n", 
        info, l2_sub, avg_l2_icnt_q_occupancy, avg_l2_icnt_q_size, l2_icnt_q_capacity);
    }
  }
}

void cache_stats::print_l1d_accesses(FILE* fout, u64 streamID, u32 kernel) const {
  auto it_stream = m_l1d_accesses.find(streamID);
  if (it_stream != m_l1d_accesses.end()) {
    std::map<u32 /* kernel */, u32> records = it_stream->second;
    auto it_kernel = records.find(kernel);
    if (it_kernel != records.end()) {
      fprintf(fout, "\tstreamID:%llu kernel:%u L1D_ACCESSES = %u\n", 
        streamID, kernel, records[kernel]);
    }
  }
}
void cache_stats::print_l1d_wr_misses(FILE* fout, u64 streamID, u32 kernel) const {
  auto it_stream = m_l1d_wr_misses.find(streamID);
  if (it_stream != m_l1d_wr_misses.end()) {
    std::map<u32 /* kernel */, u32> records = it_stream->second;
    auto it_kernel = records.find(kernel);
    if (it_kernel != records.end()) {
      fprintf(fout, "\tstreamID:%llu kernel:%u L1D_WR_MISSES = %u\n", 
        streamID, kernel, records[kernel]);
    }
  }
}
void cache_stats::print_l1d_writes(FILE* fout, u64 streamID, u32 kernel) const {
  auto it_stream = m_l1d_writes.find(streamID);
  if (it_stream != m_l1d_writes.end()) {
    std::map<u32 /* kernel */, u32> records = it_stream->second;
    auto it_kernel = records.find(kernel);
    if (it_kernel != records.end()) {
      fprintf(fout, "\tstreamID:%llu kernel:%u L1D_WRITES = %u\n", 
        streamID, kernel, records[kernel]);
    }
  }
}

void cache_stats::print_l1d_n_fill_to_evict_lines(FILE* fout, u64 streamID, u32 kernel) const {
  const LOCALITY_KEY key(streamID, kernel);
  auto it = m_n_l1d_fill_to_evict_lines.find(key);
  if (it != m_n_l1d_fill_to_evict_lines.end()) {
    fprintf(fout, "\tstreamID:%llu kernel:%u L1D_N_FILL_TO_EVICT_LINES = %u\n", 
      streamID, kernel, it->second);
  }
}

u32 cache_stats::print_l1d_max_evictions(FILE* fout, LOCALITY_KEY& loc_key, u64 cycles) const {
  auto it = m_l1d_max_evictions.find(loc_key);
  if (it != m_l1d_max_evictions.end()) {
    fprintf(fout, "\tloc_key: <streamID:%llu kernel:%u> L1D_MAX_EVICTIONS = %u\n", 
      loc_key.stream_id, loc_key.kernel, it->second);
    return it->second;
  }
  return 0;
}
u32 cache_stats::print_l1d_avg_evictions(FILE* fout, LOCALITY_KEY& loc_key, u64 cycles) const {
  auto it = m_l1d_avg_evictions.find(loc_key);
  if (it != m_l1d_avg_evictions.end()) {
    fprintf(fout, "\tloc_key: <streamID:%llu kernel:%u> L1D_AVG_EVICTIONS = %u\n", 
      loc_key.stream_id, loc_key.kernel, it->second);
    return it->second;
  }
  return 0;
}

u32 cache_stats::print_l1d_vc_hits(FILE* fout, LOCALITY_KEY& key) const {
  auto it = m_l1d_vc_hits.find(key);
  if (it != m_l1d_vc_hits.end()) {
    fprintf(fout, "\tkey: <streamID:%llu kernel:%u> L1D_VC_HITS = %u\n", 
      key.stream_id, key.kernel, it->second);
    return it->second;
  }
  return 0;
}
u32 cache_stats::print_l1d_vc_misses(FILE* fout, LOCALITY_KEY& key) const {
  auto it = m_l1d_vc_misses.find(key);
  if (it != m_l1d_vc_misses.end()) {
    fprintf(fout, "\tkey: <streamID:%llu kernel:%u> L1D_VC_MISSES = %u\n", 
      key.stream_id, key.kernel, it->second);
    return it->second;
  }
  return 0;
}
u32 cache_stats::print_l1d_vc_accesses(FILE* fout, LOCALITY_KEY& key) const {
  auto it = m_l1d_vc_accesses.find(key);
  if (it != m_l1d_vc_accesses.end()) {
    fprintf(fout, "\tkey: <streamID:%llu kernel:%u> L1D_VC_ACCESSES = %u\n", 
      key.stream_id, key.kernel, it->second);
    return it->second;
  }
  return 0;
}
void cache_stats::print_l1d_vc_hit_rate(
  FILE* fout, LOCALITY_KEY& key, u32 hits, u32 accesses) const {
  if (accesses) {
    float hit_rate = hits / (float)accesses;
    fprintf(fout, "\tkey: <streamID:%llu kernel:%u> L1D_VC_HIT_RATE = "
      "%f (hits:%u / accesses:%u)\n", 
      key.stream_id, key.kernel, hit_rate, hits, accesses);
  }
}

u32 cache_stats::print_l1d_rd_misses(FILE* fout, u64 streamID, u32 kernel, u64 cycles) const {
  auto it_stream = m_l1d_rd_misses.find(streamID);
  if (it_stream != m_l1d_rd_misses.end()) {
    std::map<u32 /* kernel */, u32> records = it_stream->second;
    auto it_kernel = records.find(kernel);
    if (it_kernel != records.end()) {
      fprintf(fout, "\tstreamID:%llu kernel:%u L1D_RD_MISSES = %u\n", 
        streamID, kernel, records[kernel]);
      return records[kernel];
    } 
    return 0;
  }
  return 0;
}
u32 cache_stats::print_l1d_reads(FILE* fout, u64 streamID, u32 kernel, u64 cycles) const {
  auto it_stream = m_l1d_reads.find(streamID);
  if (it_stream != m_l1d_reads.end()) {
    std::map<u32 /* kernel */, u32> records = it_stream->second;
    auto it_kernel = records.find(kernel);
    if (it_kernel != records.end()) {
      fprintf(fout, "\tstreamID:%llu kernel:%u L1D_READS = %u\n", 
        streamID, kernel, records[kernel]);
      return records[kernel];
    }
    return 0;
  }
  return 0;
}
void cache_stats::print_l1d_rd_miss_rate(
  FILE* fout, u64 streamID, u32 kernel, u32 misses, u32 reads, u64 cycles) const {
  if (reads) {
    float rd_miss_rate = misses / (float)reads;
    fprintf(fout, "\tstreamID:%llu kernel:%u L1D_RD_MISS_RATE = %f (misses:%u / reads:%u)\n", 
      streamID, kernel, rd_miss_rate, misses, reads);
  }
}

void cache_stats::print_l1d_avg_rd_byp_activates(FILE* fout, u64 streamID, u32 kernel) const {
  for (auto iter = m_l1d_rd_byp_activates.begin(); iter != m_l1d_rd_byp_activates.end(); ++iter)
  {
    if ((streamID != ((u64) - 1)) && (iter->first != streamID)) {
      continue;
    }
    std::map<u64, u32> records = m_l1d_rd_byp_activates.at(streamID);
    u32 total_byp_activates = 0;
    for (auto& iter : records) {
      total_byp_activates += iter.second;
    }
    float avg_byp_act = records.empty() ? 0.0f : total_byp_activates / (float)records.size();
    fprintf(fout, "\tstreamID:%llu kernel:%u "
      "l1d_avg_rd_byp_act = %f (total_byp_activates:%u / records.size():%zu)\n", 
      streamID, kernel, avg_byp_act, total_byp_activates, records.size());
  }
}
void cache_stats::print_l1d_avg_rd_byp_deactivates(FILE* fout, u64 streamID, u32 kernel) const {
  for (auto iter = m_l1d_rd_byp_deactivates.begin(); iter != m_l1d_rd_byp_deactivates.end(); ++iter)
  {
    if ((streamID != ((u64) - 1)) && (iter->first != streamID)) {
      continue;
    }
    std::map<u64, u32> records = m_l1d_rd_byp_deactivates.at(streamID);
    u32 total_byp_deactivates = 0;
    for (auto& iter : records) {
      total_byp_deactivates += iter.second;
    }
    float avg_byp_deact = total_byp_deactivates / (float)records.size();
    fprintf(fout, "\tstreamID:%llu kernel:%u "
      "l1d_avg_rd_byp_deact = %f (total_byp_deactivates:%u / records.size():%zu)\n", 
      streamID, kernel, avg_byp_deact, total_byp_deactivates, records.size());
  }
}
void cache_stats::print_l1d_avg_rd_byp_act_rate(FILE* fout, u64 streamID, u32 kernelID) const {
  for (auto iter = m_l1d_rd_byp_activates.begin(); iter != m_l1d_rd_byp_activates.end(); ++iter)
  {
    if ((streamID != ((u64) - 1)) && (iter->first != streamID)) {
      continue;
    }
    std::map<u64, u32> records = m_l1d_rd_byp_activates.at(streamID);
    u32 total_byp_activates = 0;
    for (auto& iter : records) {
      total_byp_activates += iter.second;
    }

    std::map<u32 /* kernel */, u32> l1d_read_records = m_l1d_reads.at(streamID);
    u32 total_l1d_reads = 0;
    for (auto& l1d_read : l1d_read_records) {
      total_l1d_reads += l1d_read.second;
    }
    
    float avg_byp_act_rate = total_byp_activates / (float)total_l1d_reads;
    fprintf(fout, "\tkernel:%u l1d_avg_rd_byp_act_rate = %f "
      "(total_byp_activates:%u / total_l1d_reads:%u)\n", 
      kernelID, avg_byp_act_rate, total_byp_activates, total_l1d_reads);
  }
}

void cache_stats::print_avg_l1d_rd_fill_to_evict_gap(FILE* fout, u64 streamID) const {
  for (auto iter = m_overall_avg_l1d_rd_fill_to_evict_gap.begin();
    iter != m_overall_avg_l1d_rd_fill_to_evict_gap.end(); ++iter)
  {
    if ((streamID != ((u64) - 1)) && (iter->first != streamID)) {
      continue;
    }
    fprintf(fout, "\tm_overall_avg_l1d_rd_fill_to_evict_gap = %llu\n", 
      m_overall_avg_l1d_rd_fill_to_evict_gap.at(streamID));
  }
}
void cache_stats::print_avg_l1d_rd_miss_served_cycles(FILE* fout, u64 streamID) const {
  for (auto iter = m_l1d_rd_miss_served_cycles.begin();
    iter != m_l1d_rd_miss_served_cycles.end(); ++iter)
  {
    if ((streamID != ((u64) - 1)) && (iter->first != streamID)) {
      continue;
    }
    fprintf(fout, "\tavg_l1d_rd_miss_served_cycles = %llu\n", 
      m_l1d_rd_miss_served_cycles.at(streamID));
  }  
}
void cache_stats::print_avg_l1d_wr_miss_served_cycles(FILE* fout, u64 streamID) const {
  for (auto iter = m_l1d_wr_miss_served_cycles.begin();
    iter != m_l1d_wr_miss_served_cycles.end(); ++iter)
  {
    if ((streamID != ((u64) - 1)) && (iter->first != streamID)) {
      continue;
    }
    fprintf(fout, "\tavg_l1d_wr_miss_served_cycles = %llu\n", 
      m_l1d_wr_miss_served_cycles.at(streamID));
  }  
}

void cache_stats::print_avg_l2_miss_served_cycles(FILE* fout, u64 streamID) const {
  for (auto iter = m_l2_sub_miss_served_cycles.begin();
    iter != m_l2_sub_miss_served_cycles.end(); ++iter)
  {
    if ((streamID != ((u64) - 1)) && (iter->first != streamID)) {
      continue;
    }
    u64 avg_l2_miss_served_cycles = 0;
    unsigned m_l2_misses = 0;
    for (unsigned l2_sub = 0; l2_sub < iter->second.size(); ++l2_sub) {
      float avg_l2_sub_miss_served_cycles = 
      ((float)m_l2_sub_miss_served_cycles.at(streamID)[l2_sub] / 
      m_l2_sub_misses.at(streamID)[l2_sub]);

      avg_l2_miss_served_cycles += m_l2_sub_miss_served_cycles.at(streamID)[l2_sub];
      m_l2_misses               += m_l2_sub_misses.at(streamID)[l2_sub];
      fprintf(fout, "\tavg_l2_sub_miss_served_cycles[sub:%u] = %f\n",
        l2_sub, avg_l2_sub_miss_served_cycles);      
    }
    fprintf(fout, "\tavg_l2_miss_served_cycles = %f\n",
      (float)(avg_l2_miss_served_cycles) / m_l2_misses);   
  }  
}

void cache_stats::print_l2_mshr_slots_stats(
  FILE *fout, unsigned l2_mshr_allocated_slots, u64 streamID, const char *info) const {
  for (auto iter = m_l2_mshr_slots_fills.begin(); 
    iter != m_l2_mshr_slots_fills.end(); ++iter) {
    if ((streamID != ((u64) - 1)) && (iter->first != streamID)) {
      continue;
    }
    const unsigned allocatd_mshr_slots = 4; // Replace this with m_config.xx
    for (unsigned l2_sub = 0; l2_sub < iter->second.size(); ++l2_sub) {
      float avg_l2_mshr_slots_size = 
        (m_l2_mshr_slots_fills.at(streamID)[l2_sub] / 
        (float)allocatd_mshr_slots);
      float avg_l2_mshr_slots_occupancy = avg_l2_mshr_slots_size / (float)l2_mshr_allocated_slots;
      fprintf(fout, "\tavg_l2_mshr_slots_size[sub:%u] = %.3f\n", l2_sub, avg_l2_mshr_slots_size);
      fprintf(fout, "\t%s[sub:%u] = %.3f = (avg_size:%.3f / l2_mshr_allocated_slots:%u)\n", 
        info, l2_sub, avg_l2_mshr_slots_occupancy, avg_l2_mshr_slots_size, l2_mshr_allocated_slots);
    }
  }
}

void cache_stats::print_l2_miss_q_pops(FILE *fout, const char *info) const {
  fprintf(fout, "%s = %llu\n", info, m_l2_miss_q_pops);
}

void cache_sub_stats::print_port_stats(FILE *fout, const char *cache_name) const {
  float data_port_util = 0.0f;
  if (port_available_cycles > 0) {
    data_port_util = (float)data_port_busy_cycles / port_available_cycles;
  }
  fprintf(fout, "%s_data_port_util = %.3f\n", cache_name, data_port_util);
  float fill_port_util = 0.0f;
  if (port_available_cycles > 0) {
    fill_port_util = (float)fill_port_busy_cycles / port_available_cycles;
  }
  fprintf(fout, "%s_fill_port_util = %.3f\n", cache_name, fill_port_util);
}

u64 cache_stats::get_stats(
    enum mem_access_type *access_type, unsigned num_access_type,
    enum cache_request_status *access_status,
    unsigned num_access_status) const {
  ///
  /// Returns a sum of the stats corresponding to each "access_type" and
  /// "access_status" pair. "access_type" is an array of "num_access_type"
  /// mem_access_types. "access_status" is an array of "num_access_status"
  /// cache_request_statuses.
  ///
  u64 total = 0;
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    u64 streamID = iter->first;
    for (unsigned type = 0; type < num_access_type; ++type) {
      for (unsigned status = 0; status < num_access_status; ++status) {
        if (!check_valid((int)access_type[type], (int)access_status[status]))
          assert(0 && "Unknown cache access type or access outcome");
        total += m_stats.at(streamID)[access_type[type]][access_status[status]];
      }
    }
  }
  return total;
}

void cache_stats::get_sub_stats(
  struct cache_sub_stats &css, const char* cache_name, u64 time, unsigned kernel) const {
  ///
  /// Overwrites "css" with the appropriate statistics from this cache.
  ///
  struct cache_sub_stats t_css;
  t_css.clear();

  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    u64 streamID = iter->first;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        // if (status == BYPASS_ACTIVATED) {
        //   t_css.avg_rd_byp_activates = 
        //     (t_css.avg_rd_byp_activates + m_l1d_rd_byp_activates.at(streamID)) >> 1;
        // } 
        // else if (status == BYPASS_DEACTIVATED) {
        //   t_css.avg_rd_byp_deactivates = 
        //     (t_css.avg_rd_byp_deactivates + m_l1d_rd_byp_deactivates.at(streamID)) >> 1;
        // }
        t_css.n_bypassed += m_stats.at(streamID)[type][status];        

        if (status == HIT || status == MISS || status == SECTOR_MISS || status == HIT_RESERVED) {
          t_css.accesses += m_stats.at(streamID)[type][status];
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R || type == INST_ACC_R) {
            t_css.reads += m_stats.at(streamID)[type][status];
          } else {
            t_css.writes += m_stats.at(streamID)[type][status];
          }
        }
        if (status == MISS) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R || type == INST_ACC_R) {            
            // 3/18 Commented to check if "rd_misses = 0" in perf_rpt.o (Yes)
            // L1D_rd_miss_rate = 0.4505 = (rd_misses:0 + sector_rd_misses:2153) / reads:4779
            // t_css.clear() is called before embedded loop 
            // "+=" accumulates [type][status] for this round only
            [[maybe_unused]] u64 prev_rd_misses = t_css.rd_misses;
            t_css.rd_misses += m_stats.at(streamID)[type][status];
            if (DTRACE(DEBUG_STATS)) {
              if (m_stats.at(streamID)[type][status]) {
                fprintf(Trace::out, "%llu kernel:%u %s t_css.rd_misses:%llu += "
                  "m_stats.at(streamID:%llu)[type:%u][status:%s]:%llu (prev_rd_misses:%llu->%llu)\n", 
                  time, kernel, cache_name, t_css.rd_misses, 
                  streamID, type, cache_request_status_str(cache_request_status(status)), 
                  m_stats.at(streamID)[type][status], 
                  prev_rd_misses, t_css.rd_misses);                
              }
            }
          } else if (type == GLOBAL_ACC_W) {
            t_css.wr_misses += m_stats.at(streamID)[type][status];
          }
          t_css.misses += m_stats.at(streamID)[type][status];
          t_css.avg_evict_interval += m_evict_stats.at(streamID);          
        }
        if (status == SECTOR_MISS) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R || type == INST_ACC_R) {
            t_css.sector_rd_misses += m_stats.at(streamID)[type][status];
            // for debug
            if (DTRACE(TRACE_RD_MISS_CNT)) {
              fprintf(Trace::out, "t_css.sector_rd_misses:%llu += "
                "m_stats.at(streamID)[type][status]:%llu\n", 
                t_css.sector_rd_misses, m_stats.at(streamID)[type][status]);
            }
          } else {
            t_css.sector_wr_misses += m_stats.at(streamID)[type][status];
          }      
          t_css.sector_misses += m_stats.at(streamID)[type][status];
        }        
        if (status == HIT_RESERVED) {
          t_css.pending_hits += m_stats.at(streamID)[type][status];
        }
        if (status == RESERVATION_FAIL)
          t_css.res_fails += m_stats.at(streamID)[type][status];
      }
    }
  }

  t_css.port_available_cycles = m_cache_port_available_cycles;
  t_css.data_port_busy_cycles = m_cache_data_port_busy_cycles;
  t_css.fill_port_busy_cycles = m_cache_fill_port_busy_cycles;

  css = t_css;
}

void cache_stats::get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
  ///
  /// Overwrites "css" with the appropriate statistics from this cache.
  ///
  struct cache_sub_stats_pw t_css;
  t_css.clear();

  for (auto iter = m_stats_pw.begin(); iter != m_stats_pw.end(); ++iter) {
    u64 streamID = iter->first;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        if (status == HIT || status == MISS ||
            status == SECTOR_MISS || status == HIT_RESERVED) {
          t_css.accesses += m_stats_pw.at(streamID)[type][status];
        }
        if (status == HIT) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R || type == INST_ACC_R) {
            t_css.read_hits += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_hits += m_stats_pw.at(streamID)[type][status];
          }
        }

        if (status == MISS || status == SECTOR_MISS) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            t_css.read_misses += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_misses += m_stats_pw.at(streamID)[type][status];
          }
        }

        if (status == HIT_RESERVED) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            t_css.read_pending_hits += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_pending_hits += m_stats_pw.at(streamID)[type][status];
          }
        }

        if (status == RESERVATION_FAIL) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            t_css.read_res_fails += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_res_fails += m_stats_pw.at(streamID)[type][status];
          }
        }
      }
    }
  }

  css = t_css;
}

bool cache_stats::check_valid(int type, int status) const {
  ///
  /// Verify a valid access_type/access_status
  ///
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (status >= 0) &&
      (status < NUM_CACHE_REQUEST_STATUS))
    return true;
  else
    return false;
}

bool cache_stats::check_valid(unsigned sm, unsigned warp) const {
  ///
  /// Verify a valid sm / warp index for MSHR occupancy stats
  ///
  const unsigned sms = 4; // gpgpu_n_cores_per_cluster
  const unsigned max_warps_per_sm = 64; // m_config.max_warps_per_sm  
  if ((sm >= 0) && (sm < sms) && (warp >= 0) && (warp < max_warps_per_sm))
    return true;
  else
    return false;
}

bool cache_stats::check_fail_valid(int type, int fail) const {
  ///
  /// Verify a valid access_type/access_status
  ///
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (fail >= 0) &&
      (fail < NUM_CACHE_RESERVATION_FAIL_STATUS))
    return true;
  else
    return false;
}

void cache_stats::sample_cache_port_utility(bool data_port_busy,
                                            bool fill_port_busy) {
  m_cache_port_available_cycles += 1;
  if (data_port_busy) {
    m_cache_data_port_busy_cycles += 1;
  }
  if (fill_port_busy) {
    m_cache_fill_port_busy_cycles += 1;
  }
}

void baseline_cache::dump_cache_access_info(
  const char* caller,
  new_addr_type addr, mem_fetch *mf, u64 time, 
  enum cache_request_status status,
  bool dump_inst_str) {

  new_addr_type block_addr = m_config.block_addr(addr);

  // std::pair<std::bitset<128>, std::bitset<128>> u128_pair = to_u128_pair(mf->get_access_byte_mask());  
  std::pair<uint64_t,uint64_t> byte_mask_hi_lo = to_u64_pair(mf->get_access_byte_mask());
  fprintf(Trace::out,
      "%llu %s%s%s %s %s addr: %#llx block_addr: %#llx "
      "byte_mask: 0x%016lx%016lx\n",
      (u64)time,
      caller,
      dump_inst_str ? m_gpu->gpgpu_ctx->func_sim->ptx_get_insn_str(mf->get_inst().pc).c_str() : "",
      m_is_l1d ? "L1D" : m_is_l2 ? "L2C" : "xx$",
      mf_request_type_str(mf->get_type()),
      cache_request_status_str(status), 
      (u64)mf->get_addr(),
      (u64)block_addr,
      byte_mask_hi_lo.first, byte_mask_hi_lo.second
    );
}

void baseline_cache::dump_cache_fill_info(
  const char* caller,
  new_addr_type addr, mem_fetch *mf, u64 time, 
  bool dump_inst_str) {

  new_addr_type block_addr = m_config.block_addr(addr);

  std::pair<uint64_t,uint64_t> byte_mask_hi_lo = to_u64_pair(mf->get_access_byte_mask());
  fprintf(Trace::out,
      "%llu %s%s%s %s addr: %#llx block_addr: %#llx "
      "byte_mask: 0x%016lx%016lx\n",
      (u64)time,
      caller,
      dump_inst_str ? m_gpu->gpgpu_ctx->func_sim->ptx_get_insn_str(mf->get_inst().pc).c_str() : "",      
      m_is_l1d ? "L1D" : m_is_l2 ? "L2C" : "xx$",
      mf_request_type_str(mf->get_type()),
      (u64)mf->get_addr(),
      (u64)block_addr,
      byte_mask_hi_lo.first, byte_mask_hi_lo.second
    );
}

baseline_cache::bandwidth_management::bandwidth_management(cache_config &config)
    : m_config(config) {
  m_data_port_occupied_cycles = 0;
  m_fill_port_occupied_cycles = 0;
}

/// use the data port based on the outcome and events generated by the mem_fetch
/// request
void baseline_cache::bandwidth_management::use_data_port(
    mem_fetch *mf, enum cache_request_status outcome,
    const std::list<cache_event> &events, bool is_wr) {
  unsigned data_size = mf->get_data_size();
  unsigned port_width = m_config.m_data_port_width;
  switch (outcome) {
    case HIT: 
    case VC_HIT: {
      unsigned data_cycles =
          data_size / port_width + ((data_size % port_width > 0) ? 1 : 0);
      m_data_port_occupied_cycles += data_cycles;
    } break;
    case HIT_RESERVED:
    case MISS: {
      // the data array is accessed to read out the entire line for write-back
      // in case of sector cache we need to write bank only the modified sectors
      cache_event ev(WRITE_BACK_REQUEST_SENT);
      if (was_writeback_sent(events, ev)) {
        unsigned data_cycles = ev.m_evicted_block.m_modified_size / port_width;
        m_data_port_occupied_cycles += data_cycles;
      }
    } break;
    case SECTOR_MISS:
    case RESERVATION_FAIL:
      // Does not consume any port bandwidth
      break;
    default:
      assert(0);
      break;
  }
}

/// use the fill port
void baseline_cache::bandwidth_management::use_fill_port(mem_fetch *mf) {
  // assume filling the entire line with the returned request
  unsigned fill_cycles = m_config.get_atom_sz() / m_config.m_data_port_width;
  m_fill_port_occupied_cycles += fill_cycles;
}

/// called every cache cycle to free up the ports
void baseline_cache::bandwidth_management::replenish_port_bandwidth() {
  if (m_data_port_occupied_cycles > 0) {
    m_data_port_occupied_cycles -= 1;
  }
  assert(m_data_port_occupied_cycles >= 0);

  if (m_fill_port_occupied_cycles > 0) {
    m_fill_port_occupied_cycles -= 1;
  }
  assert(m_fill_port_occupied_cycles >= 0);
}

/// query for data port availability
bool baseline_cache::bandwidth_management::data_port_free() const {
  return (m_data_port_occupied_cycles == 0);
}

/// query for fill port availability
bool baseline_cache::bandwidth_management::fill_port_free() const {
  return (m_fill_port_occupied_cycles == 0);
}

/// Sends next request to lower level of memory
void baseline_cache::cycle() {
  if (!m_miss_queue.empty()) {
    mem_fetch *mf = m_miss_queue.front();

    if (!m_memport->full(mf->size(), mf->get_is_write())) {
      if (this->m_is_l2) {
        m_stats.inc_l2_miss_q_pops();
        m_memport->set_if_name("L2-IF");
      } else if (m_is_l1d) {
        m_memport->set_if_name("L1D-IF");
      }

      m_miss_queue.pop_front();
      m_memport->push(mf);

      if (DTRACE(CACHE_EVENT)) {
        std::string event_name = "m_miss_queue.pop_front ---mf---> ";
        event_name += m_memport->get_if_name();
        event_name += " (NOC)";
        const char* cstr_event_name = event_name.c_str();
        dumpCacheEvent(
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, 
          "bandwidth_management::fill_port_free", cstr_event_name, mf);
      }
    }
  }
  bool data_port_busy = !m_bandwidth_management.data_port_free();
  bool fill_port_busy = !m_bandwidth_management.fill_port_free();
  m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
  m_bandwidth_management.replenish_port_bandwidth();
}

/// Interface for response from lower memory level (model bandwidth restictions
/// in caller)
void baseline_cache::fill(mem_fetch *mf, u64 time) {
  
  if (DTRACE(CACHELINE_FILL)) {
    if (mf) {
      fprintf(Trace::out, "%llu %s baseline_cache::fill mf "
        "[sid:%u][warp:%u][addr:%#llx]\n",
        time, m_config.get_cache_name(), 
        mf->get_sid(), mf->get_wid(), mf->get_addr());
    }
  }

  if (m_config.m_mshr_type == SECTOR_ASSOC) {
    assert(mf->get_original_mf());
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf->get_original_mf());
    assert(e != m_extra_mf_fields.end());
    e->second.pending_read--;

    if (e->second.pending_read > 0) {
      // wait for the other requests to come back
      delete mf;
      return;
    } else {
      mem_fetch *temp = mf;
      mf = mf->get_original_mf();
      delete temp;
    }
  }

  // [GPGPU-SIM][TODO]
  // assert(mf->get_original_mf());
  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);

  BYPASS_KEY byp_key(
    mf->get_streamID(), m_gpu->m_kernel_id, m_config.block_addr(mf->get_addr()));
  if (m_config.m_bypass_mode == 1 && m_tag_array->hit_l1d_bypassed_item(byp_key, mf)) {
    // nothing
  } else {
    assert(e != m_extra_mf_fields.end());
    assert(e->second.m_valid);
  }

  // 3/21 Commented because following assignment indicates redundant logic
  // https://github.com/gpgpu-sim/gpgpu-sim_distribution/issues/337
  // mf->set_data_size(e->second.m_data_size);
  // mf->set_addr(e->second.m_addr);

  if (m_config.m_alloc_policy == ON_MISS) {
    if (m_config.m_bypass_enable == 'T') {
      assert(m_is_l1d);
      LOCALITY_KEY loc_key(mf->get_streamID(), m_gpu->m_kernel_id);
      if (m_config.m_bypass_mode == 1 &&
          m_tag_array->hit_l1d_bypassed_item(byp_key, mf)) {

        if (DTRACE(HIT_L1D_BYPASSED_ITEM)) {
          dumpCacheEvent(time, "baseline_cache::fill", 
            "HIT_L1D_BYPASSED_ITEM L1D bypassed m_tag_array->fill", mf);
        }
      } else {
        assert(e->second.m_cache_index != (u32) - 1);
        m_tag_array->fill(this, e->second.m_cache_index, time, mf);
      }
      // Always fill victim cache
      // m_victim_cache->fill(e->second.m_cache_index, time, mf);
    } else {
      if (DTRACE(CACHE_EVENT) || DTRACE(L1D_FILLS)) {
        dumpCacheEvent(time, "::fill", "ap:ON_MISS m_tag_array->fill", mf);
      }
      m_tag_array->fill(this, e->second.m_cache_index, time, mf);
    }
    // m_tag_array->fill(this, e->second.m_cache_index, time, mf); // default logic
  }
  else if (m_config.m_alloc_policy == ON_FILL) {
    if (DTRACE(CACHE_EVENT)) {      
      dumpCacheEvent(time, "::fill", "ap:ON_FILL m_tag_array->fill", mf);
    }     
    m_tag_array->fill(this, e->second.m_block_addr, time, mf, mf->is_write());
  } else {
    abort();
  }

  bool has_atomic = false;
  if (m_config.m_mshr_disable == 'T') {
    has_atomic = mf->isatomic();
    m_lfb.push_back(mf); // !!! L1D bypass should only be applied to replacement    
    if (m_config.m_bypass_mode == 1 && 
      m_config.m_bypass_enable == 'T' && 
      m_tag_array->hit_l1d_bypassed_item(byp_key, mf)) {
      assert(m_is_l1d);
      m_tag_array->m_l1d_occupied[byp_key] = true;
      if (DTRACE(TRACE_BYPASSED_L1D_PKT) || DTRACE(HIT_L1D_BYPASSED_ITEM) ||
          DTRACE(REFILL_LFB_BYP_FILL_TAG)) {
        dumpCacheEvent(time, "baseline_cache::fill", 
          "HIT_L1D_BYPASSED_ITEM Bypassed L1D pkt should still m_lfb.push_back", mf);
      }
    }

    if (m_is_l2) {
      m_stats.inc_l2_sub_miss_served_cycles(
        mf->get_streamID(), mf->get_sub_partition(),
        time - mf->m_miss_serve_begin_time);
      m_stats.inc_l2_sub_misses(mf->get_streamID(), mf->get_sub_partition());
    } else if (m_is_l1d) {
      if (!mf->is_write() && !mf->isatomic()) {
        m_tag_array->set_l1d_rd_fill_time(byp_key, time);
        mf->set_l1d_rd_miss_served_time(time - mf->m_l1d_rd_miss_serve_begin_time);
        m_stats.avg_l1d_rd_miss_served_cycles(
          mf->get_streamID(), time - mf->m_l1d_rd_miss_serve_begin_time);
      } else if (mf->is_write()) {
        m_stats.avg_l1d_wr_miss_served_cycles(
          mf->get_streamID(), time - mf->m_wr_miss_serve_begin_time);
      }
    }
  } else {
    m_mshrs.mark_ready(m_is_l2 ? "L2" : m_is_l1d ? "L1D" : "other$", 
      e->second.m_block_addr, has_atomic, time);
    m_tag_array->m_total_records_in_mshr++;

    if (m_is_l2) {
      m_stats.inc_l2_sub_miss_served_cycles(
        mf->get_streamID(), mf->get_sub_partition(),
        time - mf->m_miss_serve_begin_time);
      m_stats.inc_l2_sub_misses(mf->get_streamID(), mf->get_sub_partition());
    } else if (m_is_l1d) {
      if (!mf->is_write() && !mf->isatomic()) {
        m_stats.avg_l1d_rd_miss_served_cycles(
          mf->get_streamID(), time - mf->m_l1d_rd_miss_serve_begin_time);
      } else if (mf->is_write()) {
        m_stats.avg_l1d_wr_miss_served_cycles(
          mf->get_streamID(), time - mf->m_wr_miss_serve_begin_time);
      }
    }
  }

  if (has_atomic) {
    assert(m_config.m_alloc_policy == ON_MISS);
    cache_block_t *block = m_tag_array->get_block(e->second.m_cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    // mark line as dirty for atomic operation
    block->set_status(
      time, MODIFIED, mf->get_access_sector_mask(), 
      "baseline_cache::fill()" + std::string(m_config.get_cache_name()));
    block->set_byte_mask(mf);
  }
  if (m_config.m_bypass_mode == 1 && m_tag_array->hit_l1d_bypassed_item(byp_key, mf)) {
    // nothing
  } else {
    m_extra_mf_fields.erase(mf);
  }

  m_bandwidth_management.use_fill_port(mf);

  if (DTRACE(EXTRA_MF_EVENT)) {
    dumpCacheEvent(time, 
      "baseline_cache::fill", "m_extra_mf_fields.erase(mf)", mf);
  }
}

/// Checks if mf is waiting to be filled by lower memory level
bool baseline_cache::waiting_for_fill(mem_fetch *mf) {
  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  return e != m_extra_mf_fields.end();
}

void baseline_cache::print(FILE *fp, unsigned &accesses,
                           unsigned &misses) const {
  fprintf(fp, "Cache %s:\t", m_name.c_str());
  m_tag_array->print(fp, accesses, misses);
}

void baseline_cache::display_state(FILE *fp) const {
  fprintf(fp, "Cache %s:\n", m_name.c_str());
  m_mshrs.display(fp);
  fprintf(fp, "\n");
}

void baseline_cache::dumpCacheEvent(
  u64 time, const char* stage, const char* event, mem_fetch *mf, bool short_info) {

  const char* cache_name = m_config.get_cache_name(); // {L1D, L2, ...} with no suffix
  std::string suffix = "";
  if (m_is_l2 && mf) {
    suffix = "_Sub";
    if (mf->get_sub_partition() == (unsigned) - 1) {
      assert(0);
    }
    suffix += std::to_string(mf->get_sub_partition());
  }

  if (mf == nullptr) {
    fprintf(Trace::out, "%llu %s%s stage(%s) cache_event(%s) "
      "mf = nullptr\n", 
      time, cache_name, suffix.c_str(), stage, event
    );
  } else {
    if (short_info) {
      fprintf(Trace::out, "%llu %s%s stage(%s) cache_event(%s) "
        "<streamID:%llu kernel:%u req_uid:%u uid:%u block_addr:%#llx addr:%#llx>\n", 
        time, cache_name, suffix.c_str(), stage, event,
        mf->get_streamID(), m_gpu->m_kernel_id,
        mf->get_request_uid(), mf->get_inst().get_uid(), 
        m_config.block_addr(mf->get_addr()),
        mf->get_addr());
    } else {
      fprintf(Trace::out, "%llu %s%s stage(%s) cache_event(%s) "
        "mf:{streamID:%llu kernel:%u TPC:%u SM:%u WARP:%u req_uid:%u uid:%u block_addr:%#llx addr:%#llx acc_type:%s pos:%s}\n", 
        time, cache_name, suffix.c_str(), stage, event,
        mf->get_streamID(), m_gpu->m_kernel_id,
        mf->get_tpc(), mf->get_sid(), mf->get_wid(), 
        mf->get_request_uid(), mf->get_inst().get_uid(), 
        m_config.block_addr(mf->get_addr()),
        mf->get_addr(), // mf info
        mem_access_type_str(mem_access_type(mf->get_access_type())),
        mf->mem_fetch_status_str(mf->get_status())
      );  
    }
  }
}

void baseline_cache::dumpMSHREvent(
  u64 time, mem_fetch *mf, new_addr_type mshr_addr, bool is_new_entry) {

  const char* cache_name = m_config.get_cache_name();
  std::string l2_prefix = "";
  if (strcmp(cache_name, "L2") == 0) {
    l2_prefix = "SG";
    l2_prefix += std::to_string(mf->get_sub_partition());
    l2_prefix += " ";      
  }  
  const char* action = is_new_entry ? "Created" : "Inserted";
  fprintf(Trace::out, "%llu %s %s"
    "%s mf:{TPC:%u SM:%u WARP:%u req_uid:%u addr:%#llx acc_type:%s pos:%s} %s MSHR[mshr_addr:%#llx] "
    "entries: {occupancy=(%u/%u)=%f}; slots: {occupancy=(%u/%u)=%f}\n",
    time, cache_name, l2_prefix.c_str(), action,
    mf->get_tpc(), mf->get_sid(), mf->get_wid(), mf->get_request_uid(), mf->get_addr(), // mf info
    mem_access_type_str(mem_access_type(mf->get_access_type())),
    mf->mem_fetch_status_str(mf->get_status()),
    is_new_entry ? "in" : "into",
    mshr_addr, 
    m_mshrs.occupied_entries(), m_config.m_mshr_entries,
    m_mshrs.occupied_entries() / (float)m_config.m_mshr_entries,
    m_mshrs.merged_slots(mshr_addr), m_mshrs.get_max_merged(),
    m_mshrs.merged_slots(mshr_addr) / (float)m_mshrs.get_max_merged()
  );
}

void baseline_cache::dumpMissQueue(u64 time, const char* stage, const char* event, mem_fetch *mf) {

  const char* cache_name = m_config.get_cache_name();
  std::string l2_prefix = "";
  if (strcmp(cache_name, "L2") == 0) {
    l2_prefix = "SG";
    l2_prefix += std::to_string(mf->get_sub_partition());
    l2_prefix += " ";      
  }
  fprintf(Trace::out, "%llu %s %s%s %s"
    "mf:{TPC:%u SM:%u WARP:%u req_uid:%u addr:%#llx acc_type:%s pos:%s} "
    "-> MissQueue: {occupancy=(%lu/%u)=%f}\n",
    time, cache_name, l2_prefix.c_str(), stage, event,
    mf->get_tpc(), mf->get_sid(), mf->get_wid(), mf->get_request_uid(), mf->get_addr(), // mf info
    mem_access_type_str(mem_access_type(mf->get_access_type())),
    mf->mem_fetch_status_str(mf->get_status()),
    m_miss_queue.size(), m_config.m_miss_queue_size,
    m_miss_queue.size() / (float)m_config.m_miss_queue_size
  );
}

void baseline_cache::inc_aggregated_stats(cache_request_status status,
                                          cache_request_status cache_status,
                                          mem_fetch *mf,
                                          enum cache_gpu_level level) {
  if (level == L1_GPU_CACHE) {
    m_gpu->aggregated_l1_stats.inc_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l1_stats.select_stats_status(status, cache_status));
  } else if (level == L2_GPU_CACHE) {
    m_gpu->aggregated_l2_stats.inc_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l2_stats.select_stats_status(status, cache_status));
  }
}

void baseline_cache::inc_aggregated_fail_stats(
    cache_request_status status, cache_request_status cache_status,
    mem_fetch *mf, enum cache_gpu_level level) {
  if (level == L1_GPU_CACHE) {
    m_gpu->aggregated_l1_stats.inc_fail_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l1_stats.select_stats_status(status, cache_status));
  } else if (level == L2_GPU_CACHE) {
    m_gpu->aggregated_l2_stats.inc_fail_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l2_stats.select_stats_status(status, cache_status));
  }
}

void baseline_cache::inc_aggregated_stats_pw(cache_request_status status,
                                             cache_request_status cache_status,
                                             mem_fetch *mf,
                                             enum cache_gpu_level level) {
  if (level == L1_GPU_CACHE) {
    m_gpu->aggregated_l1_stats.inc_stats_pw(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l1_stats.select_stats_status(status, cache_status));
  } else if (level == L2_GPU_CACHE) {
    m_gpu->aggregated_l2_stats.inc_stats_pw(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l2_stats.select_stats_status(status, cache_status));
  }
}

/// Read miss handler without writeback
void baseline_cache::send_read_request(new_addr_type raw_addr, new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       u64 time, bool &do_miss,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  bool wb = false;
  evicted_block_info e;
  send_read_request(raw_addr, block_addr, cache_index, mf, time, 
                    do_miss, wb, e,
                    events, read_only, wa);
}

/// Read miss handler. Check MSHR hit or MSHR available
void baseline_cache::send_read_request(new_addr_type raw_addr, new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       u64 time, bool &do_miss, bool &wb,
                                       evicted_block_info &evicted,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {

  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());
  assert(raw_addr == mf->get_addr());

  if (DTRACE(L2_SEND_RD_REQ)) { // Yes. Can reach
    fprintf(Trace::out, "%llu L2 send_read_request "
      "mf:{TPC:%u SM:%u WARP:%u req_uid:%u addr:%#llx} "
      "block_addr: %#llx mshr_addr: %#llx\n",
      time, mf->get_tpc(), mf->get_sid(), mf->get_wid(), mf->get_request_uid(),
      mf->get_addr(), block_addr, mshr_addr);
  }

  [[maybe_unused]] const char* cache_type = m_is_l2 ? "L2" : m_is_l1d ? "L1D" : "OTHER$";  
  if (m_config.m_mshr_disable == 'T') {
    // No MSHR: only gate on miss_queue capacity, no merge or ready tracking.
    if (m_miss_queue.size() < m_config.m_miss_queue_size) {
      if (read_only) {
        m_tag_array->access(this,raw_addr, block_addr, time, cache_index, mf);
      } else {
        if (m_config.m_bypass_enable == 'T') {
          assert(m_is_l1d);
          BYPASS_KEY byp_key(mf->get_streamID(), m_gpu->m_kernel_id, block_addr);
          if (m_config.m_bypass_mode == 1 && m_tag_array->hit_l1d_bypassed_item(byp_key, mf)) {
            if (DTRACE(TRACE_BYPASSED_L1D_PKT) || 
                DTRACE(BYPASS_L1D_ALLOC) || DTRACE(HIT_L1D_BYPASSED_ITEM)) {
              dumpCacheEvent(time, "baseline_cache::send_read_request", 
                "HIT_L1D_BYPASSED_ITEM Bypassed m_tag_array->access", mf);
            }
          } else {
            m_tag_array->access(this,raw_addr, block_addr, time, cache_index, wb, evicted, mf);           
          }
        } else {
          m_tag_array->access(this,raw_addr, block_addr, time, cache_index, wb, evicted, mf);
        }
      }

      // case can pass
      BYPASS_KEY byp_key(mf->get_streamID(), m_gpu->m_kernel_id, block_addr);
      if (m_is_l1d && m_config.m_bypass_mode == 1 && 
          m_tag_array->hit_l1d_bypassed_item(byp_key, mf)) {
        // nothing
      } else {
        m_extra_mf_fields[mf] = extra_mf_fields(
          mshr_addr, mf->get_addr(), 
          cache_index, mf->get_data_size(), 
          m_config, m_tag_array->hit_l1d_bypassed_item(byp_key, mf));   
      }

      // BYPASS_KEY byp_key(mf->get_streamID(), m_gpu->m_kernel_id, block_addr);
      // m_extra_mf_fields[mf] = extra_mf_fields(
      //     mshr_addr, mf->get_addr(), 
      //     cache_index, mf->get_data_size(), 
      //     m_config, m_tag_array->hit_l1d_bypassed_item(byp_key, mf));
      
      mf->set_data_size(m_config.get_atom_sz());
      mf->set_addr(mshr_addr);
      mf->set_status(m_miss_queue_status, time);
      m_miss_queue.push_back(mf);

      if (DTRACE(CACHE_EVENT) || DTRACE(MISS_QUEUE_EVENT) || DTRACE(EXTRA_MF_EVENT)) {
        dumpCacheEvent(time, "baseline_cache::send_read_request", 
          "1. new extra_mf_fields[mf] "
          "2. mf set MetaData {data_size, addr, status} "
          "3. m_miss_queue.push_back(mf)", mf);
      }
      // if (DTRACE(TRACE_BYPASSED_L1D_PKT) || DTRACE(HIT_L1D_BYPASSED_ITEM)) {
      //   if (m_tag_array->hit_l1d_bypassed_item(byp_key, mf)) {
      //     dumpCacheEvent(time, "baseline_cache::send_read_request", 
      //       "HIT_L1D_BYPASSED_ITEM l1d_bypassed_item should also "
      //       "1. new extra_mf_fields[mf] "
      //       "2. mf set MetaData {data_size, addr, status} "
      //       "3. m_miss_queue.push_back(mf)", mf);          
      //   }
      // }

      if (!wa) {        
        events.push_back(cache_event(READ_REQUEST_SENT));
        if (DTRACE(CACHE_EVENT)) {
          dumpCacheEvent(time, "RD-MISS-NO-MSHR", 
            "!wa events.push_back(cache_event(READ_REQUEST_SENT))", mf);
        }
      }
      do_miss = true;
    } else {
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                            mf->get_streamID(), miss_queue_full_driver::RD_MISS);
    }
  } // m_config.m_mshr_disable == 'T' 
  else {
    bool mshr_hit   = m_mshrs.probe(mshr_addr);
    bool mshr_avail = !m_mshrs.full(mshr_addr);
    if (mshr_hit && mshr_avail) {
      if (read_only) {
        m_tag_array->access(this,raw_addr, block_addr, time, cache_index, mf);
      } else {
        m_tag_array->access(this,raw_addr, block_addr, time, cache_index, wb, evicted, mf);
      }

      const size_t last_occupied_entries = m_mshrs.occupied_entries();

      if (DTRACE(DUMP_MSHR)) {
        fprintf(Trace::out, "%llu %s MSHR Hit! Before adding, MSHR is below:\n", 
          time, cache_type);
        m_mshrs.display(Trace::out, cache_type);      
      }

      bool is_new_mshr_entry = false;

      // if (mf && m_is_l1d) {
      //   m_tag_array->m_l1d_mshr_recorded_addr[mf->get_sid()].insert(block_addr);
      // } else if (mf && m_is_l2) {
      //   m_tag_array->m_l2_mshr_recorded_addr[mf->get_sub_partition()].insert(block_addr);
      // }

      m_mshrs.add(mshr_addr, mf, is_new_mshr_entry, cache_type); // orig GPGPU-SIM logic

      if (m_is_l2) {
        m_stats.inc_l2_mshr_slots_fills(mf->get_streamID(), mf->get_sub_partition());
      }
      if (DTRACE(RECORDED_IN_MSHR)) {
        fprintf(Trace::out, "%llu %s block_addr %#llx {tag %#llx set_index %#x} "
          "is recorded in MSHR (mshr hit) occupied_slots[mshr_addr:%#llx] = %u. "
          "mf:{ TPC:%u SM:%u WARP:%u req_uid:%u}\n",
          time, cache_type, block_addr, 
          m_config.tag(block_addr), m_config.set_index(block_addr, mf->get_wid()),
          mshr_addr, m_mshrs.occupied_slots(mshr_addr),
          mf->get_tpc(), mf->get_sid(), mf->get_wid(), mf->get_request_uid()
        );         
      }      

      if (DTRACE(CACHE_EVENT) || DTRACE(MSHR_EVENT)) {
        dumpMSHREvent(time, mf, mshr_addr, is_new_mshr_entry);
      }

      m_stats.inc_mshr_stats(mf->get_streamID(), mf->get_sid(), mf->get_wid());
      if (DTRACE(CACHE_REQ_DIST) || DTRACE(MSHR_ACCESS)) {
        assert(last_occupied_entries == m_mshrs.occupied_entries());
        fprintf(Trace::out, "%llu TPC:%u SM:%u WARP:%u "
          "%s MSHR Hit "
          "entries: {occupied:%u free:%u occupancy:%f} "
          "slots: {merged:%u free:%u occupancy:%f} "
          "Added slot {mshr_addr: %#llx, addr: %#llx} "
          "m_mshr_occupancy_stats[streamID:%llu][sm:%u][warp:%u] = %u\n", 
          time, mf->get_tpc(), mf->get_sid(), mf->get_wid(),
          cache_type, 
          m_mshrs.occupied_entries(), 
          m_config.m_mshr_entries - m_mshrs.occupied_entries(), 
          m_mshrs.occupied_entries() / (float)m_config.m_mshr_entries,
          m_mshrs.merged_slots(mshr_addr),
          m_mshrs.get_max_merged() - m_mshrs.merged_slots(mshr_addr),
          m_mshrs.merged_slots(mshr_addr) / (float)m_mshrs.get_max_merged(),
          mshr_addr, block_addr,
          mf->get_streamID(), mf->get_sid(), mf->get_wid(),
          m_stats.get_mshr_merge_dist_cnt(mf->get_streamID(), mf->get_sid(), mf->get_wid()));
      }
      if (DTRACE(DUMP_MSHR)) {
        fprintf(Trace::out, "%llu %s MSHR Hit! After adding, MSHR is below:\n", time, cache_type);
        m_mshrs.display(Trace::out);
      }

      m_stats.inc_stats(mf->get_access_type(), MSHR_HIT, mf->get_streamID());
      m_stats.update_evict_stats(mf->get_streamID(), mf->get_victim_avg_evict_interval());
      do_miss = true;
    } else if (!mshr_hit && mshr_avail &&
              (m_miss_queue.size() < m_config.m_miss_queue_size)) {
      if (read_only) {
        m_tag_array->access(this,raw_addr, block_addr, time, cache_index, mf);
      } else {
        m_tag_array->access(this,raw_addr, block_addr, time, cache_index, wb, evicted, mf);
      }

      [[maybe_unused]] const unsigned last_occupied_entries = m_mshrs.occupied_entries();
      if (DTRACE(DUMP_MSHR)) {
        fprintf(Trace::out, "%llu TPC:%u SM:%u WARP:%u "
          "%s MSHR Miss! Before adding, MSHR is below:\n", 
          time, mf->get_tpc(), mf->get_sid(), mf->get_wid(),
          cache_type);
        m_mshrs.display(Trace::out, cache_type);
      }

      bool is_new_mshr_entry = false;

      if (m_config.m_bypass_enable == 'T') {
        BYPASS_KEY byp_key(mf->get_streamID(), m_gpu->m_kernel_id, block_addr);
        if (m_config.m_bypass_mode == 1 && m_tag_array->hit_l1d_bypassed_item(byp_key, mf)) {
            // nothing
        } else {
          m_tag_array->set_recorded_in_mshr(cache_index);
        }
      } else {
        m_tag_array->set_recorded_in_mshr(cache_index);
      }
      // m_tag_array->set_recorded_in_mshr(cache_index); // default logic

      m_mshrs.add(mshr_addr, mf, is_new_mshr_entry, cache_type); // orig GPGPU-SIM logic      
      if (m_is_l2) {
        m_stats.inc_l2_mshr_slots_fills(mf->get_streamID(), mf->get_sub_partition());
      }

      if (DTRACE(CACHE_EVENT) || DTRACE(MSHR_EVENT)) {
        dumpMSHREvent(time, mf, mshr_addr, is_new_mshr_entry);
      }

      m_stats.inc_mshr_stats(mf->get_streamID(), mf->get_sid(), mf->get_wid());

      // case can pass
      BYPASS_KEY byp_key(mf->get_streamID(), m_gpu->m_kernel_id, block_addr);
      m_extra_mf_fields[mf] = extra_mf_fields(
          mshr_addr, mf->get_addr(), cache_index, 
          mf->get_data_size(), 
          m_config, m_tag_array->hit_l1d_bypassed_item(byp_key, mf));

      mf->set_data_size(m_config.get_atom_sz());
      mf->set_addr(mshr_addr);
      mf->set_status(m_miss_queue_status, time);
      m_miss_queue.push_back(mf);

      if (DTRACE(CACHE_EVENT) || DTRACE(MISS_QUEUE_EVENT)) {
        dumpMissQueue(time, 
          "RD-MISS-MSHR-MISS-AND-AVAIL", 
          "1. new m_extra_mf_fields[mf] 2. m_miss_queue.push_back ", mf);
      }

      if (!wa) {
        events.push_back(cache_event(READ_REQUEST_SENT));

        if (DTRACE(CACHE_EVENT)) {
          dumpCacheEvent(time, "RD-MISS-THEN-CHECK-MSHR", 
            "!wa events.push_back(cache_event(READ_REQUEST_SENT))", mf);
        }
      }
      do_miss = true;
    } else if (mshr_hit && !mshr_avail) {
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENTRY_FAIL,
                            mf->get_streamID(), 
                            mshr_merge_entry_fail_driver::MSHR_MERGE_ENTRY_FAIL__RD_MISS);
    } else if (!mshr_hit && !mshr_avail) {
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENTRY_FAIL, 
                            mf->get_streamID(),
                            mshr_entry_fail_driver::MSHR_ENTRY_FAIL__RD_MISS);
    } else {
      assert(0);
    }
  } // !m_mshr_disable
}

/// Sends write request to lower level memory (write or writeback)
void data_cache::send_write_request(
  std::string caller, mem_fetch *mf, cache_event request,
  u64 time, std::list<cache_event> &events) {
  events.push_back(request);
  m_miss_queue.push_back(mf);

  if (DTRACE(L1D_SEND_WR_REQ)) {
    if (m_is_l1d) {
      fprintf(Trace::out, "%llu %s L1D send_write_request "
        "mf:{TPC:%u SM:%u WARP:%u req_uid:%u addr:%#llx} "
        "to m_miss_queue\n",
        time, caller.c_str(),
        mf->get_tpc(), mf->get_sid(), mf->get_wid(), mf->get_request_uid(),
        mf->get_addr());
    }
  }

  if (DTRACE(CACHE_EVENT) || DTRACE(MISS_QUEUE_EVENT)) {
    dumpMissQueue(time, "SEND-WR-REQ-TO-LOWER-LEVEL-MEM", "m_miss_queue.push_back ", mf);
  }

  mf->set_status(m_miss_queue_status, time);
}

void data_cache::update_m_readable(mem_fetch *mf, unsigned cache_index) {
  cache_block_t *block = m_tag_array->get_block(cache_index);
  for (unsigned i = 0; i < SECTOR_CHUNK_SIZE; i++) {
    if (mf->get_access_sector_mask().test(i)) {
      bool all_set = true;      
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        // If any bit in the byte mask (within the sector) is not set,
        // the sector is unreadble
        if (!block->get_dirty_byte_mask().test(k)) {
          all_set = false;
          break;
        }
      }
      if (all_set) {
        block->set_m_readable(true, mf->get_access_sector_mask());
      } 
    }
  }
}

/****** Write-hit functions (Set by config file) ******/

/// Write-back hit: Mark block as modified
cache_request_status data_cache::wr_hit_wb(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           u64 time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(this,addr, block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(
    time, MODIFIED, mf->get_access_sector_mask(), 
    "baseline_cache::wr_hit_wb() " + std::string(m_config.get_cache_name()));
  block->set_byte_mask(mf);
  update_m_readable(mf, cache_index);

  return HIT;
}

/// Write-through hit: Directly send request to lower level memory
cache_request_status data_cache::wr_hit_wt(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           u64 time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0, "data_cache::wr_hit_wt")) {
    if (DTRACE(MISS_QUEUE_FULL_DRIVER)) {      
      if (m_is_l1d) {        
        fprintf(Trace::out, "L1D WR_THROUGH_HIT called MISS_QUEUE_FULL\n");
        fprintf(Trace::out, "[%s][MISS_QUEUE_FULL]++\n",
        mem_access_type_str(mf->get_access_type()));
      }      
    }
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID(), miss_queue_full_driver::WR_THROUGH_HIT);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(this,addr, block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(
    time, MODIFIED, mf->get_access_sector_mask(), 
    "baseline_cache::wr_hit_wt() " + std::string(m_config.get_cache_name()));
  block->set_byte_mask(mf);
  update_m_readable(mf, cache_index);

  // generate a write-through
  send_write_request("data_cache::wr_hit_wt()", 
    mf, cache_event(WRITE_REQUEST_SENT), time, events);

  if (DTRACE(CACHE_EVENT)) {
    dumpCacheEvent(time, 
      "WT-HIT-THEN-SEND-TO-LOWER-MEM", "WRITE_REQUEST_SENT", mf);
  }

  return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate
/// corresponding block
cache_request_status data_cache::wr_hit_we(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           u64 time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0, "data_cache::wr_hit_we")) {
    if (DTRACE(MISS_QUEUE_FULL_DRIVER)) {      
      if (m_is_l1d) {        
        fprintf(Trace::out, "L1D WR_EVICT_HIT called [%s][MISS_QUEUE_FULL]\n",
          mem_access_type_str(mf->get_access_type()));        
        fprintf(Trace::out, "Total_core_cache_fail_stats_breakdown[%s][MISS_QUEUE_FULL]++\n",
        mem_access_type_str(mf->get_access_type()));
      }    
    }    
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID(), miss_queue_full_driver::WR_EVICT_HIT);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // generate a write-through/evict
  cache_block_t *block = m_tag_array->get_block(cache_index);
  send_write_request("data_cache::wr_hit_we()", 
    mf, cache_event(WRITE_REQUEST_SENT), time, events);

  if (DTRACE(CACHE_EVENT)) {
    dumpCacheEvent(time, 
      "data_cache::wr_hit_we", 
      "send to lower mem, and inv current cache", mf);
  }

  // Invalidate block
  block->set_status(
    time, INVALID, mf->get_access_sector_mask(), 
    "baseline_cache::wr_hit_we() " + std::string(m_config.get_cache_name()));

  return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
enum cache_request_status data_cache::wr_hit_global_we_local_wb(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, u64 time,
    std::list<cache_event> &events, enum cache_request_status status) {
  bool evict = (mf->get_access_type() ==
                GLOBAL_ACC_W);  // evict a line that hits on global memory write
  if (evict)
    return wr_hit_we(addr, cache_index, mf, time, events,
                     status);  // Write-evict
  else
    return wr_hit_wb(addr, cache_index, mf, time, events,
                     status);  // Write-back
}

/****** Write-miss functions (Set by config file) ******/

/// Write-allocate miss: Send write request to lower level memory
// and send a read request for the same block
enum cache_request_status data_cache::wr_miss_wa_naive(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, u64 time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  new_addr_type mshr_addr  = m_config.mshr_addr(mf->get_addr());

  // Write allocate, maximum 3 requests (write miss, read request, write back
  // request) Conservatively ensure the worst-case request can be handled this
  // cycle
  bool mshr_hit   = m_mshrs.probe(mshr_addr);
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  if (miss_queue_full(2, "data_cache::wr_miss_wa_naive") ||
      (!(mshr_hit && mshr_avail) &&
       !(!mshr_hit && mshr_avail &&
         (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
    // check what is the exactly the failure reason
    if (miss_queue_full(2, "data_cache::wr_miss_wa_naive")) {
      if (DTRACE(MISS_QUEUE_FULL_DRIVER)) {        
        if (m_is_l1d) {        
          fprintf(Trace::out, "L1D WR_ALLOC_MISS called [%s][MISS_QUEUE_FULL]\n",
            mem_access_type_str(mf->get_access_type()));           
          fprintf(Trace::out, "Total_core_cache_fail_stats_breakdown[%s][MISS_QUEUE_FULL]++\n",
          mem_access_type_str(mf->get_access_type()));
        }           
      }
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID(), miss_queue_full_driver::WR_ALLOC_MISS);
    } else if (mshr_hit && !mshr_avail) {
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENTRY_FAIL,
                             mf->get_streamID(), 
                             mshr_merge_entry_fail_driver::MSHR_MERGE_ENTRY_FAIL__WR_ALLOC_MISS);
    } else if (!mshr_hit && !mshr_avail) {
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENTRY_FAIL,
                             mf->get_streamID(),
                             mshr_entry_fail_driver::MSHR_ENTRY_FAIL__WR_ALLOC_MISS);
    } else {
      assert(0);
    }
    return RESERVATION_FAIL;
  }

  send_write_request("data_cache::wr_miss_wa_naive()", 
    mf, cache_event(WRITE_REQUEST_SENT), time, events);

  if (DTRACE(CACHE_EVENT)) {
    dumpCacheEvent(time, 
      "WT-ALLOC-MISS-THEN-SEND-WR-TO-LOWER-MEM", "WRITE_REQUEST_SENT", mf);
  }  

  const mem_access_t *ma =
      new mem_access_t(m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
                       false,  // Now performing a read
                       mf->get_access_warp_mask(), mf->get_access_byte_mask(),
                       mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

  mem_fetch *n_mf = new mem_fetch(
      *ma, NULL, mf->get_streamID(), mf->get_ctrl_size(), mf->get_wid(),
      mf->get_sid(), mf->get_tpc(), mf->get_mem_config(), m_gpu->get_cycle());

  bool do_miss = false;
  bool wb = false;
  evicted_block_info evicted;

  // Send read request resulting from write miss
  send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                    evicted, events, false, true); // wa
         

  events.push_back(cache_event(WRITE_ALLOCATE_SENT));

  if (DTRACE(CACHE_EVENT)) {
    dumpCacheEvent(time, 
      "WT-ALLOC-MISS-PHASE2-SENT-RD", "WRITE_ALLOCATE_SENT", n_mf);
  }

  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      assert(status == MISS); // SECTOR_MISS and HIT_RESERVED should not send write back
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, mf->get_streamID());
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      send_write_request("data_cache::wr_miss_wa_naive()", 
        wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted), time, events);

      if (DTRACE(CACHE_EVENT)) {
        dumpCacheEvent(time, 
          "WR-ALLOC-MISS-NO-MSHR-PENDING", "WRITE_BACK_REQUEST_SENT", wb);
      }

    }
    return status;
  }

  return RESERVATION_FAIL;
}

enum cache_request_status data_cache::wr_miss_wa_fetch_on_write(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, u64 time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    // if the request writes to the whole cache line/sector, then, write and set
    // cache line Modified. and no need to send read request to memory or
    // reserve mshr

    if (miss_queue_full(0, "data_cache::wr_miss_wa_fetch_on_write")) {
      if (DTRACE(MISS_QUEUE_FULL_DRIVER)) {        
        if (m_is_l1d) {        
          fprintf(Trace::out, "L1D WR_ALLOC_MISS_FETCH_ON_WR_WHOLE_LINE called [%s][MISS_QUEUE_FULL]\n",
            mem_access_type_str(mf->get_access_type()));           
          fprintf(Trace::out, "Total_core_cache_fail_stats_breakdown[%s][MISS_QUEUE_FULL]++\n",
          mem_access_type_str(mf->get_access_type()));
        }        
      }
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID(), 
                             miss_queue_full_driver::WR_ALLOC_MISS_FETCH_ON_WR_WHOLE_LINE);
      return RESERVATION_FAIL;  // cannot handle request this cycle
    }

    bool wb = false;
    evicted_block_info evicted;

    cache_request_status status = 
      m_tag_array->access(this,addr, block_addr, time, cache_index, wb, evicted, mf);
    assert(status != HIT);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(
      time, MODIFIED, mf->get_access_sector_mask(), 
      "baseline_cache::wr_miss_wa_fetch_on_write() " + std::string(m_config.get_cache_name()));
    block->set_byte_mask(mf);
    if (status == HIT_RESERVED)
      block->set_ignore_on_fill(true, mf->get_access_sector_mask());

    if (status != RESERVATION_FAIL) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL, mf->get_streamID());        
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_partition(mf->get_tlx_addr().sub_partition);
        send_write_request("data_cache::wr_miss_wa_fetch_on_write()", 
          wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted), time, events);

        if (DTRACE(CACHE_EVENT)) {
          dumpCacheEvent(time, 
            "TO-HANDLE-THE-EVICTED-LINE", "WRITE_BACK_REQUEST_SENT", wb);
        }
      }
      return MISS;
    }
    return RESERVATION_FAIL;
  } else {
    bool mshr_hit   = m_mshrs.probe(mshr_addr);
    bool mshr_avail = !m_mshrs.full(mshr_addr);
    if (miss_queue_full(1) ||
        (!(mshr_hit && mshr_avail) &&
         !(!mshr_hit && mshr_avail &&
           (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
      // check what is the exactly the failure reason
      if (miss_queue_full(1, "data_cache::wr_miss_wa_fetch_on_write")) {
        if (DTRACE(MISS_QUEUE_FULL_DRIVER)) {          
          if (m_is_l1d) {        
            fprintf(Trace::out, "L1D WR_ALLOC_MISS_FETCH_ON_WR_PARTIAL_LINE called [%s][MISS_QUEUE_FULL]\n",
              mem_access_type_str(mf->get_access_type()));
            fprintf(Trace::out, "Total_core_cache_fail_stats_breakdown[%s][MISS_QUEUE_FULL]++\n",
            mem_access_type_str(mf->get_access_type()));
          } 
        }
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                               mf->get_streamID(), 
                               miss_queue_full_driver::WR_ALLOC_MISS_FETCH_ON_WR_PARTIAL_LINE);
      } else if (mshr_hit && !mshr_avail) {
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENTRY_FAIL,
                               mf->get_streamID(),
                               mshr_merge_entry_fail_driver::MSHR_MERGE_ENTRY_FAIL__WR_ALLOC_MISS_FETCH_ON_WR
                              );
      } else if (!mshr_hit && !mshr_avail) {
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENTRY_FAIL,
                               mf->get_streamID(),
                               mshr_entry_fail_driver::MSHR_ENTRY_FAIL__WR_ALLOC_MISS_FETCH_ON_WR);
      } else {
        assert(0);
      }
      return RESERVATION_FAIL;
    }

    // prevent Write - Read - Write in pending mshr
    // allowing another write will override the value of the first write, and
    // the pending read request will read incorrect result from the second write
    if (m_mshrs.probe(mshr_addr) &&
        m_mshrs.is_read_after_write_pending(mshr_addr) && mf->is_write()) {
      // assert(0);
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_RW_PENDING,
                             mf->get_streamID());
      return RESERVATION_FAIL;
    }

    const mem_access_t *ma = new mem_access_t(
        m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
        false,  // Now performing a read
        mf->get_access_warp_mask(), mf->get_access_byte_mask(),
        mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

    mem_fetch *n_mf = new mem_fetch(
        *ma, NULL, mf->get_streamID(), mf->get_ctrl_size(), mf->get_wid(),
        mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
        m_gpu->get_cycle(), NULL, mf);

    new_addr_type block_addr = m_config.block_addr(addr);
    bool do_miss = false;
    bool wb = false;
    evicted_block_info evicted;
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                      evicted, events, false, true); // wa

    cache_block_t *block = m_tag_array->get_block(cache_index);
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    block->set_byte_mask_on_fill(true);

    events.push_back(cache_event(WRITE_ALLOCATE_SENT));

    if (DTRACE(CACHE_EVENT)) {
      dumpCacheEvent(time, 
        "PREVENT-WR-RD-WR-IN-PENDING-MSHR", "WRITE_ALLOCATE_SENT", n_mf);
    }    

    if (do_miss) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL, mf->get_streamID());
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_partition(mf->get_tlx_addr().sub_partition);
        send_write_request("data_cache::wr_miss_wa_fetch_on_write()", 
          wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted), time, events);

        if (DTRACE(CACHE_EVENT)) {
          dumpCacheEvent(time, "xxx", "WRITE_BACK_REQUEST_SENT", wb);
        }                               
      }
      return MISS;
    }
    return RESERVATION_FAIL;
  }
}

enum cache_request_status data_cache::wr_miss_wa_lazy_fetch_on_read(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, u64 time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);

  // if the request writes to the whole cache line/sector, then, write and set
  // cache line Modified. and no need to send read request to memory or reserve
  // mshr

  if (miss_queue_full(0, "data_cache::wr_miss_wa_lazy_fetch_on_read")) {
    if (DTRACE(MISS_QUEUE_FULL_DRIVER)) {      
      if (m_is_l1d) {        
        fprintf(Trace::out, "L1D WR_ALLOC_MISS_LAZY_FETCH_ON_RD called [%s][MISS_QUEUE_FULL]\n",
          mem_access_type_str(mf->get_access_type()));        
        fprintf(Trace::out, "Total_core_cache_fail_stats_breakdown[%s][MISS_QUEUE_FULL]++\n",
        mem_access_type_str(mf->get_access_type()));
      }  
    }
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID(), 
                           miss_queue_full_driver::WR_ALLOC_MISS_LAZY_FETCH_ON_RD);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  if (m_config.m_write_policy == WRITE_THROUGH) {
    send_write_request("data_cache::wr_miss_wa_lazy_fetch_on_read()", 
      mf, cache_event(WRITE_REQUEST_SENT), time, events);
    
    if (DTRACE(CACHE_EVENT)) {
      dumpCacheEvent(time, 
        "data_cache::wr_miss_wa_lazy_fetch_on_read", 
        "WRITE_REQUEST_SENT", mf);
    }    
  }

  bool wb = false;
  evicted_block_info evicted;

  cache_request_status req_status = status;
  if (status == VC_HIT) {
    // Pass bypass_2nd_probe = true for the last in-arg
    req_status = m_tag_array->access(this, addr, block_addr, time, cache_index, wb, evicted, mf, true);
  } else {
    req_status = m_tag_array->access(this, addr, block_addr, time, cache_index, wb, evicted, mf);
  }
  // default logic (below)
  // cache_request_status req_status = 
  //   m_tag_array->access(this, addr, block_addr, time, cache_index, wb, evicted, mf);

  assert(req_status != HIT);
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  cache_block_state prev_blk_state = block->get_status(mf->get_access_sector_mask()); // for tracing

  unsigned sidx = block->set_status(
    time, MODIFIED, mf->get_access_sector_mask(), 
    "baseline_cache::wr_miss_wa_lazy_fetch_on_read() " + std::string(m_config.get_cache_name()));

  block->set_byte_mask(mf);

  if (req_status == HIT_RESERVED) {
    block->set_ignore_on_fill(true, mf->get_access_sector_mask());
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    block->set_byte_mask_on_fill(true);
  }

  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    block->set_m_readable(true, mf->get_access_sector_mask());
  } else {
    block->set_m_readable(false, mf->get_access_sector_mask());
    if (req_status == HIT_RESERVED) {
      block->set_readable_on_fill(true, mf->get_access_sector_mask());
    }      
  }
  update_m_readable(mf, cache_index);

  [[maybe_unused]] cache_block_state sector_status   = block->get_status(mf->get_access_sector_mask());
  [[maybe_unused]] mem_access_byte_mask_t dirty_mask = block->get_dirty_byte_mask();
  std::pair<uint64_t,uint64_t> byte_mask_hi_lo = to_u64_pair(dirty_mask);
  if (DTRACE(CACHELINE_STATUS)) {
    fprintf(Trace::out,
            "%llu:%s%s %s %s addr:%#llx m_sector[sidx:%u] (%s->%s) "
            "dirty_byte_mask=0x%016lx%016lx is_readable=%u\n",
            (u64)(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle),
            m_gpu->gpgpu_ctx->func_sim->ptx_get_insn_str(mf->get_inst().pc).c_str(),
            m_is_l1d ? "L1D" : m_is_l2 ? "L2C" : "xx$",
            mf_request_type_str(mf->get_type()),
            cache_request_status_str(req_status),
            (u64)block_addr, sidx,
            cache_block_state_str(prev_blk_state),
            cache_block_state_str(sector_status),
            byte_mask_hi_lo.first, byte_mask_hi_lo.second,
            block->is_readable(mf->get_access_sector_mask()));
  }

  if (req_status != RESERVATION_FAIL) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, mf->get_streamID());
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      send_write_request("data_cache::wr_miss_wa_lazy_fetch_on_read()", 
        wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted), time, events);

      if (DTRACE(CACHE_EVENT)) {
        dumpCacheEvent(time, 
          "data_cache::wr_miss_wa_lazy_fetch_on_read", 
          "WRITE_BACK_REQUEST_SENT", wb);
      }                            
    }

    return MISS;
  }

  if (DTRACE(CACHELINE_STATUS)) {
    fprintf(Trace::out,
      "%llu: RESERVATION_FAIL on block_addr=%#llx sector_mask=%s "
      "byte_mask=%s\n",
      (u64)(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle),
      (u64)block_addr,
      mf->get_access_sector_mask().to_string().c_str(),
      mf->get_access_byte_mask().to_string().c_str());
  }
  return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
enum cache_request_status data_cache::wr_miss_no_wa(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, u64 time,
    std::list<cache_event> &events, enum cache_request_status status) {
  if (miss_queue_full(0, "data_cache::wr_miss_no_wa")) {
    if (DTRACE(MISS_QUEUE_FULL_DRIVER)) {      
      if (m_is_l1d) {        
        fprintf(Trace::out, "L1D WR_MISS_NO_WR_ALLOC called [%s][MISS_QUEUE_FULL]\n",
          mem_access_type_str(mf->get_access_type())); 
        fprintf(Trace::out, "Total_core_cache_fail_stats_breakdown[%s][MISS_QUEUE_FULL]++\n",
        mem_access_type_str(mf->get_access_type()));
      } 
    }
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID(), 
                           miss_queue_full_driver::WR_MISS_NO_WR_ALLOC);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // on miss, generate write through (no write buffering -- too many threads for
  // that)
  send_write_request("data_cache::wr_miss_no_wa()", 
    mf, cache_event(WRITE_REQUEST_SENT), time, events);

  if (DTRACE(CACHE_EVENT)) {
    dumpCacheEvent(time, 
      "data_cache::wr_miss_no_wa", "WRITE_REQUEST_SENT", mf);
  }

  return MISS;
}

/****** Read hit functions (Set by config file) ******/

/// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
enum cache_request_status data_cache::rd_hit_base(
    new_addr_type addr /* raw_addr */, 
    unsigned cache_index, mem_fetch *mf, u64 time, 
    std::list<cache_event> &events, enum cache_request_status status) {

  assert(addr == mf->get_addr());
  new_addr_type block_addr  = m_config.block_addr(addr);
  [[maybe_unused]] new_addr_type sector_addr = m_config.mshr_addr(addr);

  if (m_config.m_bypass_enable == 'T') {
    BYPASS_KEY byp_key(mf->get_streamID(), m_gpu->m_kernel_id, block_addr);
    if (m_is_l1d &&
      m_config.m_bypass_mode == 1 && 
      m_tag_array->hit_l1d_bypassed_item(byp_key, mf)) {
      assert(0);
    }
  }
  
  if (status == VC_HIT) {
    m_tag_array->access(this, addr, block_addr, time, cache_index, mf, true /* bypass 2nd tag_array->probe*/);
    // evicted_block_info evicted;
    // bool wb = false;
    // m_tag_array->access(this, addr, block_addr, time, cache_index, wb, evicted, mf);
  } else {    
    m_tag_array->access(this, addr, block_addr, time, cache_index, mf); // default
  }

  // Atomics treated as global read/write requests - Perform read, mark line as
  // MODIFIED
  if (mf->isatomic()) {
    assert(mf->get_access_type() == GLOBAL_ACC_R);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(
      time, MODIFIED, mf->get_access_sector_mask(), 
      "data_cache::rd_hit_base" + std::string(m_config.get_cache_name()));
    block->set_byte_mask(mf);
  }
  return HIT;
}

/****** Read miss functions (Set by config file) ******/

/// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
enum cache_request_status data_cache::rd_miss_base(
    new_addr_type addr /* raw_addr */, unsigned cache_index, mem_fetch *mf, 
    u64 time,
    std::list<cache_event> &events, enum cache_request_status status) {

  // "addr" is just used for generating "block_addr" in this function

  // 4/10 Following assertion would fail when 
  // '-DEXCLUDE_MEMCPY_CYCLES_FROM_CACHE_TIMING' were not in CXXFLAGS
  // assert(time == m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);

  new_addr_type block_addr = m_config.block_addr(addr);  
  
  if (status == cache_request_status::MISS || \
      status == cache_request_status::SECTOR_MISS) {

    mf->set_status(IN_PARTITION_L2, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    m_l1d_rd_miss_addresses.push_back(addr);
  }

  if (miss_queue_full(1, "data_cache::rd_miss_base")) {
    // cannot handle request this cycle
    // (might need to generate two requests)
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID(), miss_queue_full_driver::RD_MISS);
    return RESERVATION_FAIL;
  }

  if (DTRACE(CACHE_EVENT)) {
    dumpCacheEvent(time, "data_cache::rd_miss_base", 
      "send_read_request to next-level-$", mf);
  }

  bool do_miss = false;
  bool wb = false;
  evicted_block_info evicted;
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb,
                    evicted, events, false, false); // !wa

  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, mf->get_streamID());
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      send_write_request("data_cache::rd_miss_base()", 
        wb, WRITE_BACK_REQUEST_SENT, time, events);
    }
    return status; // 4/2
    // return MISS;
  }
  if (status != RESERVATION_FAIL) {
    if (DTRACE(DEBUG_DUP_TAG_PROBE)) {
      fprintf(Trace::out, "status is %s != RESERVATION_FAIL\n", cache_request_status_str(status));    
    }    
    assert(0);
  }
  
  return RESERVATION_FAIL;
}

/// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
enum cache_request_status read_only_cache::access(
  new_addr_type addr, mem_fetch *mf, u64 time,
  std::list<cache_event> &events) {
  assert(mf->get_data_size() <= m_config.get_atom_sz());
  assert(m_config.m_write_policy == READ_ONLY);
  assert(!mf->get_is_write());
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;

  bool inter_warp_has_interference = false;
  WARP_INTERFERE_RECORD inter_warp_interfere_record((unsigned )- 1, (unsigned) - 1);

  enum cache_request_status status = m_tag_array->probe(
    "read_only_cache::access", this, addr, block_addr, cache_index, mf, mf->is_write(), 
    time, inter_warp_has_interference, inter_warp_interfere_record);
  
  enum cache_request_status cache_status = RESERVATION_FAIL;

  if (status == HIT) {
    cache_status = m_tag_array->access(this,addr, block_addr, time, cache_index, mf); // update LRU state
  } else if (status != RESERVATION_FAIL) {
    if (!miss_queue_full(0, "read_only_cache::access")) {
      bool do_miss = false;
      send_read_request(addr, block_addr, cache_index, mf, time, do_miss,
                        events, true, false); // !wa
      if (do_miss) {
        cache_status = MISS;
      } else {
        cache_status = RESERVATION_FAIL;
      }        
    } else {
      cache_status = RESERVATION_FAIL;
      if (DTRACE(MISS_QUEUE_FULL_DRIVER)) {        
        if (m_is_l1d) {        
          fprintf(Trace::out, "L1D RD_ONLY_MISS called [%s][MISS_QUEUE_FULL]\n",
            mem_access_type_str(mf->get_access_type())); 
          fprintf(Trace::out, "Total_core_cache_fail_stats_breakdown[%s][MISS_QUEUE_FULL]++\n",
          mem_access_type_str(mf->get_access_type()));
        }      
      }
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID(), miss_queue_full_driver::RD_ONLY_MISS);
    }
  } else {
    m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                           mf->get_streamID(), line_alloc_fail_driver::LINE_ALLOC_FAIL__RD_ONLY_MISS);
  }

  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status),
                    mf->get_streamID());
  m_stats.update_evict_stats(mf->get_streamID(), mf->get_victim_avg_evict_interval());

  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status),
                       mf->get_streamID());
  return cache_status;
}

//! A general function that takes the result of a tag_array probe
//  and performs the correspding functions based on the cache configuration
//  The access fucntion calls this function
enum cache_request_status data_cache::process_tag_probe(
    bool wr, enum cache_request_status probe_status, 
    new_addr_type addr /* raw_addr */,
    unsigned cache_index, mem_fetch *mf, u64 time,
    std::list<cache_event> &events) {
  
  // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
  // data_cache constructor to reflect the corresponding cache configuration
  // options. Function pointers were used to avoid many long conditional
  // branches resulting from many cache configuration options.
  cache_request_status access_status = probe_status;
  [[maybe_unused]] new_addr_type sector_addr = m_config.mshr_addr(addr);
  if (wr) { // Write
    if (probe_status == HIT) {
      access_status = (this->*m_wr_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if ((probe_status != RESERVATION_FAIL) ||
               (probe_status == RESERVATION_FAIL &&
                m_config.m_write_alloc_policy == NO_WRITE_ALLOCATE)) {

      if (DTRACE(CACHE_EVENT)) {
        dumpCacheEvent(time, 
          "data_cache::process_tag_probe", "m_wr_miss", mf);
      }

      if (m_is_l1d) {
        m_stats.inc_l1d_wr_misses(mf->get_streamID(), mf->get_sid());
      }      

      access_status = (this->*m_wr_miss)(addr, cache_index, mf, time, events, probe_status);
      if (access_status == cache_request_status::MISS) {
        mf->set_wr_miss_serve_begin_time(time);
      }      
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                             mf->get_streamID(), line_alloc_fail_driver::LINE_ALLOC_FAIL__WR_PROBE_MISS);
    }
  } else {  // Read
    BYPASS_KEY byp_key(mf->get_streamID(), m_gpu->m_kernel_id, m_config.block_addr(addr));
    LOCALITY_KEY loc_key(mf->get_streamID(), m_gpu->m_kernel_id);
    if (probe_status == HIT || probe_status == VC_HIT) {
      access_status = (this->*m_rd_hit)(addr, cache_index, mf, time, events, probe_status);  
    } else if (probe_status != RESERVATION_FAIL) {
      if (DTRACE(CACHE_EVENT)) {
        dumpCacheEvent(time, "data_cache::process_tag_probe", "m_rd_miss", mf);
      }

      if (!m_tag_array->hit_l1d_bypassed_item(byp_key, mf)) {
        m_stats.inc_l1d_rd_misses(mf->get_streamID(), m_gpu->m_kernel_id);
        m_stats.update_l1d_max_evictions(loc_key, m_tag_array->get_l1d_evictions(byp_key));
        m_stats.update_l1d_avg_evictions(loc_key, m_tag_array->get_l1d_evictions(byp_key));
      }

      access_status = (this->*m_rd_miss)(addr, cache_index, mf, time, events, probe_status);
      if (access_status == cache_request_status::MISS) {
        mf->set_rd_miss_serve_begin_time(time);
      }
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                             mf->get_streamID(), line_alloc_fail_driver::LINE_ALLOC_FAIL__RD_PROBE_MISS);
    }
  }

  m_bandwidth_management.use_data_port(mf, access_status, events, wr);

  return access_status;
}

// Both the L1 and L2 currently use the same access function.
// Differentiation between the two caches is done through configuration
// of caching policies.
// Both the L1 and L2 override this function to provide a means of
// performing actions specific to each cache when such actions are implemented.
enum cache_request_status data_cache::access(new_addr_type addr, mem_fetch *mf,
                                             u64 time,
                                             std::list<cache_event> &events) {
  
  assert(mf->get_data_size() <= m_config.get_atom_sz());
  bool wr = mf->get_is_write();
  new_addr_type block_addr  = m_config.block_addr(addr);
  [[maybe_unused]] new_addr_type sector_addr = m_config.mshr_addr(addr);
  unsigned cache_index = (unsigned) - 1;

  bool inter_warp_has_interference = false;
  WARP_INTERFERE_RECORD inter_warp_interfere_record((unsigned )- 1, (unsigned) - 1);

  if (DTRACE(LOAD_PIPE)) {    
    if (mf) {
      assert(get_inst_info().find("mem_req_uid") != std::string::npos);
      fprintf(Trace::out, "%llu data_cache::access addr:%#llx "
        "(block_addr:%#llx) for inst %s\n",
        time, addr, block_addr, get_inst_info().c_str());
    }
  }

  // Pick one victim to update "cache_index" that was init as "-1" above.
  // enum cache_request_status probe_status = m_tag_array->probe(
  //   "data_cache::access",
  //   block_addr, cache_index, mf, mf->is_write(), time,
  //   inter_warp_has_interference, inter_warp_interfere_record,
  //   true /* probe_mode */); // default logic

  if (m_is_l1d && mf) {
    if (!mf->is_write() && !mf->isatomic()) {
      m_stats.inc_l1d_reads(mf->get_streamID(), m_gpu->m_kernel_id);

      if (DTRACE(INC_L1D_READS)) {
        fprintf(Trace::out, "%llu inc_l1d_reads[streamID:%llu][kernel:%u] = %u\n",
          time, mf->get_streamID(), m_gpu->m_kernel_id,
          m_stats.get_l1d_reads(mf->get_streamID(), m_gpu->m_kernel_id));
      }    
    } else if (mf->is_write()) {
      m_stats.inc_l1d_writes(mf->get_streamID(), m_gpu->m_kernel_id);
    } 
    m_stats.inc_l1d_accesses(mf->get_streamID(), m_gpu->m_kernel_id);
  }

  enum cache_request_status probe_status = cache_request_status::MISS;
  if (m_config.m_bypass_enable == 'T') {
    BYPASS_KEY byp_key(mf->get_streamID(), m_gpu->m_kernel_id, block_addr);
    if (m_config.m_bypass_mode == 1 && m_tag_array->hit_l1d_bypassed_item(byp_key, mf)) {
      assert(!mf->is_write());
      assert(!mf->isatomic());
      if (DTRACE(TRACE_BYPASSED_L1D_PKT) || DTRACE(HIT_L1D_BYPASSED_ITEM)) {
        dumpCacheEvent(time, "data_cache::access", 
          "HIT_L1D_BYPASSED_ITEM Bypassed m_tag_array->probe", mf);
      }
    } else {
      probe_status = m_tag_array->probe(
          "data_cache::access", this,
          addr, block_addr, cache_index, mf, mf->is_write(), time,
          inter_warp_has_interference, inter_warp_interfere_record,
          true /* probe_mode */);
    }
  } else {
    probe_status = m_tag_array->probe(
      "data_cache::access", this,
      addr, block_addr, cache_index, mf, mf->is_write(), time,
      inter_warp_has_interference, inter_warp_interfere_record,
      true /* probe_mode */); // default logic     
  }

  unsigned interfered  = inter_warp_interfere_record.last_warp_id;
  unsigned interfering = inter_warp_interfere_record.curr_warp_id;
  const shader_core_config *shader_cfg = m_gpu->getShaderCoreConfig();
  const unsigned max_warps_per_shader = shader_cfg->max_warps_per_shader;
  const unsigned num_shader = shader_cfg->num_shader();
  assert(mf);
  const unsigned sid = mf->get_sid();
  if (inter_warp_has_interference) {
    if (m_is_l2) {
      assert(0);
    }
    assert(sid < num_shader);
    assert(interfered < max_warps_per_shader);
    assert(interfering < max_warps_per_shader);
    assert(interfered != interfering);
    m_gpu->get_shader_stats()->inter_warp_interfere[sid][interfered][interfering]++;

    if (DTRACE(INC_WARP_INTERFERE)) {
      const char* inst_name = 
        m_gpu->gpgpu_ctx->func_sim->ptx_get_insn_str(mf->get_inst().pc).c_str();

      fprintf(Trace::out, "%llu Increased "
        "inter_warp_interfere[sid:%u][interfered:%u][interfering:%u] = %u inst = %s\n",
        time, sid, interfered, interfering,
        m_gpu->get_shader_stats()->inter_warp_interfere[sid][interfered][interfering],
        inst_name
      );
    }    
  }

  // access_status is not always equal to probe_status
  // Ex.1: probe_status (SECTOR_MISS) -> access_status (MISS)
  // Ex.2: probe_status (HIT_RESERVED) -> access_status (RESERVATION_FAIL) when miss_queue_full(1) during a 2nd tag_array->probe
  enum cache_request_status access_status = process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);

  if (DTRACE(LOAD_PIPE)) {
    if (access_status != HIT) {
      assert(get_inst_info().find("mem_req_uid") != std::string::npos);
      fprintf(Trace::out, "%llu access_status:%s for accessing addr:%#llx "
        "(block_addr:%#llx) for inst %s\n",
        time, cache_request_status_str(access_status), addr, block_addr, 
        get_inst_info().c_str());
    }
  }

  enum cache_request_status access_stats_bak = access_status;

  if (DTRACE(CACHE_EVENT)) {
    dumpCacheEvent(time, "data_cache::access", "process_tag_probe", mf);
  }

  if (m_config.m_bypass_enable == 'T') {
    if (mf->get_l1d_rd_byp_change() == 2) {
      access_status = cache_request_status::BYPASS_ACTIVATED; // A hook for inc_stats
      if (DTRACE(CHECK_BYPASS_RET_STATUS)) {
        fprintf(Trace::out, "%llu CHECK_BYPASS_RET_STATUS: BYPASS_ACTIVATED\n", time);
      }
    } else if (mf->get_l1d_rd_byp_change() == 1) {
      access_status = cache_request_status::BYPASS_DEACTIVATED; // A hook for inc_stats
      if (DTRACE(CHECK_BYPASS_RET_STATUS)) {
        fprintf(Trace::out, "%llu CHECK_BYPASS_RET_STATUS: BYPASS_DEACTIVATED\n", time);
      }
    }
  }

  const LOCALITY_KEY loc_key(mf->get_streamID(), m_gpu->m_kernel_id);
  m_stats.update_n_l1d_fill_to_evict(loc_key, 
    m_tag_array->m_l1d_fill_to_evict_lines[loc_key].size());
  
  [[maybe_unused]] bool en_inc_byp_act   = m_config.m_bypass_enable == 'T' && mf->get_l1d_rd_byp_change() == 2;
  [[maybe_unused]] bool en_inc_byp_deact = m_config.m_bypass_enable == 'T' && mf->get_l1d_rd_byp_change() == 1;    
  // m_stats.update_l1d_rd_byp_act(
  //   en_inc_byp_act, loc_key, m_tag_array->m_l1d_rd_byp_activated_times[loc_key], mf->get_streamID());
  // m_stats.update_l1d_rd_byp_deact(
  //   en_inc_byp_deact, loc_key, m_tag_array->m_l1d_rd_byp_deactivated_times[loc_key], mf->get_streamID());

  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(probe_status, access_status),
                    mf->get_streamID());

  m_stats.update_evict_stats(mf->get_streamID(), mf->get_victim_avg_evict_interval());

  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(probe_status, access_status),
                       mf->get_streamID());

  access_status = access_stats_bak; // Restore access_status for ack with upper-level

  if (m_is_l1d) {
    uint64_t lat_from_sched_to_access = time - m_gpu->sched_cycle[mf->get_pc()];
    m_gpu->tot_l1d_accesses++;
    m_gpu->tot_l1d_lat_from_sched_to_access += lat_from_sched_to_access;
    if (wr) {
      m_gpu->tot_l1d_writes++;
      m_gpu->tot_l1d_wr_lat_from_sched += lat_from_sched_to_access;
    } else {
      m_gpu->tot_l1d_reads++;
      m_gpu->tot_l1d_rd_lat_from_sched += lat_from_sched_to_access;
    }
  }

  return access_status;
}

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at the
/// granularity of individual blocks (Set by GPGPU-Sim configuration file)
/// (the policy used in fermi according to the CUDA manual)
enum cache_request_status l1_cache::access(new_addr_type addr, mem_fetch *mf,
                                           u64 time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status l2_cache::access(
  new_addr_type addr, mem_fetch *mf, u64 time,
  std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

/// Access function for tex_cache
/// return values: RESERVATION_FAIL if request could not be accepted
/// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
/// since unlike a normal CPU cache, a "HIT" in texture cache does not
/// mean the data is ready (still need to get through fragment fifo)
enum cache_request_status tex_cache::access(new_addr_type addr, mem_fetch *mf,
                                            u64 time,
                                            std::list<cache_event> &events) {
  if (m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full())
    return RESERVATION_FAIL;

  assert(mf->get_data_size() <= m_config.get_line_sz());

  // at this point, we will accept the request : access tags and immediately
  // allocate line
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;
  enum cache_request_status status = m_tags.access(nullptr, addr, block_addr, time, cache_index, mf);
  enum cache_request_status cache_status = RESERVATION_FAIL;
  assert(status != RESERVATION_FAIL);
  assert(status != HIT_RESERVED);  // as far as tags are concerned: HIT or MISS
  m_fragment_fifo.push(fragment_entry(mf, cache_index, status == MISS, mf->get_data_size()));
  if (status == MISS) {
    // we need to send a memory request...
    unsigned rob_index = m_rob.push(rob_entry(cache_index, mf, block_addr));
    m_extra_mf_fields[mf] = extra_mf_fields(rob_index, m_config);
    mf->set_data_size(m_config.get_line_sz());
    m_tags.fill(nullptr, cache_index, time, mf);  // mark block as valid
    m_request_fifo.push(mf);
    mf->set_status(m_request_queue_status, time);
    events.push_back(cache_event(READ_REQUEST_SENT));
    cache_status = status;
  } else {
    // the value *will* *be* in the cache already
    cache_status = HIT_RESERVED;
  }
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status),
                    mf->get_streamID());
  m_stats.update_evict_stats(mf->get_streamID(), mf->get_victim_avg_evict_interval());

  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status),
                       mf->get_streamID());
  return cache_status;
}

void tex_cache::cycle() {
  // send next request to lower level of memory
  // TODO: Use different full() for sst_mem_interface?
  if (!m_request_fifo.empty()) {
    mem_fetch *mf = m_request_fifo.peek();
    if (!m_memport->full(mf->get_ctrl_size(), false)) {
      m_request_fifo.pop();
      m_memport->push(mf);
    }
  }
  // read ready lines from cache
  if (!m_fragment_fifo.empty() && !m_result_fifo.full()) {
    const fragment_entry &e = m_fragment_fifo.peek();
    if (e.m_miss) {
      // check head of reorder buffer to see if data is back from memory
      unsigned rob_index = m_rob.next_pop_index();
      const rob_entry &r = m_rob.peek(rob_index);
      assert(r.m_request == e.m_request);
      // assert( r.m_block_addr == m_config.block_addr(e.m_request->get_addr())
      // );
      if (r.m_ready) {
        assert(r.m_index == e.m_cache_index);
        m_cache[r.m_index].m_valid = true;
        m_cache[r.m_index].m_block_addr = r.m_block_addr;
        m_result_fifo.push(e.m_request);
        m_rob.pop();
        m_fragment_fifo.pop();
      }
    } else {
      // hit:
      assert(m_cache[e.m_cache_index].m_valid);
      assert(m_cache[e.m_cache_index].m_block_addr ==
             m_config.block_addr(e.m_request->get_addr()));
      m_result_fifo.push(e.m_request);
      m_fragment_fifo.pop();
    }
  }
}

/// Place returning cache block into reorder buffer
void tex_cache::fill(mem_fetch *mf, unsigned time) {
  if (m_config.m_mshr_type == SECTOR_TEX_FIFO) {
    assert(mf->get_original_mf());
    extra_mf_fields_lookup::iterator e =
        m_extra_mf_fields.find(mf->get_original_mf());
    assert(e != m_extra_mf_fields.end());
    e->second.pending_read--;

    if (e->second.pending_read > 0) {
      // wait for the other requests to come back
      delete mf;
      return;
    } else {
      mem_fetch *temp = mf;
      mf = mf->get_original_mf();
      delete temp;
    }
  }

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  assert(e != m_extra_mf_fields.end());
  assert(e->second.m_valid);
  assert(!m_rob.empty());
  mf->set_status(m_rob_status, time);

  unsigned rob_index = e->second.m_rob_index;
  rob_entry &r = m_rob.peek(rob_index);
  assert(!r.m_ready);
  r.m_ready = true;
  r.m_time = time;
  assert(r.m_block_addr == m_config.block_addr(mf->get_addr()));
}

void tex_cache::display_state(FILE *fp) const {
  fprintf(fp, "%s (texture cache) state:\n", m_name.c_str());
  fprintf(fp, "fragment fifo entries  = %u / %u\n", m_fragment_fifo.size(),
          m_fragment_fifo.capacity());
  fprintf(fp, "reorder buffer entries = %u / %u\n", m_rob.size(),
          m_rob.capacity());
  fprintf(fp, "request fifo entries   = %u / %u\n", m_request_fifo.size(),
          m_request_fifo.capacity());
  if (!m_rob.empty()) fprintf(fp, "reorder buffer contents:\n");
  for (int n = m_rob.size() - 1; n >= 0; n--) {
    unsigned index = (m_rob.next_pop_index() + n) % m_rob.capacity();
    const rob_entry &r = m_rob.peek(index);
    fprintf(fp, "tex rob[%3d] : %s ", index,
            (r.m_ready ? "ready  " : "pending"));
    if (r.m_ready)
      fprintf(fp, "@%6u", r.m_time);
    else
      fprintf(fp, "       ");
    fprintf(fp, "[idx=%4u]", r.m_index);
    r.m_request->print(fp, false);
  }
  if (!m_fragment_fifo.empty()) {
    fprintf(fp, "fragment fifo (oldest) :");
    fragment_entry &f = m_fragment_fifo.peek();
    fprintf(fp, "%s:          ", f.m_miss ? "miss" : "hit ");
    f.m_request->print(fp, false);
  }
}
/******************************************************************************************************************************************/