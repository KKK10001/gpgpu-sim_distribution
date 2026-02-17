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
      "HIT_RESERVED", 
      "MISS",
      "RESERVATION_FAIL",
      "SECTOR_MISS",
      "MSHR_HIT"
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

unsigned cache_config::recalc_orig_addr(new_addr_type tag, unsigned set_index) const {
  new_addr_type orig_addr = 
    (tag << (m_line_sz_log2 + m_nset_log2)) | (set_index << m_line_sz_log2);
  return orig_addr;
}

unsigned cache_config::set_index(new_addr_type addr) const {
  return cache_config::hash_function(addr, m_nset, m_line_sz_log2, m_nset_log2,
                                     m_set_index_function);
}

unsigned cache_config::hash_function(new_addr_type addr, unsigned m_nset,
                                     unsigned m_line_sz_log2,
                                     unsigned m_nset_log2,
                                     unsigned m_index_function) const {
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

unsigned l2_cache_config::set_index(new_addr_type addr) const {
  new_addr_type part_addr = addr;

  if (m_address_mapping) {
    // Calculate set index without memory partition bits to reduce set camping
    part_addr = m_address_mapping->partition_address(addr);
  }

  return cache_config::set_index(part_addr);
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

tag_array::tag_array(cache_config &config, int core_id, int type_id)
    : m_config(config) {
      
  // fprintf("m_config.m_shader_cores = %u\n", m_config.m_shader_cores);
  // m_unique_lines.resize(config.m_L1D_config.m_shader_cores);
  m_unique_lines.resize(4); // replace m_shader_core later than
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
  gpgpu_sim *gpu,
  new_addr_type addr, unsigned &idx,
  mem_fetch *mf, bool is_write,
  unsigned long long time,
  bool& got_warp_interfere_info, 
  WARP_INTERFERE_RECORD& warp_interfere_record,
  bool probe_mode) {

  mem_access_sector_mask_t mask = mf->get_access_sector_mask();
  std::string final_caller = caller + "-> tag_array::probe";

  return probe(
    final_caller.c_str(), gpu, addr, idx, mask, is_write, time, 
    probe_mode, got_warp_interfere_info, warp_interfere_record, mf);
}

void tag_array::gather_rep_candidates(
  cache_block_t* line, const unsigned& index,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates) {

  LINE_RECENCY recency(
    line->get_last_access_time(), 
    line->get_last_fill_time(),
    line->get_rrpv(),
    line->get_max_rrpv(),
    line->get_recorded_times_in_mshr(),
    line->get_total_hits(),
    line->get_total_evictions(), 
    line->get_total_accesses(),
    line->get_evict_interval(), 
    line->get_avg_evict_interval(),
    line->get_last_warp_id(),
    line->get_last_core_id()
  );
  if (!line->was_recorded_in_mshr()) {
    hybrid_rep_candidates_no_record_in_mshr.push_back(std::pair<unsigned, LINE_RECENCY>(index, recency));
  } else {
    hybrid_rep_candidates_recorded_in_mshr.push_back(std::pair<unsigned, LINE_RECENCY>(index, recency));
  }
  hybrid_rep_candidates.push_back(std::pair<unsigned, LINE_RECENCY>(index, recency));
}

void tag_array::lru_pick(
  cache_block_t* line, unsigned long long& valid_timestamp, 
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
  cache_block_t* line, unsigned long long& valid_timestamp, 
  unsigned& valid_line, const unsigned& index) {

  if (line->get_last_fill_time() < valid_timestamp) {
    valid_timestamp = line->get_last_fill_time();
    valid_line      = index;
  }
}

void tag_array::warp_interfere_awared_pick(
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  unsigned& valid_line /* victim index */,
  unsigned& warp_id, unsigned& core_id,
  mem_fetch *mf
) {
  std::vector<std::pair<unsigned, LINE_RECENCY>> hybrid_rep_candidates_in_use;
  if (m_config.m_mshr_corr_repl == 'T') {
    if (hybrid_rep_candidates_no_record_in_mshr.size()) {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_no_record_in_mshr;
    } else {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_recorded_in_mshr;
    }
  } else {
    hybrid_rep_candidates_in_use = hybrid_rep_candidates;
  }
  assert(hybrid_rep_candidates_in_use.size());

  std::vector<std::pair<unsigned, LINE_RECENCY>> target_warp_candidates;
  for (unsigned i = 0; i < hybrid_rep_candidates_in_use.size(); i++)
  {
    if (hybrid_rep_candidates_in_use[i].second.warp_id == mf->get_wid()) {
      target_warp_candidates.push_back(hybrid_rep_candidates_in_use[i]);
    }
  }

  if (!target_warp_candidates.empty()) {
    std::sort(target_warp_candidates.begin(), target_warp_candidates.end(), cmpForSmallerTimestamp);
    valid_line           = target_warp_candidates[0].first;
    warp_id              = target_warp_candidates[0].second.warp_id;
    core_id              = target_warp_candidates[0].second.core_id;
  } else {
    std::sort(hybrid_rep_candidates_in_use.begin(), hybrid_rep_candidates_in_use.end(), cmpForSmallerTimestamp);
    valid_line           = hybrid_rep_candidates_in_use[0].first;
    warp_id              = hybrid_rep_candidates_in_use[0].second.warp_id;
    core_id              = hybrid_rep_candidates_in_use[0].second.core_id;
  }
}

void tag_array::pick_with_lru(
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  unsigned& valid_line,
  unsigned& warp_id, unsigned& core_id,
  unsigned long long& smallest_access_time,
  unsigned& lru_picked_total_hits,
  unsigned long long& lru_picked_avg_evict_interval) {

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
    lru_picked_avg_evict_interval = hybrid_rep_candidates_in_use[0].second.avg_evict_interval;
    warp_id                       = hybrid_rep_candidates_in_use[0].second.warp_id;
    core_id                       = hybrid_rep_candidates_in_use[0].second.core_id;

  } else {
    assert(hybrid_rep_candidates.size());
    std::sort(hybrid_rep_candidates.begin(), hybrid_rep_candidates.end(), cmpForSmallerTimestamp);
    valid_line                    = hybrid_rep_candidates[0].first; // update valid_line 
    smallest_access_time          = hybrid_rep_candidates[0].second.last_access_time;
    lru_picked_total_hits         = hybrid_rep_candidates[0].second.total_hits;
    lru_picked_avg_evict_interval = hybrid_rep_candidates[0].second.avg_evict_interval;
    warp_id                       = hybrid_rep_candidates[0].second.warp_id;
    core_id                       = hybrid_rep_candidates[0].second.core_id;    

    if (DTRACE(CHECK_LRU_ARRAY)) {
      if (valid_line != hybrid_rep_candidates[0].first) {
        fprintf(Trace::out, "valid_line:%u != hybrid_rep_candidates[0].first:%u\n",
          valid_line, hybrid_rep_candidates[0].first);
        for (unsigned i = 0; i < hybrid_rep_candidates.size(); i++)
        {
          fprintf(Trace::out, "hybrid_rep_candidates[%u] = {idx:%u timestamp:%llu}\n", 
            i, hybrid_rep_candidates[i].first, hybrid_rep_candidates[i].second.last_access_time);
        }
      }
    }    
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
void tag_array::warp_interference_awared_modification_for_srrip(
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  unsigned& valid_line,
  unsigned& warp_id, unsigned& core_id,
  mem_fetch *mf, gpgpu_sim *gpu
) {
  if (m_config.m_mshr_corr_repl == 'T') {
    std::vector<std::pair<unsigned, LINE_RECENCY>> hybrid_rep_candidates_in_use;
    if (hybrid_rep_candidates_no_record_in_mshr.size()) {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_no_record_in_mshr;
    } else {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_recorded_in_mshr;
    }

    std::sort(hybrid_rep_candidates_in_use.begin(), hybrid_rep_candidates_in_use.end(), 
      [gpu, mf](const std::pair<unsigned, LINE_RECENCY>& a, 
         const std::pair<unsigned, LINE_RECENCY>& b) {
         return gpu->get_shader_stats()->warp_interfere[mf->get_sid()][a.second.warp_id][mf->get_wid()] <
                gpu->get_shader_stats()->warp_interfere[mf->get_sid()][b.second.warp_id][mf->get_wid()];
      });
    if (hybrid_rep_candidates_in_use[0].second.rrpv == hybrid_rep_candidates_in_use[0].second.max_rrpv) {
      valid_line = hybrid_rep_candidates_in_use[0].first;
      warp_id    = hybrid_rep_candidates_in_use[0].second.warp_id;
      core_id    = hybrid_rep_candidates_in_use[0].second.core_id;
    }
  } else {
    std::sort(hybrid_rep_candidates.begin(), hybrid_rep_candidates.end(), 
      [gpu, mf](const std::pair<unsigned, LINE_RECENCY>& a, 
         const std::pair<unsigned, LINE_RECENCY>& b) {
         return gpu->get_shader_stats()->warp_interfere[mf->get_sid()][a.second.warp_id][mf->get_wid()] <
                gpu->get_shader_stats()->warp_interfere[mf->get_sid()][b.second.warp_id][mf->get_wid()];
      });
    if (hybrid_rep_candidates[0].second.rrpv == hybrid_rep_candidates[0].second.max_rrpv) {
      valid_line = hybrid_rep_candidates[0].first;
      warp_id    = hybrid_rep_candidates[0].second.warp_id;
      core_id    = hybrid_rep_candidates[0].second.core_id;
    }
  }
}

void tag_array::fill_time_awared_modification_for_lru(
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  unsigned& valid_line,
  const unsigned long long& smallest_last_access_time) {

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
void tag_array::mshr_corr_warp_interference_awared_pick(
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
  std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
  unsigned& valid_line,
  unsigned& warp_id, unsigned& core_id,
  mem_fetch *mf, gpgpu_sim *gpu, 
  unsigned long long time) {

  if (m_config.m_mshr_corr_repl == 'T') {
    std::vector<std::pair<unsigned, LINE_RECENCY>> hybrid_rep_candidates_in_use;
    if (hybrid_rep_candidates_no_record_in_mshr.size()) {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_no_record_in_mshr;
    } else {
      hybrid_rep_candidates_in_use = hybrid_rep_candidates_recorded_in_mshr;
    }
    std::sort(hybrid_rep_candidates_in_use.begin(), hybrid_rep_candidates_in_use.end(), 
      [gpu, mf](const std::pair<unsigned, LINE_RECENCY>& a, 
         const std::pair<unsigned, LINE_RECENCY>& b) {
         return gpu->get_shader_stats()->warp_interfere[mf->get_sid()][a.second.warp_id][mf->get_wid()] <
                gpu->get_shader_stats()->warp_interfere[mf->get_sid()][b.second.warp_id][mf->get_wid()];
      });
    valid_line = hybrid_rep_candidates_in_use[0].first;
    warp_id    = hybrid_rep_candidates_in_use[0].second.warp_id;
    core_id    = hybrid_rep_candidates_in_use[0].second.core_id;
  } else {
    if (DTRACE(VERIFY_WARP_INTERFERED_SCHED)) {
      for (unsigned i = 0; i < hybrid_rep_candidates.size(); i++) {
        fprintf(Trace::out, "warp_interfere[sid:%u][interfered:%u][interfering:%u] = %u\n", 
          mf->get_sid(), hybrid_rep_candidates[i].second.warp_id, mf->get_wid(), 
          gpu->get_shader_stats()->warp_interfere[mf->get_sid()][hybrid_rep_candidates[i].second.warp_id][mf->get_wid()]);
      }
    }

    std::sort(hybrid_rep_candidates.begin(), hybrid_rep_candidates.end(), 
      [gpu, mf](const std::pair<unsigned, LINE_RECENCY>& a, 
         const std::pair<unsigned, LINE_RECENCY>& b) {
         return gpu->get_shader_stats()->warp_interfere[mf->get_sid()][a.second.warp_id][mf->get_wid()] <
                gpu->get_shader_stats()->warp_interfere[mf->get_sid()][b.second.warp_id][mf->get_wid()];
      });

    if (DTRACE(VERIFY_WARP_INTERFERED_SCHED) || DTRACE(WARP_INTERFERE_AWARED_SCHED)) {
      if (time == 54810) {
        fprintf(Trace::out, "After sorting..................\n");
        for (unsigned i = 0; i < hybrid_rep_candidates.size(); i++) {
          fprintf(Trace::out, "warp_interfere[sid:%u][interfered:%u][interfering:%u] = %u. valid_line = %u\n", 
            mf->get_sid(), hybrid_rep_candidates[i].second.warp_id, mf->get_wid(), 
            gpu->get_shader_stats()->warp_interfere[mf->get_sid()][hybrid_rep_candidates[i].second.warp_id][mf->get_wid()],
            hybrid_rep_candidates[i].first
          );
        }
      }
    }

    valid_line = hybrid_rep_candidates[0].first; // update valid_line 
    warp_id    = hybrid_rep_candidates[0].second.warp_id;
    core_id    = hybrid_rep_candidates[0].second.core_id;

    if (DTRACE(WARP_INTERFERE_AWARED_SCHED)) {
      if (time == 54810) {
        fprintf(Trace::out, "-------- Finally picked warp_interfere[sid:%u][interfered:%u][interfering:%u] = %u. "
          "valid_line = %u\n", 
          mf->get_sid(), warp_id, mf->get_wid(), 
          gpu->get_shader_stats()->warp_interfere[mf->get_sid()][warp_id][mf->get_wid()],
          valid_line
        );
      }
    }

    // valid_line = hybrid_rep_candidates[hybrid_rep_candidates.size() >> 1].first; // update valid_line 
    // warp_id    = hybrid_rep_candidates[hybrid_rep_candidates.size() >> 1].second.warp_id;
    // core_id    = hybrid_rep_candidates[hybrid_rep_candidates.size() >> 1].second.core_id;

    // 34.381 (+0.194%)
    // unsigned picker = (hybrid_rep_candidates.size() >> 1) + (hybrid_rep_candidates.size() >> 2);
    // 34.205 (-0.320%)
    // unsigned picker = (hybrid_rep_candidates.size() >> 2);
    // unsigned picker = (hybrid_rep_candidates.size() >> 1);
    // valid_line = hybrid_rep_candidates[picker].first; // update valid_line 
    // warp_id    = hybrid_rep_candidates[picker].second.warp_id;
    // core_id    = hybrid_rep_candidates[picker].second.core_id;
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

enum cache_request_status tag_array::probe(
  const std::string& caller,
  gpgpu_sim *gpu,
  new_addr_type addr, unsigned &idx,
  mem_access_sector_mask_t mask,
  bool is_write, 
  unsigned long long time,
  bool probe_mode,
  bool& got_warp_interfere_info, 
  WARP_INTERFERE_RECORD& warp_interfere_record,
  mem_fetch *mf) {

  if (DTRACE(TAG_PROBE)) {
    fprintf(Trace::out, "%llu %s called tag_array::probe(3rd in-arg mask) addr:%#llx\n", 
      time, caller.c_str(), addr);
  }

  unsigned set_index = m_config.set_index(addr);
  new_addr_type tag = m_config.tag(addr);

  // Just for checking if (tag === block_addr) (It seems not) // 1-14
  [[maybe_unused]] new_addr_type block_addr = m_config.block_addr(addr); 

  // for debug
  if (m_config.get_sif() != 'L') {
    assert(1);
    // assert(addr == m_config.block_addr(addr));
    // assert(tag == m_config.block_addr(addr));
  }

  std::string str_cache_name = m_config.get_cache_name();
  if (!strcmp(m_config.get_cache_name(), "L2") && mf) {
    str_cache_name += "_sub[";
    str_cache_name += std::to_string(mf->get_sub_partition());
    str_cache_name += "]";
  } else if (!strcmp(m_config.get_cache_name(), "L1D") && mf) {
    // printf("tag probe for L1D");
  }

  unsigned invalid_line = (unsigned) - 1;
  unsigned valid_line = (unsigned) - 1;
  unsigned long long valid_timestamp = (unsigned) - 1;
  
  bool srrip_has_picked = false;

  bool all_reserved = true;
  bool all_miss = true;
  bool all_sector_valid = true;

  bool lru_has_picked            = false;
  unsigned lru_picked_line       = (unsigned) - 1;
  unsigned lru_picked_total_hits = (unsigned) - 1;
  unsigned long long lru_picked_avg_evict_interval = (unsigned long long) - 1;
  unsigned last_warp_id = (unsigned) - 1;
  unsigned last_core_id = (unsigned) - 1;

  std::vector<std::pair<unsigned /* unfolded index */, LINE_RECENCY>> hybrid_rep_candidates;
  std::vector<std::pair<unsigned /* unfolded index */, LINE_RECENCY>> hybrid_rep_candidates_no_record_in_mshr;
  std::vector<std::pair<unsigned /* unfolded index */, LINE_RECENCY>> hybrid_rep_candidates_recorded_in_mshr;

  bool cache_hit = false;
  [[maybe_unused]] bool has_unreserved_line = false;
  
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];

    if (line->m_tag == tag) {
      line->inc_total_hits();
      cache_hit = true;
      lines_locality[addr]++;
      // unique_lines.insert(addr);
      if (mf) {
        m_unique_lines[mf->get_sid()].insert(addr);
      }      
      if (DTRACE(DATA_LOCALITY)) {
        if (mf) {
          fprintf(Trace::out, "%llu %s lines_locality: "
            "{size = %lu, lines_locality[addr:%#llx]++ = %u}. "
            "m_unique_lines[sid:%u]: {wid:%u size:%lu}\n",
            time, str_cache_name.c_str(), 
            lines_locality.size(), addr, lines_locality[addr],
            mf->get_sid(), mf->get_wid(), m_unique_lines[mf->get_sid()].size());
        }
      } // if (DTRACE(DATA_LOCALITY)) {

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
        // 2/13 debug
        if (DTRACE(DEBUG_SINGLE_MF)) {
          if (mf) {
            fprintf(Trace::out, "%llu return HIT_RESERVED for addr = %#llx wid = %u sid = %u\n",
              time, mf->get_addr(), mf->get_wid(), mf->get_sid());
          }
        }
        return HIT_RESERVED;
      } else if (line->get_status(mask) == VALID) {
        idx = index;
        return HIT;
      } else if (line->get_status(mask) == MODIFIED) {
        if ((!is_write && line->is_readable(mask)) || is_write) {        
          idx = index;
          return HIT;
        } else {
          idx = index;
          return SECTOR_MISS;
        }
      } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
        idx = index;
        return SECTOR_MISS;
      } else {
        assert(line->get_status(mask) == INVALID);
        all_sector_valid = false;
      }
    } // cacheline hit

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
        } else {
          gather_rep_candidates(
            line, index,
            hybrid_rep_candidates_no_record_in_mshr,
            hybrid_rep_candidates_recorded_in_mshr,
            hybrid_rep_candidates);

          if (m_config.m_warp_interfere_aware == 'T') {
            // nothing
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
    idx = invalid_line;
  } else if (m_config.m_warp_interfere_aware == 'T') {
    assert(valid_line == (unsigned) - 1);
    warp_interfere_awared_pick(
      hybrid_rep_candidates,
      hybrid_rep_candidates_no_record_in_mshr,
      hybrid_rep_candidates_recorded_in_mshr,
      valid_line /* later used */, 
      last_warp_id, last_core_id, mf);
    idx = valid_line;
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

        if (m_config.m_total_hits_ascend == 'T') {
          pick_modified_by_total_hits_ascend(
            hybrid_rep_candidates,
            hybrid_rep_candidates_no_record_in_mshr,
            hybrid_rep_candidates_recorded_in_mshr,
            valid_line,
            lru_picked_total_hits
          );
        } else if (m_config.m_fill_time_ascend == 'T') {
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
        if (m_config.m_total_hits_ascend == 'T') {
          pick_modified_by_total_hits_ascend(
            hybrid_rep_candidates,
            hybrid_rep_candidates_no_record_in_mshr,
            hybrid_rep_candidates_recorded_in_mshr,
            valid_line,
            lru_picked_total_hits
          );
        } else if (m_config.m_fill_time_ascend == 'T') {
          fill_time_awared_modification_for_srrip(
            hybrid_rep_candidates,
            hybrid_rep_candidates_no_record_in_mshr,
            hybrid_rep_candidates_recorded_in_mshr,
            valid_line);
        }
      } // SRRIP
      m_lines[valid_line]->update_recency_info(time);
      if (mf) {
        mf->set_victim_avg_evict_interval(m_lines[valid_line]->get_avg_evict_interval());
      }
      idx = valid_line;
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
        if (m_config.m_total_hits_ascend == 'T') {
          pick_modified_by_total_hits_ascend(
            hybrid_rep_candidates, 
            hybrid_rep_candidates_no_record_in_mshr,
            hybrid_rep_candidates_recorded_in_mshr,
            lru_picked_line, lru_picked_total_hits
          );
          idx = lru_picked_line;
        } else if (m_config.m_fill_time_ascend == 'T') {
          fill_time_awared_modification_for_lru(
            hybrid_rep_candidates,
            hybrid_rep_candidates_no_record_in_mshr,
            hybrid_rep_candidates_recorded_in_mshr,
            idx,
            valid_timestamp);
        }
        if (DTRACE(LRU_SAVED_SRRIP_PICKING)) {
          fprintf(Trace::out, "%llu %s LRU saved SRRIP picking idx:%x\n",
            time, m_config.m_cache_name, idx);
        } 
      } // SRRIP
    } // else if (valid_line == (unsigned) - 1) 
  } // else if (m_config.m_warp_interfere_aware == 'F')

  bool has_warp_interfere = 
    !strcmp(m_config.get_cache_name(), "L1D") && mf && 
    idx != ((unsigned) - 1) && 
    valid_line != ((unsigned) - 1) &&
    last_warp_id != mf->get_wid();
  if (has_warp_interfere) {
    assert(last_core_id == mf->get_sid());
    got_warp_interfere_info = true;
    warp_interfere_record.last_warp_id = last_warp_id;
    warp_interfere_record.curr_warp_id = mf->get_wid();
  }
  if (DTRACE(WARP_INTERFERE)) {
    if (has_warp_interfere) {
      fprintf(Trace::out, "%llu warp:%u evicted idx:%#x hit by warp:%u last time\n",
        time, mf->get_wid(), idx, last_warp_id);
    }
  }

  if (DTRACE(WARP_INTERFERE_AWARED_SCHED)) {
    if (time == 54810) 
    {
      fprintf(Trace::out, "%llu caller is %s. "
        "idx = %#x %s\n", time, caller.c_str(),
        idx, str_cache_name.c_str());
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

enum cache_request_status tag_array::access(gpgpu_sim *gpu, new_addr_type addr, 
                                            unsigned long long time,
                                            unsigned &idx, mem_fetch *mf) {
  bool wb = false;
  evicted_block_info evicted;
  enum cache_request_status result = access(gpu, addr, time, idx, wb, evicted, mf);
  assert(!wb);
  return result;
}

enum cache_request_status tag_array::access(gpgpu_sim *gpu, new_addr_type addr, 
                                            unsigned long long time,
                                            unsigned &idx, bool &wb,
                                            evicted_block_info &evicted,
                                            mem_fetch *mf) {
  m_access++;
  is_used = true;
  shader_cache_access_log(m_core_id, m_type_id, 0);  // log accesses to cache

  bool got_warp_interfere_info = false;
  WARP_INTERFERE_RECORD warp_interfere_record((unsigned )- 1, (unsigned) - 1);

  enum cache_request_status status = 
    probe("tag_array::access", gpu, addr, idx, mf, mf->is_write(), time, 
      got_warp_interfere_info, warp_interfere_record);

  m_lines[idx]->set_last_warp_id(mf->get_wid());
  m_lines[idx]->set_last_core_id(mf->get_sid());

  switch (status) {
    case HIT_RESERVED:
      m_pending_hit++;
    case HIT:
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
          // m_lines[idx]->set_byte_mask(mf);
          evicted.set_info(m_lines[idx]->m_block_addr,
                           m_lines[idx]->get_modified_size(),
                           m_lines[idx]->get_dirty_byte_mask(),
                           m_lines[idx]->get_dirty_sector_mask());
          m_dirty--;
        }
        m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr),
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

void tag_array::fill(gpgpu_sim *gpu, new_addr_type addr, unsigned long long time, mem_fetch *mf,
                     bool is_write) {
  fill(gpu, addr, time, mf->get_access_sector_mask(), mf->get_access_byte_mask(), is_write);
}

void tag_array::fill(gpgpu_sim *gpu, new_addr_type addr, unsigned long long time,
                     mem_access_sector_mask_t mask,
                     mem_access_byte_mask_t byte_mask, bool is_write) {
  // assert( m_config.m_alloc_policy == ON_FILL );
  unsigned idx;

  bool got_warp_interfere_info = false;
  WARP_INTERFERE_RECORD warp_interfere_record((unsigned )- 1, (unsigned) - 1);

  enum cache_request_status status = 
    probe("tag_array::fill", gpu, addr, idx, mask, is_write, time, 
      false /* probe_mode */,
      got_warp_interfere_info, warp_interfere_record);

  if (status == RESERVATION_FAIL) {
    return;
  }

  bool before = m_lines[idx]->is_modified_line();
  // assert(status==MISS||status==SECTOR_MISS); // MSHR should have prevented
  // redundant memory request
  if (status == MISS) {
    m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr), time, mask);
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

void tag_array::fill(unsigned index, unsigned long long time, mem_fetch *mf) {
  assert(m_config.m_alloc_policy == ON_MISS);
  bool before = m_lines[index]->is_modified_line();
  m_lines[index]->fill(time, mf->get_access_sector_mask(),
                       mf->get_access_byte_mask());
  if (m_lines[index]->is_modified_line() && !before) {
    m_dirty++;
  }
}

void tag_array::set_recorded_in_mshr(unsigned index, unsigned long long time) {
  m_lines[index]->m_was_recorded_in_mshr = true;
  m_lines[index]->m_record_interval_in_mshr = 
    !(m_lines[index]->m_recorded_times_in_mshr) ? time :
    (time - m_lines[index]->m_last_record_time_in_mshr); 

  m_lines[index]->m_avg_record_interval_in_mshr = 
    !(m_lines[index]->m_recorded_times_in_mshr) ? time :
    ((m_lines[index]->m_avg_record_interval_in_mshr + 
      m_lines[index]->m_record_interval_in_mshr) >> 1); 

  m_lines[index]->m_last_record_time_in_mshr = 
    !(m_lines[index]->m_recorded_times_in_mshr) ? 0 : time;

  m_lines[index]->m_recorded_times_in_mshr++;
}

// TODO: we need write back the flushed data to the upper level
void tag_array::flush() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    if (m_lines[i]->is_modified_line()) {
      for (unsigned j = 0; j < SECTOR_CHUNK_SIZE; j++) {
        m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));
      }
    }

  m_dirty = 0;
  is_used = false;
}

void tag_array::invalidate() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    for (unsigned j = 0; j < SECTOR_CHUNK_SIZE; j++)
      m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));

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
void mshr_table::mark_ready(const char* cache_name, new_addr_type block_addr, bool &has_atomic, unsigned long long cycle) {
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
mem_fetch* mshr_table::next_access(const char* cache_type, unsigned long long cycle) {
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
      cache_type, (unsigned long long)block_addr,
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
        (unsigned long long)block_addr,
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
    unsigned long long block_addr = e->first;
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
  
  m_l1d_miss_served_cycles.clear();  
  m_l1d_misses.clear();
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
  unsigned long long streamID, unsigned sm_id, unsigned warp_id) {
  return m_mshr_occupancy_stats[streamID][sm_id][warp_id];
}

void cache_stats::inc_mshr_stats(
  unsigned long long streamID, unsigned sm_id, unsigned warp_id) {

  const unsigned sms = 4; // gpgpu_n_cores_per_cluster
  const unsigned max_warps_per_sm = 64; // m_config.max_warps_per_sm

  if (m_mshr_occupancy_stats.find(streamID) == m_mshr_occupancy_stats.end()) {
    std::vector<std::vector<unsigned>> new_val;
    new_val.resize(sms);
    for (unsigned sm = 0; sm < sms; ++sm) {
      new_val[sm].resize(max_warps_per_sm, 0);
    }
    m_mshr_occupancy_stats.insert(std::pair<unsigned long long,
        std::vector<std::vector<unsigned>>>(streamID, new_val));
  }
  m_mshr_occupancy_stats.at(streamID)[sm_id][warp_id]++;
}

void cache_stats::inc_accu_l2_dram_queue_size(
  unsigned long long streamID, unsigned l2_sub, unsigned size) {

  if (m_accu_l2_dram_queue_size.find(streamID) == m_accu_l2_dram_queue_size.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(get_sub_partitions());
    m_accu_l2_dram_queue_size.insert(std::pair<unsigned long long,
        std::vector<unsigned>>(streamID, new_val));
  }
  m_accu_l2_dram_queue_size.at(streamID)[l2_sub] += size;
}
void cache_stats::inc_accu_l2_icnt_queue_size(
  unsigned long long streamID, unsigned l2_sub, unsigned size) {

  if (m_accu_l2_icnt_queue_size.find(streamID) == m_accu_l2_icnt_queue_size.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(get_sub_partitions());
    m_accu_l2_icnt_queue_size.insert(std::pair<unsigned long long,
        std::vector<unsigned>>(streamID, new_val));
  }
  m_accu_l2_icnt_queue_size.at(streamID)[l2_sub] += size;
}
void cache_stats::inc_l2_dram_q_accesses(unsigned long long streamID, unsigned l2_sub) {

  if (m_l2_dram_q_accesses.find(streamID) == m_l2_dram_q_accesses.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(get_sub_partitions());
    m_l2_dram_q_accesses.insert(std::pair<unsigned long long,
        std::vector<unsigned>>(streamID, new_val));
  }
  m_l2_dram_q_accesses.at(streamID)[l2_sub]++;
}
void cache_stats::inc_l2_icnt_q_accesses(unsigned long long streamID, unsigned l2_sub) {

  if (m_l2_icnt_q_accesses.find(streamID) == m_l2_icnt_q_accesses.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(get_sub_partitions());
    m_l2_icnt_q_accesses.insert(std::pair<unsigned long long,
        std::vector<unsigned>>(streamID, new_val));
  }
  m_l2_icnt_q_accesses.at(streamID)[l2_sub]++;
}

void cache_stats::inc_l2_mshr_slots_fills(unsigned long long streamID, unsigned l2_sub) {

  if (m_l2_mshr_slots_fills.find(streamID) == m_l2_mshr_slots_fills.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(get_sub_partitions());
    m_l2_mshr_slots_fills.insert(std::pair<unsigned long long,
        std::vector<unsigned>>(streamID, new_val));
  }
  m_l2_mshr_slots_fills.at(streamID)[l2_sub]++;
}

void cache_stats::inc_l1d_miss_served_cycles(
  unsigned long long streamID, unsigned long long served_cycles) {
  if (m_l1d_miss_served_cycles.find(streamID) == m_l1d_miss_served_cycles.end()) {
    unsigned long long new_val;
    m_l1d_miss_served_cycles.insert(
      std::pair<unsigned long long, unsigned long long>(streamID, new_val));    
  }
  m_l1d_miss_served_cycles.at(streamID) += served_cycles;
}
void cache_stats::inc_l1d_misses(unsigned long long streamID) {
  if (m_l1d_misses.find(streamID) == m_l1d_misses.end()) {
    unsigned new_val;
    m_l1d_misses.insert(std::pair<unsigned long long, unsigned>(streamID, new_val));    
  }
  m_l1d_misses.at(streamID)++;
}

void cache_stats::inc_l2_sub_miss_served_cycles(
  unsigned long long streamID, unsigned l2_sub,
  unsigned long long served_cycles) {
  if (m_l2_sub_miss_served_cycles.find(streamID) == m_l2_sub_miss_served_cycles.end()) {
    std::vector<unsigned long long> new_val;
    new_val.resize(get_sub_partitions());
    m_l2_sub_miss_served_cycles.insert(
      std::pair<unsigned long long, std::vector<unsigned long long>>(streamID, new_val));    
  }
  m_l2_sub_miss_served_cycles.at(streamID)[l2_sub] += served_cycles;
}
void cache_stats::inc_l2_sub_misses(unsigned long long streamID, unsigned l2_sub) {
  if (m_l2_sub_misses.find(streamID) == m_l2_sub_misses.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(get_sub_partitions());
    m_l2_sub_misses.insert(
      std::pair<unsigned long long, std::vector<unsigned>>(streamID, new_val));    
  }
  m_l2_sub_misses.at(streamID)[l2_sub]++;
}

void cache_stats::inc_l2_miss_q_pops() {
  m_l2_miss_q_pops++;
}

void cache_stats::gather_lines_stats(unsigned long long streamID, unsigned unfolded_index) {
  if (m_lines_evictons.find(streamID) == m_lines_evictons.end()) {
    std::vector<unsigned> new_val;
    new_val.resize(1); 
  }
}

void cache_stats::inc_stats(int access_type, int access_outcome,
                            unsigned long long streamID) {
  ///
  /// Increment the stat corresponding to (access_type, access_outcome) by 1.
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");

  if (m_stats.find(streamID) == m_stats.end()) {
    std::vector<std::vector<unsigned long long>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0);
    }
    m_stats.insert(std::pair<unsigned long long,
        std::vector<std::vector<unsigned long long>>>(streamID, new_val));
  }
  m_stats.at(streamID)[access_type][access_outcome]++;
}

void cache_stats::update_evict_stats(
    unsigned long long streamID, 
    unsigned long long victim_avg_evict_interval) {
  
  if (m_evict_stats.find(streamID) == m_evict_stats.end()) {
    unsigned new_val;
    m_evict_stats.insert(std::pair<unsigned long long, unsigned>(streamID, new_val));
  }

  m_evict_stats.at(streamID) = 
  (m_evict_stats.at(streamID) + victim_avg_evict_interval) >> 1;
}

void cache_stats::inc_stats_pw(int access_type, int access_outcome,
                               unsigned long long streamID) {
  ///
  /// Increment the corresponding per-window cache stat
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");

  if (m_stats_pw.find(streamID) == m_stats_pw.end()) {
    std::vector<std::vector<unsigned long long>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0);
    }
    m_stats_pw.insert(std::pair<unsigned long long,
                                std::vector<std::vector<unsigned long long>>>(
        streamID, new_val));
  }
  m_stats_pw.at(streamID)[access_type][access_outcome]++;
}

void cache_stats::inc_fail_stats(
  int access_type, int fail_outcome, 
  unsigned long long streamID, int fail_driver) {

  if (!check_fail_valid(access_type, fail_outcome))
    assert(0 && "Unknown cache access type or access fail");

  if (m_fail_stats.find(streamID) == m_fail_stats.end()) {
    std::vector<std::vector<unsigned long long>> new_val;    
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_RESERVATION_FAIL_STATUS, 0);
    }
        
    m_fail_stats.insert(std::pair<unsigned long long,
                        std::vector<std::vector<unsigned long long>>>(
                        streamID, new_val));
    m_fail_stats_total.insert(
      std::pair<unsigned long long, std::vector<unsigned long long>>(
        streamID, std::vector<unsigned long long>(NUM_MEM_ACCESS_TYPE, 0)));
  } // if (m_fail_stats.find(streamID) == m_fail_stats.end()) { ---> Create new entry
  m_fail_stats.at(streamID)[access_type][fail_outcome]++;
  m_fail_stats_total.at(streamID)[access_type]++;

  if (m_line_alloc_fail.find(streamID) == m_line_alloc_fail.end()) {
    std::vector<std::vector<unsigned long long>> new_line_alloc_fail_driver;
    new_line_alloc_fail_driver.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_line_alloc_fail_driver[j].resize(NUM_LINE_ALLOC_FAIL_DRIVER, 0);
    }    
    m_line_alloc_fail.insert(std::pair<unsigned long long,
      std::vector<std::vector<unsigned long long>>>(streamID, new_line_alloc_fail_driver));
  }
  if (m_mshr_entry_fail.find(streamID) == m_mshr_entry_fail.end()) {
    std::vector<std::vector<unsigned long long>> new_mshr_entry_fail_driver;
    new_mshr_entry_fail_driver.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_mshr_entry_fail_driver[j].resize(NUM_MSHR_ENTRY_FAIL_DRIVER, 0);
    }    
    m_mshr_entry_fail.insert(std::pair<unsigned long long,
      std::vector<std::vector<unsigned long long>>>(streamID, new_mshr_entry_fail_driver));
  }
  if (m_miss_q_full.find(streamID) == m_miss_q_full.end()) {
    std::vector<std::vector<unsigned long long>> new_miss_q_full_driver;
    new_miss_q_full_driver.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_miss_q_full_driver[j].resize(NUM_MISS_QUEUE_FULL_DRIVER, 0);
    }    
    m_miss_q_full.insert(std::pair<unsigned long long,
      std::vector<std::vector<unsigned long long>>>(streamID, new_miss_q_full_driver));
  }
  if (m_mshr_merge_entry_fail.find(streamID) == m_mshr_merge_entry_fail.end()) {
    std::vector<std::vector<unsigned long long>> new_mshr_merge_entry_fail_driver;
    new_mshr_merge_entry_fail_driver.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_mshr_merge_entry_fail_driver[j].resize(NUM_MSHR_MERGE_ENTRY_FAIL_DRIVER, 0);
    }    
    m_mshr_merge_entry_fail.insert(std::pair<unsigned long long,
      std::vector<std::vector<unsigned long long>>>(streamID, new_mshr_merge_entry_fail_driver));
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

unsigned long long &cache_stats::operator()(int access_type, int access_outcome,
                                            bool fail_outcome,
                                            unsigned long long streamID) {
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

unsigned long long cache_stats::operator()(
  int access_type, 
  int access_outcome,
  bool is_fail_outcome,  
  int fail_driver, unsigned long long streamID) const {

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
    return (unsigned long long) - 1;
  }
}

unsigned long long cache_stats::operator()(int access_type, int access_outcome,
                                           bool fail_outcome,
                                           unsigned long long streamID) const {
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

unsigned long long cache_stats::operator()(
  unsigned sm, unsigned warp, unsigned long long streamID) const {
  if (!check_valid(sm, warp)) {
    assert(0 && "Unknown sm_id or warp_id");
  }
  return m_mshr_occupancy_stats.at(streamID)[sm][warp];
}

unsigned cache_stats::operator()(
  unsigned l2_sub, unsigned long long streamID) const {
  auto it = m_accu_l2_dram_queue_size.find(streamID);
  if (it == m_accu_l2_dram_queue_size.end()) return 0;
  if (l2_sub >= it->second.size()) return 0;
  return it->second[l2_sub];
}
unsigned long long cache_stats::operator()(
  unsigned l2_sub, unsigned long long streamID, const char* tgt_name) const {
  if (!strcmp(tgt_name, "m_l2_sub_miss_served_cycles")) {
    auto it = m_l2_sub_miss_served_cycles.find(streamID);
    if (it == m_l2_sub_miss_served_cycles.end()) {
      return 0;
    } else {
      return it->second[l2_sub];
    }
  } else if (!strcmp(tgt_name, "m_l2_sub_misses")) {
    auto it = m_l2_sub_misses.find(streamID);
    if (it == m_l2_sub_misses.end()) {
      return 0;
    } else {
      return it->second[l2_sub];
    }
  } 
}
unsigned long long cache_stats::operator()(
  unsigned long long streamID, const char* tgt_name) const {
  if (!strcmp(tgt_name, "m_l1d_miss_served_cycles")) {
    auto it = m_l1d_miss_served_cycles.find(streamID);
    if (it == m_l1d_miss_served_cycles.end()) {
      return 0;
    } else {
      return it->second;
    }
  } else if (!strcmp(tgt_name, "m_l1d_misses")) {
    auto it = m_l1d_misses.find(streamID);
    if (it == m_l1d_misses.end()) {
      return 0;
    } else {
      return it->second;
    }
  }
}

cache_stats cache_stats::operator+(const cache_stats &cs) {
  ///
  /// Overloaded + operator to allow for simple stat accumulation
  ///
  // 1-1 Init
  cache_stats ret;
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_stats.insert(std::pair<unsigned long long,
      std::vector<std::vector<unsigned long long>>>(streamID, m_stats.at(streamID)));
  }
  for (auto iter = m_stats_pw.begin(); iter != m_stats_pw.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_stats_pw.insert(std::pair<unsigned long long,
      std::vector<std::vector<unsigned long long>>>(streamID, m_stats_pw.at(streamID)));
  }

  for (auto iter = m_fail_stats.begin(); iter != m_fail_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_fail_stats.insert(std::pair<unsigned long long,
      std::vector<std::vector<unsigned long long>>>(streamID, m_fail_stats.at(streamID)));
  }
  for (auto iter = m_line_alloc_fail.begin(); iter != m_line_alloc_fail.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_line_alloc_fail.insert(std::pair<unsigned long long,
      std::vector<std::vector<unsigned long long>>>(streamID, m_line_alloc_fail.at(streamID)));
  }  
  for (auto iter = m_mshr_entry_fail.begin(); iter != m_mshr_entry_fail.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_mshr_entry_fail.insert(std::pair<unsigned long long,
      std::vector<std::vector<unsigned long long>>>(streamID, m_mshr_entry_fail.at(streamID)));
  }    
  for (auto iter = m_miss_q_full.begin(); iter != m_miss_q_full.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_miss_q_full.insert(std::pair<unsigned long long,
      std::vector<std::vector<unsigned long long>>>(streamID, m_miss_q_full.at(streamID)));
  }  
  for (auto iter = m_mshr_merge_entry_fail.begin(); iter != m_mshr_merge_entry_fail.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_mshr_merge_entry_fail.insert(std::pair<unsigned long long,
      std::vector<std::vector<unsigned long long>>>(streamID, m_mshr_merge_entry_fail.at(streamID)));
  }  
  for (auto iter = m_fail_stats_total.begin(); iter != m_fail_stats_total.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_fail_stats_total.insert(
      std::pair<unsigned long long, std::vector<unsigned long long>>(
      streamID, m_fail_stats_total.at(streamID)));      
  }
  for (auto iter = m_mshr_occupancy_stats.begin(); iter != m_mshr_occupancy_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_mshr_occupancy_stats.insert(
      std::pair<unsigned long long,
        std::vector<std::vector<unsigned>>>(streamID, m_mshr_occupancy_stats.at(streamID)));
  }
  for (auto iter = m_l1d_miss_served_cycles.begin(); 
    iter != m_l1d_miss_served_cycles.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_l1d_miss_served_cycles.insert(
      std::pair<unsigned long long, unsigned long long>(
        streamID, m_l1d_miss_served_cycles.at(streamID)));
  }  
  for (auto iter = m_l1d_misses.begin(); 
    iter != m_l1d_misses.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_l1d_misses.insert(
      std::pair<unsigned long long, unsigned>(streamID, m_l1d_misses.at(streamID)));
  }
  for (auto iter = m_l2_sub_miss_served_cycles.begin(); 
    iter != m_l2_sub_miss_served_cycles.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_l2_sub_miss_served_cycles.insert(
      std::pair<unsigned long long, std::vector<unsigned long long>>(
        streamID, m_l2_sub_miss_served_cycles.at(streamID)));
  }  
  for (auto iter = m_l2_sub_misses.begin(); 
    iter != m_l2_sub_misses.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_l2_sub_misses.insert(
      std::pair<unsigned long long, std::vector<unsigned>>(
        streamID, m_l2_sub_misses.at(streamID)));
  }    
  for (auto iter = m_accu_l2_dram_queue_size.begin(); iter != m_accu_l2_dram_queue_size.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_accu_l2_dram_queue_size.insert(
      std::pair<unsigned long long, std::vector<unsigned>>(streamID, m_accu_l2_dram_queue_size.at(streamID)));
  }
  for (auto iter = m_accu_l2_icnt_queue_size.begin(); iter != m_accu_l2_icnt_queue_size.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_accu_l2_icnt_queue_size.insert(
      std::pair<unsigned long long, std::vector<unsigned>>(streamID, m_accu_l2_icnt_queue_size.at(streamID)));
  }

  // 1-2 Overload "+"
  for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (ret.m_stats.find(streamID) == ret.m_stats.end()) {
      ret.m_stats.insert(std::pair<unsigned long long,
          std::vector<std::vector<unsigned long long>>>(streamID, cs.m_stats.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          ret.m_stats.at(streamID)[type][status] += cs(type, status, false, streamID);
          // 12-22 for debug
          fprintf(Trace::out, "In cache_stats::operator+ ret.m_stats.at(%d)[%d][%d] += cs(%d, %d, false, %d);\n",
            (int)streamID, type, status, type, status, (int)streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_stats_pw.begin(); iter != cs.m_stats_pw.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (ret.m_stats_pw.find(streamID) == ret.m_stats_pw.end()) {
      ret.m_stats_pw.insert(std::pair<unsigned long long,
          std::vector<std::vector<unsigned long long>>>(streamID, cs.m_stats_pw.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          ret.m_stats_pw.at(streamID)[type][status] += cs(type, status, false, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_fail_stats.begin(); iter != cs.m_fail_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (ret.m_fail_stats.find(streamID) == ret.m_fail_stats.end()) {
      ret.m_fail_stats.insert(
        std::pair<unsigned long long,
            std::vector<std::vector<unsigned long long>>>(streamID, cs.m_fail_stats.at(streamID)));
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
    unsigned long long streamID = iter->first;
    if (ret.m_mshr_occupancy_stats.find(streamID) == ret.m_mshr_occupancy_stats.end()) {
      ret.m_mshr_occupancy_stats.insert(
        std::pair<unsigned long long,
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
  for (auto iter = cs.m_l1d_miss_served_cycles.begin(); 
    iter != cs.m_l1d_miss_served_cycles.end(); ++iter) {  
    unsigned long long streamID = iter->first;
    if (ret.m_l1d_miss_served_cycles.find(streamID) == 
      ret.m_l1d_miss_served_cycles.end()) {
      ret.m_l1d_miss_served_cycles.insert(
        std::pair<unsigned long long, unsigned long long>(
          streamID, cs.m_l1d_miss_served_cycles.at(streamID)));
    } else {
        ret.m_l1d_miss_served_cycles.at(streamID) += cs.m_l1d_miss_served_cycles.at(streamID);      
    }
  }
  for (auto iter = cs.m_l1d_misses.begin(); 
    iter != cs.m_l1d_misses.end(); ++iter) {  
    unsigned long long streamID = iter->first;
    if (ret.m_l1d_misses.find(streamID) == 
      ret.m_l1d_misses.end()) {
      ret.m_l1d_misses.insert(
        std::pair<unsigned long long, unsigned>(streamID, cs.m_l1d_misses.at(streamID)));
    } else {
        ret.m_l1d_misses.at(streamID) += cs.m_l1d_misses.at(streamID);      
    }
  }

  for (auto iter = cs.m_l2_sub_miss_served_cycles.begin(); 
    iter != cs.m_l2_sub_miss_served_cycles.end(); ++iter) {  
    unsigned long long streamID = iter->first;
    if (ret.m_l2_sub_miss_served_cycles.find(streamID) == 
      ret.m_l2_sub_miss_served_cycles.end()) {
      ret.m_l2_sub_miss_served_cycles.insert(
        std::pair<unsigned long long, std::vector<unsigned long long>>(
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
    unsigned long long streamID = iter->first;
    if (ret.m_l2_sub_misses.find(streamID) == 
      ret.m_l2_sub_misses.end()) {
      ret.m_l2_sub_misses.insert(
        std::pair<unsigned long long, std::vector<unsigned>>(
          streamID, cs.m_l2_sub_misses.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_l2_sub_misses.at(streamID)[l2_sub] += 
        cs.m_l2_sub_misses.at(streamID)[l2_sub];
      }
    }
  }

  for (auto iter = cs.m_accu_l2_dram_queue_size.begin(); iter != cs.m_accu_l2_dram_queue_size.end(); ++iter) {  
    unsigned long long streamID = iter->first;
    if (ret.m_accu_l2_dram_queue_size.find(streamID) == ret.m_accu_l2_dram_queue_size.end()) {
      ret.m_accu_l2_dram_queue_size.insert(
        std::pair<unsigned long long, std::vector<unsigned>>(streamID, cs.m_accu_l2_dram_queue_size.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_accu_l2_dram_queue_size.at(streamID)[l2_sub] += cs.m_accu_l2_dram_queue_size.at(streamID)[l2_sub];
      }      
    }
  }
  for (auto iter = cs.m_accu_l2_icnt_queue_size.begin(); iter != cs.m_accu_l2_icnt_queue_size.end(); ++iter) {  
    unsigned long long streamID = iter->first;
    if (ret.m_accu_l2_icnt_queue_size.find(streamID) == ret.m_accu_l2_icnt_queue_size.end()) {
      ret.m_accu_l2_icnt_queue_size.insert(
        std::pair<unsigned long long, std::vector<unsigned>>(streamID, cs.m_accu_l2_icnt_queue_size.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_accu_l2_icnt_queue_size.at(streamID)[l2_sub] += cs.m_accu_l2_icnt_queue_size.at(streamID)[l2_sub];
      }
    }
  }
  for (auto iter = cs.m_l2_dram_q_accesses.begin(); iter != cs.m_l2_dram_q_accesses.end(); ++iter) {  
    unsigned long long streamID = iter->first;
    if (ret.m_l2_dram_q_accesses.find(streamID) == ret.m_l2_dram_q_accesses.end()) {
      ret.m_l2_dram_q_accesses.insert(
        std::pair<unsigned long long, std::vector<unsigned>>(streamID, cs.m_l2_dram_q_accesses.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_l2_dram_q_accesses.at(streamID)[l2_sub] += cs.m_l2_dram_q_accesses.at(streamID)[l2_sub];
      }
    }
  }
  for (auto iter = cs.m_l2_icnt_q_accesses.begin(); iter != cs.m_l2_icnt_q_accesses.end(); ++iter) {  
    unsigned long long streamID = iter->first;
    if (ret.m_l2_icnt_q_accesses.find(streamID) == ret.m_l2_icnt_q_accesses.end()) {
      ret.m_l2_icnt_q_accesses.insert(
        std::pair<unsigned long long, std::vector<unsigned>>(streamID, cs.m_l2_icnt_q_accesses.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_l2_icnt_q_accesses.at(streamID)[l2_sub] += cs.m_l2_icnt_q_accesses.at(streamID)[l2_sub];
      }
    }
  }  
  for (auto iter = cs.m_l2_mshr_slots_fills.begin(); iter != cs.m_l2_mshr_slots_fills.end(); ++iter) {  
    unsigned long long streamID = iter->first;
    if (ret.m_l2_mshr_slots_fills.find(streamID) == ret.m_l2_mshr_slots_fills.end()) {
      ret.m_l2_mshr_slots_fills.insert(
        std::pair<unsigned long long, std::vector<unsigned>>(streamID, cs.m_l2_mshr_slots_fills.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); l2_sub++) {
        ret.m_l2_mshr_slots_fills.at(streamID)[l2_sub] += cs.m_l2_mshr_slots_fills.at(streamID)[l2_sub];
      }
    }
  }

  for (auto iter = cs.m_line_alloc_fail.begin(); iter != cs.m_line_alloc_fail.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (ret.m_line_alloc_fail.find(streamID) == ret.m_line_alloc_fail.end()) {
      ret.m_line_alloc_fail.insert(std::pair<unsigned long long,
        std::vector<std::vector<unsigned long long>>>(streamID, cs.m_line_alloc_fail.at(streamID)));
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
    unsigned long long streamID = iter->first;
    if (ret.m_mshr_entry_fail.find(streamID) == ret.m_mshr_entry_fail.end()) {
      ret.m_mshr_entry_fail.insert(std::pair<unsigned long long,
        std::vector<std::vector<unsigned long long>>>(streamID, cs.m_mshr_entry_fail.at(streamID)));
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
    unsigned long long streamID = iter->first;
    if (ret.m_miss_q_full.find(streamID) == ret.m_miss_q_full.end()) {
      ret.m_miss_q_full.insert(std::pair<unsigned long long,
        std::vector<std::vector<unsigned long long>>>(streamID, cs.m_miss_q_full.at(streamID)));
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
    unsigned long long streamID = iter->first;
    if (ret.m_mshr_merge_entry_fail.find(streamID) == ret.m_mshr_merge_entry_fail.end()) {
      ret.m_mshr_merge_entry_fail.insert(std::pair<unsigned long long,
        std::vector<std::vector<unsigned long long>>>(streamID, cs.m_mshr_merge_entry_fail.at(streamID)));
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
    unsigned long long streamID = iter->first;
    if (m_stats.find(streamID) == m_stats.end()) {
      if (DTRACE(M_STATS)) {
        fprintf(Trace::out, "%s %sm_stats.size:%lu m_stats.insert(cs.m_stats.at(streamID:%llu))\n",
          local_cache_type,
          !strcmp(local_cache_type, "L2") ? l2_prefix.c_str() : "", 
          m_stats.size(), streamID
        );
      }

      m_stats.insert(std::pair<unsigned long long,
        std::vector<std::vector<unsigned long long>>>(streamID, cs.m_stats.at(streamID)));

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
          unsigned long long orig_stats_val = m_stats.at(streamID)[type][status];
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
    unsigned long long streamID = iter->first;
    if (m_stats_pw.find(streamID) == m_stats_pw.end()) {
      m_stats_pw.insert(std::pair<unsigned long long,
                                  std::vector<std::vector<unsigned long long>>>(
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
    unsigned long long streamID = iter->first;
    if (m_fail_stats.find(streamID) == m_fail_stats.end()) {
      m_fail_stats.insert(std::pair<unsigned long long,
          std::vector<std::vector<unsigned long long>>>(streamID, cs.m_fail_stats.at(streamID)));            
      m_fail_stats_total.insert(std::pair<unsigned long long,
          std::vector<unsigned long long>>(streamID, cs.m_fail_stats_total.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS; ++status) {
          unsigned long long orig_fail_stats = m_fail_stats.at(streamID)[type][status];
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
    unsigned long long streamID = iter->first;
    if (m_line_alloc_fail.find(streamID) == m_line_alloc_fail.end()) {
      m_line_alloc_fail.insert(std::pair<unsigned long long,
          std::vector<std::vector<unsigned long long>>>(streamID, cs.m_line_alloc_fail.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned driver = 0; driver < NUM_LINE_ALLOC_FAIL_DRIVER; ++driver) {          
          m_line_alloc_fail.at(streamID)[type][driver] += cs(type, LINE_ALLOC_FAIL, true, driver, streamID);
        }
      }
    }
  }  
  for (auto iter = cs.m_mshr_entry_fail.begin(); iter != cs.m_mshr_entry_fail.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_mshr_entry_fail.find(streamID) == m_mshr_entry_fail.end()) {
      m_mshr_entry_fail.insert(std::pair<unsigned long long,
          std::vector<std::vector<unsigned long long>>>(streamID, cs.m_mshr_entry_fail.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned driver = 0; driver < NUM_MSHR_ENTRY_FAIL_DRIVER; ++driver) {          
          m_mshr_entry_fail.at(streamID)[type][driver] += cs(type, MSHR_ENTRY_FAIL, true, driver, streamID);
        }
      }
    }
  }  
  for (auto iter = cs.m_miss_q_full.begin(); iter != cs.m_miss_q_full.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_miss_q_full.find(streamID) == m_miss_q_full.end()) {
      m_miss_q_full.insert(std::pair<unsigned long long,
          std::vector<std::vector<unsigned long long>>>(streamID, cs.m_miss_q_full.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned driver = 0; driver < NUM_MISS_QUEUE_FULL_DRIVER; ++driver) {          
          m_miss_q_full.at(streamID)[type][driver] += cs(type, MISS_QUEUE_FULL, true, driver, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_mshr_merge_entry_fail.begin(); iter != cs.m_mshr_merge_entry_fail.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_mshr_merge_entry_fail.find(streamID) == m_mshr_merge_entry_fail.end()) {
      m_mshr_merge_entry_fail.insert(std::pair<unsigned long long,
          std::vector<std::vector<unsigned long long>>>(streamID, cs.m_mshr_merge_entry_fail.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned driver = 0; driver < NUM_MSHR_MERGE_ENTRY_FAIL_DRIVER; ++driver) {          
          m_mshr_merge_entry_fail.at(streamID)[type][driver] += cs(type, MSHR_MERGE_ENTRY_FAIL, true, driver, streamID);
        }
      }
    }
  }

  for (auto iter = cs.m_mshr_occupancy_stats.begin(); iter != cs.m_mshr_occupancy_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_mshr_occupancy_stats.find(streamID) == m_mshr_occupancy_stats.end()) {
      m_mshr_occupancy_stats.insert(
        std::pair<unsigned long long,
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
              local_cache_type,
              !strcmp(local_cache_type, "L2") ? l2_prefix.c_str() : "",
              streamID, sm, warp,
              orig_mshr_occupancy, m_mshr_occupancy_stats.at(streamID)[sm][warp],
              sm, warp, streamID
            );
          }
        }
      }      
    }
  } // for (auto iter = cs.m_mshr_occupancy_stats.begin(); iter != cs.m_mshr_occupancy_stats.end(); ++iter) {

  for (auto iter = cs.m_l1d_miss_served_cycles.begin(); 
    iter != cs.m_l1d_miss_served_cycles.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_l1d_miss_served_cycles.find(streamID) == m_l1d_miss_served_cycles.end()) {
      m_l1d_miss_served_cycles.insert(
        std::pair<unsigned long long, unsigned long long>(
          streamID, cs.m_l1d_miss_served_cycles.at(streamID)));
    } else {
      const char* tgt_item = "m_l1d_miss_served_cycles";
      m_l1d_miss_served_cycles.at(streamID) += cs(streamID, tgt_item);
    }
  }
  for (auto iter = cs.m_l1d_misses.begin(); 
    iter != cs.m_l1d_misses.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_l1d_misses.find(streamID) == m_l1d_misses.end()) {
      m_l1d_misses.insert(
        std::pair<unsigned long long, unsigned>(
          streamID, cs.m_l1d_misses.at(streamID)));
    } else {
      const char* tgt_item = "m_l1d_misses";
      m_l1d_misses.at(streamID) += static_cast<unsigned>(cs(streamID, tgt_item));
    }
  }  

  for (auto iter = cs.m_l2_sub_miss_served_cycles.begin(); 
    iter != cs.m_l2_sub_miss_served_cycles.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_l2_sub_miss_served_cycles.find(streamID) == m_l2_sub_miss_served_cycles.end()) {
      m_l2_sub_miss_served_cycles.insert(
        std::pair<unsigned long long, std::vector<unsigned long long>>(
          streamID, cs.m_l2_sub_miss_served_cycles.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
        const char* tgt_item = "m_l2_sub_miss_served_cycles";
        m_l2_sub_miss_served_cycles.at(streamID)[l2_sub] += cs(l2_sub, streamID, tgt_item);
        // printf("m_l2_sub_miss_served_cycles[streamID:%llu][sub:%u] += %llu\n",
        //   streamID, l2_sub, cs(l2_sub, streamID));
      }      
    }
  }
  for (auto iter = cs.m_l2_sub_misses.begin(); 
    iter != cs.m_l2_sub_misses.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_l2_sub_misses.find(streamID) == m_l2_sub_misses.end()) {
      m_l2_sub_misses.insert(
        std::pair<unsigned long long, std::vector<unsigned>>(
          streamID, cs.m_l2_sub_misses.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
        const char* tgt_item = "m_l2_sub_misses";
        m_l2_sub_misses.at(streamID)[l2_sub] += static_cast<unsigned>(cs(l2_sub, streamID, tgt_item));
        // printf("m_l2_sub_misses[streamID:%llu][sub:%u] += %u\n",
        //   streamID, l2_sub, cs(l2_sub, streamID));        
      }
    }
  }  

  for (auto iter = cs.m_accu_l2_dram_queue_size.begin(); iter != cs.m_accu_l2_dram_queue_size.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_accu_l2_dram_queue_size.find(streamID) == m_accu_l2_dram_queue_size.end()) {
      m_accu_l2_dram_queue_size.insert(
        std::pair<unsigned long long, std::vector<unsigned>>(streamID, cs.m_accu_l2_dram_queue_size.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
        m_accu_l2_dram_queue_size.at(streamID)[l2_sub] += cs(l2_sub, streamID);
      }      
    }
  }
  for (auto iter = cs.m_accu_l2_icnt_queue_size.begin(); iter != cs.m_accu_l2_icnt_queue_size.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_accu_l2_icnt_queue_size.find(streamID) == m_accu_l2_icnt_queue_size.end()) {
      m_accu_l2_icnt_queue_size.insert(
        std::pair<unsigned long long, std::vector<unsigned>>(streamID, cs.m_accu_l2_icnt_queue_size.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
        m_accu_l2_icnt_queue_size.at(streamID)[l2_sub] += cs(l2_sub, streamID);
      }      
    }
  }
  for (auto iter = cs.m_l2_dram_q_accesses.begin(); iter != cs.m_l2_dram_q_accesses.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_l2_dram_q_accesses.find(streamID) == m_l2_dram_q_accesses.end()) {
      m_l2_dram_q_accesses.insert(
        std::pair<unsigned long long, std::vector<unsigned>>(streamID, cs.m_l2_dram_q_accesses.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
        m_l2_dram_q_accesses.at(streamID)[l2_sub] += cs(l2_sub, streamID);
      }      
    }
  }  
  for (auto iter = cs.m_l2_icnt_q_accesses.begin(); iter != cs.m_l2_icnt_q_accesses.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_l2_icnt_q_accesses.find(streamID) == m_l2_icnt_q_accesses.end()) {
      m_l2_icnt_q_accesses.insert(
        std::pair<unsigned long long, std::vector<unsigned>>(streamID, cs.m_l2_icnt_q_accesses.at(streamID)));
    } else {
      for (unsigned l2_sub = 0; l2_sub < get_sub_partitions(); ++l2_sub) {
        m_l2_icnt_q_accesses.at(streamID)[l2_sub] += cs(l2_sub, streamID);
      }      
    }
  }
  for (auto iter = cs.m_l2_mshr_slots_fills.begin(); iter != cs.m_l2_mshr_slots_fills.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_l2_mshr_slots_fills.find(streamID) == m_l2_mshr_slots_fills.end()) {
      m_l2_mshr_slots_fills.insert(
        std::pair<unsigned long long, std::vector<unsigned>>(streamID, cs.m_l2_mshr_slots_fills.at(streamID)));
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

void cache_stats::print_stats(FILE *fout, unsigned long long streamID,
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
    unsigned long long streamid = iter->first;
    // when streamID is specified, skip stats for all other streams, otherwise,
    // print stats from all streams
    if ((streamID != ((unsigned long long) - 1)) && (streamid != streamID)) { 
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

void cache_stats::print_fail_stats(FILE *fout, unsigned long long streamID,
                                   const char *cache_info) const {
  std::string m_cache_info = cache_info;
  for (auto iter = m_fail_stats.begin(); iter != m_fail_stats.end(); ++iter) {
    unsigned long long streamid = iter->first;
    // when streamID is specified, skip stats for all other streams, otherwise,
    // print stats from all streams
    if ((streamID != ((unsigned long long) - 1)) && (streamid != streamID)) {
      continue;
    }

    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      fprintf(
        fout, "\t%s[%s] = %llu\n", m_cache_info.c_str(),
        mem_access_type_str((enum mem_access_type)type),
        m_fail_stats_total.at(streamid)[type]);      
      for (unsigned fail = 0; fail < NUM_CACHE_RESERVATION_FAIL_STATUS; ++fail) {
        if (m_fail_stats.at(streamid)[type][fail] > 0) {
          fprintf(
              fout, "\t%s[%s][%s] = %llu\n", m_cache_info.c_str(),
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

void cache_stats::print_mshr_stats(FILE *fout, unsigned long long streamID,
                                   const char *cache_info) const {
  for (auto iter = m_mshr_occupancy_stats.begin(); iter != m_mshr_occupancy_stats.end(); ++iter) {
    unsigned long long streamid = iter->first;
    // when streamID is specified, skip stats for all other streams, otherwise,
    // print stats from all streams
    if ((streamID != ((unsigned long long) - 1)) && (streamid != streamID)) {
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
  FILE *fout, unsigned l2_dram_q_capacity, unsigned long long streamID, const char *info) const {
  for (auto iter = m_accu_l2_dram_queue_size.begin(); 
    iter != m_accu_l2_dram_queue_size.end(); ++iter) {
    if ((streamID != ((unsigned long long) - 1)) && (iter->first != streamID)) {
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
  FILE *fout, unsigned l2_icnt_q_capacity, unsigned long long streamID, const char *info) const {
  for (auto iter = m_accu_l2_icnt_queue_size.begin(); 
    iter != m_accu_l2_icnt_queue_size.end(); ++iter) {
    if ((streamID != ((unsigned long long) - 1)) && (iter->first != streamID)) {
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

void cache_stats::print_avg_core_cache_miss_served_cycles(
  FILE* fout, unsigned long long streamID) const {
  for (auto iter = m_l1d_miss_served_cycles.begin();
    iter != m_l1d_miss_served_cycles.end(); ++iter)
  {
    if ((streamID != ((unsigned long long) - 1)) && (iter->first != streamID)) {
      continue;
    }

    float avg_l1d_miss_served_cycles = 
    ((float)m_l1d_miss_served_cycles.at(streamID) / m_l1d_misses.at(streamID));

    fprintf(fout, "\tavg_l1d_miss_served_cycles = %f\n", avg_l1d_miss_served_cycles);      
  }  
}

void cache_stats::print_avg_l2_miss_served_cycles(
  FILE* fout, unsigned long long streamID) const {
  for (auto iter = m_l2_sub_miss_served_cycles.begin();
    iter != m_l2_sub_miss_served_cycles.end(); ++iter)
  {
    if ((streamID != ((unsigned long long) - 1)) && (iter->first != streamID)) {
      continue;
    }
    unsigned long long avg_l2_miss_served_cycles = 0;
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
  FILE *fout, unsigned l2_mshr_allocated_slots, unsigned long long streamID, const char *info) const {
  for (auto iter = m_l2_mshr_slots_fills.begin(); 
    iter != m_l2_mshr_slots_fills.end(); ++iter) {
    if ((streamID != ((unsigned long long) - 1)) && (iter->first != streamID)) {
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

void cache_sub_stats::print_port_stats(FILE *fout,
                                       const char *cache_name) const {
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

unsigned long long cache_stats::get_stats(
    enum mem_access_type *access_type, unsigned num_access_type,
    enum cache_request_status *access_status,
    unsigned num_access_status) const {
  ///
  /// Returns a sum of the stats corresponding to each "access_type" and
  /// "access_status" pair. "access_type" is an array of "num_access_type"
  /// mem_access_types. "access_status" is an array of "num_access_status"
  /// cache_request_statuses.
  ///
  unsigned long long total = 0;
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
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

void cache_stats::get_sub_stats(struct cache_sub_stats &css) const {
  ///
  /// Overwrites "css" with the appropriate statistics from this cache.
  ///
  struct cache_sub_stats t_css;
  t_css.clear();

  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        if (status == HIT || status == MISS || status == SECTOR_MISS || status == HIT_RESERVED) {
          t_css.accesses += m_stats.at(streamID)[type][status];
        }      
        if (status == MISS) {
          t_css.misses += m_stats.at(streamID)[type][status];
          t_css.avg_evict_interval += m_evict_stats.at(streamID);
        }
        if (status == SECTOR_MISS) {          
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
    unsigned long long streamID = iter->first;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        if (status == HIT || status == MISS ||
            status == SECTOR_MISS || status == HIT_RESERVED) {
          t_css.accesses += m_stats_pw.at(streamID)[type][status];
        }
        if (status == HIT) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
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
  new_addr_type addr, mem_fetch *mf, unsigned time, 
  enum cache_request_status status,
  bool dump_inst_str) {

  new_addr_type block_addr = m_config.block_addr(addr);

  // std::pair<std::bitset<128>, std::bitset<128>> u128_pair = to_u128_pair(mf->get_access_byte_mask());  
  std::pair<uint64_t,uint64_t> byte_mask_hi_lo = to_u64_pair(mf->get_access_byte_mask());
  fprintf(Trace::out,
      "%llu %s%s%s %s %s addr: %#llx block_addr: %#llx "
      "byte_mask: 0x%016lx%016lx\n",
      (unsigned long long)time,
      caller,
      dump_inst_str ? m_gpu->gpgpu_ctx->func_sim->ptx_get_insn_str(mf->get_inst().pc).c_str() : "",
      m_is_l1d ? "L1D" : m_is_l2 ? "L2C" : "xx$",
      mf_request_type_str(mf->get_type()),
      cache_request_status_str(status), 
      (unsigned long long)mf->get_addr(),
      (unsigned long long)block_addr,
      byte_mask_hi_lo.first, byte_mask_hi_lo.second
    );
}

void baseline_cache::dump_cache_fill_info(
  const char* caller,
  new_addr_type addr, mem_fetch *mf, unsigned time, 
  bool dump_inst_str) {

  new_addr_type block_addr = m_config.block_addr(addr);

  std::pair<uint64_t,uint64_t> byte_mask_hi_lo = to_u64_pair(mf->get_access_byte_mask());
  fprintf(Trace::out,
      "%llu %s%s%s %s addr: %#llx block_addr: %#llx "
      "byte_mask: 0x%016lx%016lx\n",
      (unsigned long long)time,
      caller,
      dump_inst_str ? m_gpu->gpgpu_ctx->func_sim->ptx_get_insn_str(mf->get_inst().pc).c_str() : "",      
      m_is_l1d ? "L1D" : m_is_l2 ? "L2C" : "xx$",
      mf_request_type_str(mf->get_type()),
      (unsigned long long)mf->get_addr(),
      (unsigned long long)block_addr,
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
    case HIT: {
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
      }

      m_miss_queue.pop_front();
      m_memport->push(mf);

      if (DTRACE(L2_DRAM_QUEUE)) {
        if (m_is_l2) {
          fprintf(Trace::out, "%llu L2_sub[%d] l2_dram_queue added "
            "mf:{ TPC:%u SM:%u WARP:%u req_uid:%u %#llx }\n", 
            m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
            mf->get_sub_partition(),
            mf->get_tpc(), mf->get_sid(), mf->get_wid(),
            mf->get_request_uid(), mf->get_addr());
        }
      }

      if (DTRACE(MISS_QUEUE_EVENT)) {
        dumpCacheEvent(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, 
          "baseline_cache::cycle()", "m_miss_queue.pop_front -> mem_fetch_interface (ICNT)", mf);
      }
    } else {
      if (DTRACE(MISS_QUEUE_EVENT)) {
        dumpCacheEvent(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, 
          "baseline_cache::cycle()", "m_memport is full. "
          "Failed pushd mf from miss_queue to lower-level-mem-port", mf);
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
void baseline_cache::fill(mem_fetch *mf, unsigned long long time) {
  if (m_config.m_mshr_type == SECTOR_ASSOC) {
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

  // [GPGPU-SIM][TODO]
  // assert(mf->get_original_mf());

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  assert(e != m_extra_mf_fields.end());
  assert(e->second.m_valid);
  mf->set_data_size(e->second.m_data_size);
  mf->set_addr(e->second.m_addr);
  if (m_config.m_alloc_policy == ON_MISS) {
    if (DTRACE(CACHE_MISS)) {      
      dump_cache_fill_info("::fill ", e->second.m_addr, mf, time);
    }
    if (DTRACE(CACHE_EVENT)) {
      dumpCacheEvent(time, "::fill", "ap:ON_MISS m_tag_array->fill", mf);
    }    
    m_tag_array->fill(e->second.m_cache_index, time, mf);
  }    
  else if (m_config.m_alloc_policy == ON_FILL) {
    if (DTRACE(CACHE_EVENT)) {
      dumpCacheEvent(time, "::fill", "ap:ON_FILL m_tag_array->fill", mf);
    }     
    m_tag_array->fill(m_gpu, e->second.m_block_addr, time, mf, mf->is_write());
  } else {
    abort();
  }

  bool has_atomic = false;
  if (m_config.m_mshr_disable == 'T') {
    has_atomic = mf->isatomic();
    m_lfb.push_back(mf);
    if (!strcmp(m_config.m_cache_name,"L2")) {
      m_stats.inc_l2_sub_miss_served_cycles(
        mf->get_streamID(), mf->get_sub_partition(),
        time - mf->m_miss_serve_begin_time);
      m_stats.inc_l2_sub_misses(mf->get_streamID(), mf->get_sub_partition());
    } else if (!strcmp(m_config.m_cache_name,"L1D")) {
      m_stats.inc_l1d_miss_served_cycles(
        mf->get_streamID(), time - mf->m_miss_serve_begin_time);
      m_stats.inc_l1d_misses(mf->get_streamID());
    }
  } else {
    m_mshrs.mark_ready(m_is_l2 ? "L2" : m_is_l1d ? "L1D" : "other$", 
      e->second.m_block_addr, has_atomic, time);
    m_tag_array->set_recorded_in_mshr(e->second.m_cache_index, time);
    m_tag_array->m_total_records_in_mshr++;

    if (!strcmp(m_config.m_cache_name,"L2")) {
      m_stats.inc_l2_sub_miss_served_cycles(
        mf->get_streamID(), mf->get_sub_partition(),
        time - mf->m_miss_serve_begin_time);
      m_stats.inc_l2_sub_misses(mf->get_streamID(), mf->get_sub_partition());
    } else if (!strcmp(m_config.m_cache_name,"L1D")) {
      m_stats.inc_l1d_miss_served_cycles(
        mf->get_streamID(), time - mf->m_miss_serve_begin_time);
      m_stats.inc_l1d_misses(mf->get_streamID());
    }

    std::string str_cache_name = m_config.get_cache_name();
    if (!strcmp(m_config.get_cache_name(), "L2") && mf) {
      str_cache_name += "_sub[";
      str_cache_name += std::to_string(mf->get_sub_partition());
      str_cache_name += "]";
    }    
    if (DTRACE(MSHR_RECORD_MONITOR)) {
      fprintf(Trace::out, "%llu %s m_total_records_in_mshr = %u\n", 
        time, str_cache_name.c_str(), m_tag_array->m_total_records_in_mshr);
    }
  }

  if (has_atomic) {
    assert(m_config.m_alloc_policy == ON_MISS);
    cache_block_t *block = m_tag_array->get_block(e->second.m_cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    // mark line as dirty for atomic operation
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    block->set_byte_mask(mf);
  }
  m_extra_mf_fields.erase(mf);
  m_bandwidth_management.use_fill_port(mf);
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
  unsigned long long time, const char* stage, const char* event, mem_fetch *mf) {

  const char* cache_name = m_config.get_cache_name();
  std::string l2_prefix = "";
  if (strcmp(cache_name, "L2") == 0) {
    l2_prefix = "SG";
    l2_prefix += std::to_string(mf->get_sub_partition());
    l2_prefix += " ";      
  }
  fprintf(Trace::out, "%llu %s %sstage(%s) cache_event(%s) "
    "mf:{TPC:%u SM:%u WARP:%u req_uid:%u addr:%#llx acc_type:%s pos:%s}\n", 
    time, cache_name, l2_prefix.c_str(), stage, event,
    mf->get_tpc(), mf->get_sid(), mf->get_wid(), mf->get_request_uid(), mf->get_addr(), // mf info
    mem_access_type_str(mem_access_type(mf->get_access_type())),
    mf->mem_fetch_status_str(mf->get_status())
  );
}

void baseline_cache::dumpMSHREvent(
  unsigned long long time, mem_fetch *mf, new_addr_type mshr_addr, bool is_new_entry) {

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

void baseline_cache::dumpMissQueue(unsigned long long time, const char* stage, const char* event, mem_fetch *mf) {

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
void baseline_cache::send_read_request(new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned long long time, bool &do_miss,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  bool wb = false;
  evicted_block_info e;
  send_read_request(block_addr, cache_index, mf, time, do_miss, wb, e,
                    events, read_only, wa);
}

/// Read miss handler. Check MSHR hit or MSHR available
void baseline_cache::send_read_request(new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned long long time, bool &do_miss, bool &wb,
                                       evicted_block_info &evicted,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {

  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());
  [[maybe_unused]] const char* cache_type = m_is_l2 ? "L2" : m_is_l1d ? "L1D" : "OTHER$";  
  if (m_config.m_mshr_disable == 'T') {
    // No MSHR: only gate on miss_queue capacity, no merge or ready tracking.
    if (m_miss_queue.size() < m_config.m_miss_queue_size) {
      if (read_only) {
        m_tag_array->access(m_gpu, block_addr, time, cache_index, mf);
      } else {
        m_tag_array->access(m_gpu, block_addr, time, cache_index, wb, evicted, mf);
      }

      m_extra_mf_fields[mf] = extra_mf_fields(
          mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
      mf->set_data_size(m_config.get_atom_sz());
      mf->set_addr(mshr_addr);
      m_miss_queue.push_back(mf);
      mf->set_status(m_miss_queue_status, time);

      if (DTRACE(CACHE_EVENT) || DTRACE(MISS_QUEUE_EVENT)) {
        dumpMissQueue(time, "RD-MISS-NO-MSHR", "m_miss_queue.push_back ", mf);
      }

      if (!wa) {
        events.push_back(cache_event(READ_REQUEST_SENT));
        if (DTRACE(CACHE_EVENT)) {
          dumpCacheEvent(time, "RD-MISS-NO-MSHR", "READ_REQUEST_SENT", mf);
        }
      }
      do_miss = true;
    } else {
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                            mf->get_streamID(), miss_queue_full_driver::RD_MISS);
    }
  } else {
    bool mshr_hit   = m_mshrs.probe(mshr_addr);
    bool mshr_avail = !m_mshrs.full(mshr_addr);
    if (mshr_hit && mshr_avail) {
      if (read_only) {
        m_tag_array->access(m_gpu, block_addr, time, cache_index, mf);
      } else {
        m_tag_array->access(m_gpu, block_addr, time, cache_index, wb, evicted, mf);
      }

      const size_t last_occupied_entries = m_mshrs.occupied_entries();

      if (DTRACE(DUMP_MSHR)) {
        fprintf(Trace::out, "%llu %s MSHR Hit! Before adding, MSHR is below:\n", 
          time, cache_type);
        m_mshrs.display(Trace::out, cache_type);      
      }

      [[maybe_unused]] bool is_l2 = !strcmp(m_config.get_cache_name(), "L2");
      bool is_new_mshr_entry = false;
      // m_tag_array->set_recorded_in_mshr(block_addr);
      m_mshrs.add(mshr_addr, mf, is_new_mshr_entry, cache_type); // orig GPGPU-SIM logic

      if (m_is_l2) {
        m_stats.inc_l2_mshr_slots_fills(mf->get_streamID(), mf->get_sub_partition());
      }
      if (DTRACE(RECORDED_IN_MSHR)) {
        fprintf(Trace::out, "%llu %s block_addr %#llx {tag %#llx set_index %#x} "
          "is recorded in MSHR (mshr hit) occupied_slots[mshr_addr:%#llx] = %u. "
          "mf:{ TPC:%u SM:%u WARP:%u req_uid:%u}\n",
          time, cache_type, block_addr, 
          m_config.tag(block_addr), m_config.set_index(block_addr),
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
        m_tag_array->access(m_gpu, block_addr, time, cache_index, mf);
      } else {
        m_tag_array->access(m_gpu, block_addr, time, cache_index, wb, evicted, mf);
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
      // m_tag_array->set_recorded_in_mshr(block_addr);
      m_mshrs.add(mshr_addr, mf, is_new_mshr_entry, cache_type); // orig GPGPU-SIM logic      
      if (m_is_l2) {
        m_stats.inc_l2_mshr_slots_fills(mf->get_streamID(), mf->get_sub_partition());
      }
      if (DTRACE(RECORDED_IN_MSHR)) {
        fprintf(Trace::out, "%llu %s block_addr %#llx {tag %#llx set_index %#x} "
          "is recorded in MSHR (mshr miss) occupied_slots[mshr_addr:%#llx] = %u. "
          "mf:{ TPC:%u SM:%u WARP:%u req_uid:%u}\n",
          time, cache_type, block_addr, 
          m_config.tag(block_addr), m_config.set_index(block_addr),
          mshr_addr, m_mshrs.occupied_slots(mshr_addr),
          mf->get_tpc(), mf->get_sid(), mf->get_wid(), mf->get_request_uid()
        );         
      }

      if (DTRACE(CACHE_EVENT) || DTRACE(MSHR_EVENT)) {
        dumpMSHREvent(time, mf, mshr_addr, is_new_mshr_entry);
      }

      m_stats.inc_mshr_stats(mf->get_streamID(), mf->get_sid(), mf->get_wid());

      m_extra_mf_fields[mf] = extra_mf_fields(
          mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
      mf->set_data_size(m_config.get_atom_sz());
      mf->set_addr(mshr_addr);
      m_miss_queue.push_back(mf);
      mf->set_status(m_miss_queue_status, time);

      if (DTRACE(CACHE_EVENT) || DTRACE(MISS_QUEUE_EVENT)) {
        dumpMissQueue(time, "RD-MISS-MSHR-MISS-AND-AVAIL", "m_miss_queue.push_back ", mf);
      }

      if (!wa) {
        events.push_back(cache_event(READ_REQUEST_SENT));

        if (DTRACE(CACHE_EVENT)) {
          dumpCacheEvent(time, "RD-MISS-THEN-CHECK-MSHR", "READ_REQUEST_SENT", mf);
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
void data_cache::send_write_request(mem_fetch *mf, cache_event request,
                                    unsigned time,
                                    std::list<cache_event> &events) {
  events.push_back(request);
  m_miss_queue.push_back(mf);

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
      [[maybe_unused]] bool is_l2 = !strcmp(m_config.get_cache_name(), "L2")? true : false;
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
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(m_gpu, block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  update_m_readable(mf, cache_index);

  return HIT;
}

/// Write-through hit: Directly send request to lower level memory
cache_request_status data_cache::wr_hit_wt(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0)) {
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
  m_tag_array->access(m_gpu, block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  update_m_readable(mf, cache_index);

  // generate a write-through
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  if (DTRACE(CACHE_EVENT)) {
    dumpCacheEvent(time, "WT-HIT-THEN-SEND-TO-LOWER-MEM", "WRITE_REQUEST_SENT", mf);
  }

  return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate
/// corresponding block
cache_request_status data_cache::wr_hit_we(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0)) {
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
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  if (DTRACE(CACHE_EVENT)) {
    dumpCacheEvent(time, "WR-EVICT-HIT-THEN-SEND-TO-LOWER-MEM-AND-INV-CURR-CACHE", 
      "WRITE_REQUEST_SENT", mf);
  }

  // Invalidate block
  block->set_status(INVALID, mf->get_access_sector_mask());

  return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
enum cache_request_status data_cache::wr_hit_global_we_local_wb(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
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
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  // Write allocate, maximum 3 requests (write miss, read request, write back
  // request) Conservatively ensure the worst-case request can be handled this
  // cycle
  bool mshr_hit   = m_mshrs.probe(mshr_addr);
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  if (miss_queue_full(2) ||
      (!(mshr_hit && mshr_avail) &&
       !(!mshr_hit && mshr_avail &&
         (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
    // check what is the exactly the failure reason
    if (miss_queue_full(2)) {
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

  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  // Tries to send write allocate request, returns true on success and false on
  // failure
  // if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
  //    return RESERVATION_FAIL;
  if (DTRACE(CACHE_EVENT)) {
    dumpCacheEvent(time, "WT-ALLOC-MISS-THEN-SEND-WR-TO-LOWER-MEM", "WRITE_REQUEST_SENT", mf);
  }  

  const mem_access_t *ma =
      new mem_access_t(m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
                       false,  // Now performing a read
                       mf->get_access_warp_mask(), mf->get_access_byte_mask(),
                       mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

  mem_fetch *n_mf = new mem_fetch(
      *ma, NULL, mf->get_streamID(), mf->get_ctrl_size(), mf->get_wid(),
      mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
      m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
  // 2/13 debug
  if (DTRACE(DEBUG_SINGLE_MF)) {
    fprintf(Trace::out, "%llu wr_miss_wa_naive "
      "new mem_fetch addr = %#llx sid:%u warp_id:%u\n",
      m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
      mf->get_addr(), mf->get_sid(), mf->get_wid());
  }

  bool do_miss = false;
  bool wb = false;
  evicted_block_info evicted;

  // Send read request resulting from write miss
  send_read_request(block_addr, cache_index, n_mf, time, do_miss, wb,
                    evicted, events, false, true);              

  events.push_back(cache_event(WRITE_ALLOCATE_SENT));

  if (DTRACE(CACHE_EVENT)) {
    dumpCacheEvent(time, "WT-ALLOC-MISS-PHASE2-SENT-RD", "WRITE_ALLOCATE_SENT", mf);
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
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);

      if (DTRACE(CACHE_EVENT)) {
        dumpCacheEvent(time, "WR-ALLOC-MISS-NO-MSHR-PENDING", "WRITE_BACK_REQUEST_SENT", mf);
      }

    }
    return status;
  }

  return RESERVATION_FAIL;
}

enum cache_request_status data_cache::wr_miss_wa_fetch_on_write(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    // if the request writes to the whole cache line/sector, then, write and set
    // cache line Modified. and no need to send read request to memory or
    // reserve mshr

    if (miss_queue_full(0)) {
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
        m_tag_array->access(m_gpu, block_addr, time, cache_index, wb, evicted, mf);
    assert(status != HIT);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED, mf->get_access_sector_mask());
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
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);

        if (DTRACE(CACHE_EVENT)) {
          dumpCacheEvent(time, "TO-HANDLE-THE-EVICTED-LINE", "WRITE_BACK_REQUEST_SENT", mf);
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
      if (miss_queue_full(1)) {
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
        m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, NULL, mf);
    // 2/13 debug
    if (DTRACE(DEBUG_SINGLE_MF)) {
      fprintf(Trace::out, "%llu wr_miss_wa_fetch_on_write "
        "new mem_fetch addr = %#llx sid:%u warp_id:%u\n",
        m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
        mf->get_addr(), mf->get_sid(), mf->get_wid());
    }

    new_addr_type block_addr = m_config.block_addr(addr);
    bool do_miss = false;
    bool wb = false;
    evicted_block_info evicted;
    send_read_request(block_addr, cache_index, n_mf, time, do_miss, wb,
                      evicted, events, false, true);

    cache_block_t *block = m_tag_array->get_block(cache_index);
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    block->set_byte_mask_on_fill(true);

    events.push_back(cache_event(WRITE_ALLOCATE_SENT));

    if (DTRACE(CACHE_EVENT)) {
      dumpCacheEvent(time, "PREVENT-WR-RD-WR-IN-PENDING-MSHR", "WRITE_ALLOCATE_SENT", mf);
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
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);

        if (DTRACE(CACHE_EVENT)) {
          dumpCacheEvent(time, "xxx", "WRITE_BACK_REQUEST_SENT", mf);
        }                               
      }
      return MISS;
    }
    return RESERVATION_FAIL;
  }
}

enum cache_request_status data_cache::wr_miss_wa_lazy_fetch_on_read(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);

  // if the request writes to the whole cache line/sector, then, write and set
  // cache line Modified. and no need to send read request to memory or reserve
  // mshr

  if (miss_queue_full(0)) {
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
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
    if (DTRACE(CACHE_EVENT)) {
      dumpCacheEvent(time, "WR-MISS {wp: WRITE_THROUGH wap: LAZY_FETCH_ON_READ}", 
        "WRITE_REQUEST_SENT", mf);
    }    
  }

  bool wb = false;
  evicted_block_info evicted;

  cache_request_status req_status =
      m_tag_array->access(m_gpu, block_addr, time, cache_index, wb, evicted, mf);

  assert(req_status != HIT);
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  cache_block_state prev_blk_state = block->get_status(mf->get_access_sector_mask()); // for tracing

  unsigned sidx = block->set_status(MODIFIED, mf->get_access_sector_mask());

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
            (unsigned long long)(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle),
            m_gpu->gpgpu_ctx->func_sim->ptx_get_insn_str(mf->get_inst().pc).c_str(),
            m_is_l1d ? "L1D" : m_is_l2 ? "L2C" : "xx$",
            mf_request_type_str(mf->get_type()),
            cache_request_status_str(req_status),
            (unsigned long long)block_addr, sidx,
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
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);

      if (DTRACE(CACHE_EVENT)) {
        dumpCacheEvent(time, "WR-MISS {wp: WRITE_BACK wap: LAZY_FETCH_ON_READ}", 
          "WRITE_BACK_REQUEST_SENT", mf);
      }                            
    }

    return MISS;
  }
  if (DTRACE(CACHELINE_STATUS)) {
    fprintf(Trace::out,
      "%llu: RESERVATION_FAIL on block_addr=%#llx sector_mask=%s "
      "byte_mask=%s\n",
      (unsigned long long)(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle),
      (unsigned long long)block_addr,
      mf->get_access_sector_mask().to_string().c_str(),
      mf->get_access_byte_mask().to_string().c_str());
  }
  return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
enum cache_request_status data_cache::wr_miss_no_wa(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  if (miss_queue_full(0)) {
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
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  if (DTRACE(CACHE_EVENT)) {
    dumpCacheEvent(time, "WR-MISS {wap: NO_WRITE_ALLOC} DIRECTLY-SEND-TO-LOWER-MEM", 
      "WRITE_REQUEST_SENT", mf);
  }

  return MISS;
}

/****** Read hit functions (Set by config file) ******/

/// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
enum cache_request_status data_cache::rd_hit_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, 
    unsigned long long time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(m_gpu, block_addr, time, cache_index, mf);
  // Atomics treated as global read/write requests - Perform read, mark line as
  // MODIFIED
  if (mf->isatomic()) {
    assert(mf->get_access_type() == GLOBAL_ACC_R);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    block->set_byte_mask(mf);
  }
  return HIT;
}

/****** Read miss functions (Set by config file) ******/

/// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
enum cache_request_status data_cache::rd_miss_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, 
    unsigned long long time,
    std::list<cache_event> &events, enum cache_request_status status) {

  new_addr_type block_addr = m_config.block_addr(addr);  
  
  if (status == cache_request_status::MISS || \
      status == cache_request_status::SECTOR_MISS) {

    mf->set_status(IN_PARTITION_L2, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    if (DTRACE(CACHE_EVENT)) {
      dumpCacheEvent(time, "::rd_miss_base entered", 
        cache_request_status_str(status), mf);
    }

    if (DTRACE(CACHE_MISS)) {
      dump_cache_access_info("::rd_miss_base ", addr, mf, time, status);
    }

    m_l1d_rd_miss_addresses.push_back(addr);

    if (DTRACE(SIMPLE_PREFETCH)) {
      fprintf(Trace::out, "%llu added new addr: %#llx "
        "into m_l1d_rd_miss_addresses (size %zu->%zu)\n", 
        time, addr, 
        m_l1d_rd_miss_addresses.size() - 1, m_l1d_rd_miss_addresses.size());
      
      fprintf(Trace::out, "m_l1d_rd_miss_addresses holds:\n");
      for (size_t i = 0; i < m_l1d_rd_miss_addresses.size(); i++)
      {
        fprintf(Trace::out, "m_l1d_rd_miss_addresses[%u] = %#llx\n", 
          (unsigned)i, m_l1d_rd_miss_addresses[i]);
      } 
    }
  }

  if (miss_queue_full(1)) {
    // cannot handle request this cycle
    // (might need to generate two requests)
    if (DTRACE(MISS_QUEUE_FULL_DRIVER)) {      
      if (m_is_l1d) {        
        fprintf(Trace::out, "L1D RD_MISS called [%s][MISS_QUEUE_FULL]\n",
          mem_access_type_str(mf->get_access_type())); 
        fprintf(Trace::out, "Total_core_cache_fail_stats_breakdown[%s][MISS_QUEUE_FULL]++\n",
        mem_access_type_str(mf->get_access_type()));
      }  
    }
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID(), miss_queue_full_driver::RD_MISS);
    return RESERVATION_FAIL;
  }

  bool do_miss = false;
  bool wb = false;
  evicted_block_info evicted;
  send_read_request(block_addr, cache_index, mf, time, do_miss, wb,
                    evicted, events, false, false);

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
      send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}

/// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
enum cache_request_status read_only_cache::access(
  new_addr_type addr, mem_fetch *mf, unsigned long long time,
  std::list<cache_event> &events) {
  assert(mf->get_data_size() <= m_config.get_atom_sz());
  assert(m_config.m_write_policy == READ_ONLY);
  assert(!mf->get_is_write());
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;

  bool got_warp_interfere_info = false;
  WARP_INTERFERE_RECORD warp_interfere_record((unsigned )- 1, (unsigned) - 1);

  enum cache_request_status status = m_tag_array->probe(
    "read_only_cache::access", m_gpu, block_addr, cache_index, mf, mf->is_write(), 
    time, got_warp_interfere_info, warp_interfere_record);
  
    enum cache_request_status cache_status = RESERVATION_FAIL;

  if (status == HIT) {
    cache_status = m_tag_array->access(m_gpu, block_addr, time, cache_index, mf); // update LRU state
  } else if (status != RESERVATION_FAIL) {
    if (!miss_queue_full(0)) {
      bool do_miss = false;
      send_read_request(block_addr, cache_index, mf, time, do_miss,
                        events, true, false);
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
    bool wr, enum cache_request_status probe_status, new_addr_type addr,
    unsigned cache_index, mem_fetch *mf, unsigned long long time,
    std::list<cache_event> &events) {
  // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
  // data_cache constructor to reflect the corresponding cache configuration
  // options. Function pointers were used to avoid many long conditional
  // branches resulting from many cache configuration options.
  cache_request_status access_status = probe_status;
  if (wr) {  // Write
    if (probe_status == HIT) {
      access_status = (this->*m_wr_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if ((probe_status != RESERVATION_FAIL) ||
               (probe_status == RESERVATION_FAIL &&
                m_config.m_write_alloc_policy == NO_WRITE_ALLOCATE)) {

      if (DTRACE(CACHE_EVENT)) {
        dumpCacheEvent(time, "tag_probe", "m_wr_miss", mf);
      }

      access_status = (this->*m_wr_miss)(addr, cache_index, mf, time, events, probe_status);
      if (access_status == cache_request_status::MISS) {
        mf->set_miss_serve_begin_time(time);
      }      
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                             mf->get_streamID(), line_alloc_fail_driver::LINE_ALLOC_FAIL__WR_PROBE_MISS);
    }
  } else {  // Read
    if (probe_status == HIT) {
      access_status = (this->*m_rd_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if (probe_status != RESERVATION_FAIL) {
      if (DTRACE(CACHE_EVENT)) {
        dumpCacheEvent(time, "tag_probe", "m_rd_miss", mf);
      }
      access_status = (this->*m_rd_miss)(addr, cache_index, mf, time, events, probe_status);
      if (access_status == cache_request_status::MISS) {
        mf->set_miss_serve_begin_time(time);
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
                                             unsigned long long time,
                                             std::list<cache_event> &events) {
  
  assert(mf->get_data_size() <= m_config.get_atom_sz());
  bool wr = mf->get_is_write();
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned) - 1;

  bool got_warp_interfere_info = false;
  WARP_INTERFERE_RECORD warp_interfere_record((unsigned )- 1, (unsigned) - 1);

  enum cache_request_status probe_status = m_tag_array->probe(
    "data_cache::access", m_gpu,
    block_addr, cache_index, mf, mf->is_write(), time,
    got_warp_interfere_info, warp_interfere_record,
    true /* probe_mode */);

  if (got_warp_interfere_info) {
    assert(warp_interfere_record.curr_warp_id == mf->get_wid());
    unsigned interfered  = warp_interfere_record.last_warp_id;
    unsigned interfering = warp_interfere_record.curr_warp_id;
    m_gpu->get_shader_stats()->warp_interfere[mf->get_sid()][interfered][interfering]++;
    if (DTRACE(INC_WARP_INTERFERE)) {
      fprintf(Trace::out, "%llu Increased warp_interfere[sid:%u][interfered:%u][interfering:%u] = %u\n",
        time, mf->get_sid(), interfered, interfering, 
        m_gpu->get_shader_stats()->warp_interfere[mf->get_sid()][interfered][interfering]
      );
    }
  }
 
  const unsigned sid = mf->get_sid();
  const std::vector<std::set<new_addr_type>>& unique_lines = m_tag_array->get_unique_lines();
  m_gpu->get_shader_stats()->m_unique_cachelines[sid] = unique_lines[sid].size();

  enum cache_request_status access_status =
      process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(probe_status, access_status),
                    mf->get_streamID());
  m_stats.update_evict_stats(mf->get_streamID(), mf->get_victim_avg_evict_interval());

  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(probe_status, access_status),
                       mf->get_streamID());

  if (DTRACE(CACHE_ACCESS)) {
    // For l2 access, time - "gpgpu-sim cycle counters" == m_memcpy_cycle_offset,
    // indicating extra cycles on cudaMemCpy operations.
    // assert((m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle) == time);
    dump_cache_access_info("::access ", addr, mf, time, access_status);
  }

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
                                           unsigned long long time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status l2_cache::access(
  new_addr_type addr, mem_fetch *mf, unsigned long long time,
  std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

/// Access function for tex_cache
/// return values: RESERVATION_FAIL if request could not be accepted
/// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
/// since unlike a normal CPU cache, a "HIT" in texture cache does not
/// mean the data is ready (still need to get through fragment fifo)
enum cache_request_status tex_cache::access(new_addr_type addr, mem_fetch *mf,
                                            unsigned long long time,
                                            std::list<cache_event> &events) {
  if (m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full())
    return RESERVATION_FAIL;

  assert(mf->get_data_size() <= m_config.get_line_sz());

  // at this point, we will accept the request : access tags and immediately
  // allocate line
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;
  enum cache_request_status status =
      m_tags.access(nullptr, block_addr, time, cache_index, mf);
  enum cache_request_status cache_status = RESERVATION_FAIL;
  assert(status != RESERVATION_FAIL);
  assert(status != HIT_RESERVED);  // as far as tags are concerned: HIT or MISS
  m_fragment_fifo.push(fragment_entry(mf, cache_index, status == MISS, mf->get_data_size()));
  if (status == MISS) {
    // we need to send a memory request...
    unsigned rob_index = m_rob.push(rob_entry(cache_index, mf, block_addr));
    m_extra_mf_fields[mf] = extra_mf_fields(rob_index, m_config);
    mf->set_data_size(m_config.get_line_sz());
    m_tags.fill(cache_index, time, mf);  // mark block as valid
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