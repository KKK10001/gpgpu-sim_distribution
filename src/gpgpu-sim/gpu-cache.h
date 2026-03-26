// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington, Vijay Kandiah,
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

#ifndef GPU_CACHE_H
#define GPU_CACHE_H

#include <stdio.h>
#include <stdlib.h>
#include <unordered_set>
#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"
#include "gpu-misc.h"
#include "mem_fetch.h"
#include "shader_trace.h"

#include <iostream>
#include "addrdec.h"

#define MAX_DEFAULT_CACHE_SIZE_MULTIBLIER 4

typedef unsigned u32;
typedef unsigned long long u64;

enum cache_block_state { 
  INVALID = 0, RESERVED, VALID, MODIFIED,
  NUM_CACHE_BLOCK_STATES
 };

enum cache_request_status {
  HIT = 0,
  HIT_RESERVED,
  MISS,
  RESERVATION_FAIL,
  SECTOR_MISS,
  MSHR_HIT, // only stats related
  BYPASS_ACTIVATED,
  BYPASS_DEACTIVATED,
  NUM_CACHE_REQUEST_STATUS
};

enum mshr_config_t {
  TEX_FIFO,         // Tex cache
  ASSOC,            // normal cache
  SECTOR_TEX_FIFO,  // Tex cache sends requests to high-level sector cache
  SECTOR_ASSOC,      // normal cache sends requests to high-level sector cache
  NUM_MSHR_CONFIGS
};

enum replacement_policy_t { 
  LRU = 0, 
  FIFO, 
  SRRIP,
  NUM_REPLACEMENT_POLICY
};

enum srrip_update_policy_t {
  HP = 0, // Hit Policy
  FP,     // Frequency Policy
  NUM_SRRIP_UPDATE_POLICY
};

enum write_policy_t {
  READ_ONLY,
  WRITE_BACK,
  WRITE_THROUGH,
  WRITE_EVICT,
  LOCAL_WB_GLOBAL_WE,
  NUM_WRITE_POLICIES
};

enum allocation_policy_t { ON_MISS, ON_FILL, STREAMING };

enum write_allocate_policy_t {
  NO_WRITE_ALLOCATE,
  WRITE_ALLOCATE,
  FETCH_ON_WRITE,
  LAZY_FETCH_ON_READ,
  NUM_WRITE_ALLOCATE_POLICIES
};

enum cache_reservation_fail_reason {
  LINE_ALLOC_FAIL = 0,  // done. all line are reserved
  MSHR_ENTRY_FAIL,      // done.
  MISS_QUEUE_FULL,      // done. MISS queue (i.e. interconnect or DRAM) is full
  MSHR_MERGE_ENTRY_FAIL,// done.
  MSHR_RW_PENDING,
  NUM_CACHE_RESERVATION_FAIL_STATUS
};

enum line_alloc_fail_driver {
  LINE_ALLOC_FAIL__RD_ONLY_MISS = 0,
  LINE_ALLOC_FAIL__RD_PROBE_MISS,
  LINE_ALLOC_FAIL__WR_PROBE_MISS,
  NUM_LINE_ALLOC_FAIL_DRIVER
};

enum mshr_entry_fail_driver {
  MSHR_ENTRY_FAIL__RD_MISS = 0,
  MSHR_ENTRY_FAIL__WR_ALLOC_MISS,
  MSHR_ENTRY_FAIL__WR_ALLOC_MISS_FETCH_ON_WR,
  NUM_MSHR_ENTRY_FAIL_DRIVER
};

enum miss_queue_full_driver {
  WR_THROUGH_HIT = 0,
  WR_EVICT_HIT,
  WR_ALLOC_MISS, // data_cache::wr_miss_wa_naive
  WR_ALLOC_MISS_FETCH_ON_WR_WHOLE_LINE,
  WR_ALLOC_MISS_FETCH_ON_WR_PARTIAL_LINE,
  WR_ALLOC_MISS_LAZY_FETCH_ON_RD,
  WR_MISS_NO_WR_ALLOC,
  RD_MISS,
  RD_ONLY_MISS,
  NUM_MISS_QUEUE_FULL_DRIVER
};

enum mshr_merge_entry_fail_driver {
  MSHR_MERGE_ENTRY_FAIL__RD_MISS = 0,
  MSHR_MERGE_ENTRY_FAIL__WR_ALLOC_MISS,
  MSHR_MERGE_ENTRY_FAIL__WR_ALLOC_MISS_FETCH_ON_WR,
  NUM_MSHR_MERGE_ENTRY_FAIL_DRIVER
};

enum cache_event_type {
  WRITE_BACK_REQUEST_SENT,
  READ_REQUEST_SENT,
  WRITE_REQUEST_SENT,
  WRITE_ALLOCATE_SENT
};

enum cache_gpu_level {
  L1_GPU_CACHE = 0,
  L2_GPU_CACHE,
  OTHER_GPU_CACHE,
  NUM_CACHE_GPU_LEVELS
};

struct evicted_block_info {
  new_addr_type m_block_addr;
  unsigned m_modified_size;
  mem_access_byte_mask_t m_byte_mask;
  mem_access_sector_mask_t m_sector_mask;
  evicted_block_info() {
    m_block_addr = 0;
    m_modified_size = 0;
    m_byte_mask.reset();
    m_sector_mask.reset();
  }
  void set_info(new_addr_type block_addr, unsigned modified_size) {
    m_block_addr = block_addr;
    m_modified_size = modified_size;
  }
  void set_info(new_addr_type block_addr, unsigned modified_size,
                mem_access_byte_mask_t byte_mask,
                mem_access_sector_mask_t sector_mask) {
    m_block_addr = block_addr;
    m_modified_size = modified_size;
    m_byte_mask = byte_mask;
    m_sector_mask = sector_mask;
  }
};

struct cache_event {
  enum cache_event_type m_cache_event_type;
  evicted_block_info m_evicted_block;  // if it was write_back event, fill the
                                       // the evicted block info

  cache_event(enum cache_event_type m_cache_event) {
    m_cache_event_type = m_cache_event;
  }

  cache_event(enum cache_event_type cache_event,
              evicted_block_info evicted_block) {
    m_cache_event_type = cache_event;
    m_evicted_block = evicted_block;
  }
};

const char *cache_request_status_str(enum cache_request_status status);
const char* mshr_config_t_str(enum mshr_config_t mshr_config);
const char* write_policy_str(enum write_policy_t wp);
const char* write_allocate_policy_str(enum write_allocate_policy_t wap);

struct cache_block_t {
  cache_block_t() {
    m_tag = 0;
    m_block_addr = 0;
    m_owner = (unsigned) - 1;
    m_n_acc_after_bypass = 0;    
    m_n_reused = 0;
    m_n_rereferenced = 0;
    m_was_recorded_in_mshr = false;
  }

  virtual void allocate(new_addr_type tag, new_addr_type block_addr,
                        unsigned long long time,
                        mem_access_sector_mask_t sector_mask) = 0;
  virtual void fill(
    unsigned long long time, 
    mem_access_sector_mask_t sector_mask, mem_access_byte_mask_t byte_mask, 
    mem_fetch *mf = nullptr) = 0;

  virtual bool is_invalid_line() = 0;
  virtual bool is_valid_line() = 0;
  virtual bool is_reserved_line() = 0;
  virtual bool is_modified_line() = 0;

  virtual void set_recorded_in_mshr() {
    m_was_recorded_in_mshr = true;
  }
  virtual unsigned rereferenced_times() { return m_n_rereferenced; }
  virtual bool was_recorded_in_mshr() { return m_was_recorded_in_mshr; }

  virtual enum cache_block_state get_status(
      mem_access_sector_mask_t sector_mask) = 0;
  virtual unsigned set_status(
    unsigned long long time, enum cache_block_state m_status,
    mem_access_sector_mask_t sector_mask, std::string caller = "") = 0;
  virtual unsigned set_status(
    enum cache_block_state m_status,
    mem_access_sector_mask_t sector_mask, std::string caller = "") = 0;                          
  virtual void set_byte_mask(mem_fetch *mf) = 0;
  virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) = 0;
  virtual mem_access_byte_mask_t get_dirty_byte_mask() = 0;
  virtual mem_access_sector_mask_t get_dirty_sector_mask() = 0;

  virtual void set_rrpv(unsigned rrpv) = 0;
  virtual unsigned get_rrpv() = 0;
  virtual void set_max_rrpv(unsigned rrpv) = 0;
  virtual unsigned get_max_rrpv() = 0;

  virtual void set_cache_name(char* cache_name) = 0;
  virtual char* get_cache_name() = 0;  
  virtual void inc_rrpv() = 0;
  virtual void dec_rrpv() = 0;
  virtual void set_last_access_time(unsigned long long time,
                                    mem_access_sector_mask_t sector_mask) = 0;
  virtual unsigned long long get_reref_gap() = 0;
  virtual unsigned long long get_last_access_time() = 0;
  virtual void set_last_fill_time(unsigned long long time) = 0;
  virtual unsigned long long get_last_fill_time() = 0;
  virtual unsigned long long get_last_evict_time() = 0;
  virtual unsigned long long get_evict_gap() = 0;
  virtual void set_last_warp_id(unsigned warp_id) = 0;
  virtual unsigned get_last_warp_id() = 0;
  virtual void set_last_core_id(unsigned core_id) = 0;
  virtual unsigned get_last_core_id() = 0;

  virtual void inc_total_hits() = 0;
  virtual void inc_total_evictions() = 0;
  virtual unsigned long long get_total_accesses() = 0;
  virtual unsigned long long get_total_evictions() = 0;
  virtual unsigned get_total_hits() = 0;

  virtual unsigned long long get_alloc_time() = 0;
  virtual void set_ignore_on_fill(bool m_ignore,
                                  mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_modified_on_fill(bool m_modified,
                                    mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_readable_on_fill(bool readable,
                                    mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_byte_mask_on_fill(bool m_modified) = 0;
  virtual unsigned get_modified_size() = 0;
  virtual void set_m_readable(bool readable,
                              mem_access_sector_mask_t sector_mask) = 0;
  virtual bool is_readable(mem_access_sector_mask_t sector_mask) = 0;
  virtual void print_status() = 0;
  virtual void print_readable() = 0;
  virtual std::string get_sector_status(unsigned sidx) = 0;
  virtual ~cache_block_t() {}

  new_addr_type m_tag;
  new_addr_type m_block_addr;
  u32 m_owner; // warp_id
  u32 m_n_acc_after_bypass;
  u32 m_n_reused;
  u32 m_n_rereferenced;
  bool m_was_recorded_in_mshr; // "was" means that the line might be invalid now
};

struct line_cache_block : public cache_block_t {
  line_cache_block() {
    m_alloc_time = 0;
    m_last_access_time = 0;

    m_total_evictions     = 0;
    m_total_accesses      = 0;

    m_total_hits         = 0;

    m_fill_time = 0;
    m_last_fill_time = 0;
    m_last_evict_time = 0;
    m_last_warp_id = (unsigned) - 1;
    m_last_core_id = (unsigned) - 1;
    m_rrpv     = get_max_rrpv();
    m_max_rrpv = get_max_rrpv();
    m_status = INVALID;
    m_ignore_on_fill_status = false;
    m_set_modified_on_fill = false;
    m_set_readable_on_fill = false;
    m_readable = true;
  }
  void allocate(new_addr_type tag, new_addr_type block_addr, unsigned long long time,
                mem_access_sector_mask_t sector_mask) {
    m_tag = tag;
    m_block_addr = block_addr;
    m_alloc_time = time;
    m_last_access_time = time;
    m_fill_time = 0;
    m_rrpv = (get_max_rrpv() >> 1) + 1;
    m_status = RESERVED;
    m_ignore_on_fill_status = false;
    m_set_modified_on_fill = false;
    m_set_readable_on_fill = false;
    m_set_byte_mask_on_fill = false;
  }

  virtual void fill(
    unsigned long long time, 
    mem_access_sector_mask_t sector_mask, mem_access_byte_mask_t byte_mask,
    mem_fetch *mf = nullptr) {

    m_status = m_set_modified_on_fill ? MODIFIED : VALID;
    m_last_warp_id = mf->get_wid();
    m_last_core_id = mf->get_sid();

    if (m_set_readable_on_fill) {
      m_readable = true;
    }
    if (m_set_byte_mask_on_fill) {
      set_byte_mask(byte_mask);
    }

    m_fill_time = time;
    m_rrpv = (get_max_rrpv() >> 1) + 1;
  }
  virtual bool is_invalid_line() { return m_status == INVALID; }
  virtual bool is_valid_line() { return m_status == VALID; }
  virtual bool is_reserved_line() { return m_status == RESERVED; }
  virtual bool is_modified_line() { return m_status == MODIFIED; }

  virtual enum cache_block_state get_status(
      mem_access_sector_mask_t sector_mask) {
    return m_status;
  }

  virtual std::string get_sector_status(unsigned sidx) {
    return "xx -> impl in derived class";
  }

  virtual unsigned set_status(
    unsigned long long time, enum cache_block_state status,
    mem_access_sector_mask_t sector_mask, std::string caller = "") {

    m_status = status;
    return 0;
  }  
  virtual unsigned set_status(
    enum cache_block_state status,
    mem_access_sector_mask_t sector_mask, std::string caller = "") {

    m_status = status;
    return 0;
  }    
  virtual void set_byte_mask(mem_fetch *mf) {
    m_dirty_byte_mask = m_dirty_byte_mask | mf->get_access_byte_mask();
  }
  virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) {
    m_dirty_byte_mask = m_dirty_byte_mask | byte_mask;
  }
  virtual mem_access_byte_mask_t get_dirty_byte_mask() {
    return m_dirty_byte_mask;
  }
  virtual mem_access_sector_mask_t get_dirty_sector_mask() {
    mem_access_sector_mask_t sector_mask;
    if (m_status == MODIFIED) sector_mask.set();
    return sector_mask;
  }

  virtual void set_last_access_time(unsigned long long time,
                                    mem_access_sector_mask_t sector_mask) {
    m_last_access_time = time;
  }
  virtual unsigned long long get_reref_gap() {
    return m_reref_gap;
  }
  virtual unsigned long long get_last_access_time() {
    return m_last_access_time;
  }  
  virtual void set_last_fill_time(unsigned long long time) {
    m_last_fill_time = time;
  }
  virtual unsigned long long get_last_fill_time() {
    return m_last_fill_time;
  }
  virtual unsigned long long get_last_evict_time() {
    return m_last_evict_time;
  }

  virtual void set_last_warp_id(unsigned warp_id) {
    m_last_warp_id = warp_id;
  }
  virtual unsigned get_last_warp_id() {
    return m_last_warp_id;
  }
  virtual void set_last_core_id(unsigned core_id) {
    m_last_core_id = core_id;
  }
  virtual unsigned get_last_core_id() {
    return m_last_core_id;
  }  

  virtual void inc_total_hits() {
    m_total_hits++;
  }
  virtual void inc_total_evictions() {
    m_total_evictions++;
  }
  virtual unsigned long long get_total_accesses() {
    return m_total_accesses;
  }
  virtual unsigned long long get_total_evictions() {
    return m_total_evictions;
  }  

  virtual unsigned get_total_hits() {
    return m_total_hits;
  }

  virtual unsigned long long get_evict_gap() { return m_evict_gap; }

  virtual void set_rrpv(unsigned rrpv) { m_rrpv = rrpv; }
  virtual unsigned get_rrpv() { return m_rrpv; }
  virtual void set_max_rrpv(unsigned rrpv) { m_max_rrpv = rrpv; }
  virtual unsigned get_max_rrpv() { return m_max_rrpv; }
  virtual void set_cache_name(char* cache_name) { m_cache_name = cache_name; }
  virtual char* get_cache_name() { return m_cache_name; }
  virtual void inc_rrpv() { m_rrpv++; }
  virtual void dec_rrpv() { m_rrpv--; }

  virtual unsigned long long get_alloc_time() { return m_alloc_time; }
  virtual void set_ignore_on_fill(bool m_ignore,
                                  mem_access_sector_mask_t sector_mask) {
    m_ignore_on_fill_status = m_ignore;
  }
  virtual void set_modified_on_fill(bool m_modified,
                                    mem_access_sector_mask_t sector_mask) {
    m_set_modified_on_fill = m_modified;
  }
  virtual void set_readable_on_fill(bool readable,
                                    mem_access_sector_mask_t sector_mask) {
    m_set_readable_on_fill = readable;
  }
  virtual void set_byte_mask_on_fill(bool m_modified) {
    m_set_byte_mask_on_fill = m_modified;
  }
  virtual unsigned get_modified_size() {
    return SECTOR_CHUNK_SIZE * SECTOR_SIZE;  // i.e. cache line size
  }
  virtual void set_m_readable(bool readable,
                              mem_access_sector_mask_t sector_mask) {
    m_readable = readable;
  }
  virtual bool is_readable(mem_access_sector_mask_t sector_mask) {
    return m_readable;
  }
  virtual void print_status() {
    printf("m_block_addr is %#llx, status = %u\n", m_block_addr, m_status);
  }
  virtual void print_readable() {
    printf("m_block_addr is %#llx, readable = %u\n", m_block_addr, m_readable);
  }  

 private:
  unsigned long long m_alloc_time;
  unsigned long long m_reref_gap;
  unsigned long long m_last_access_time;
  unsigned long long m_last_fill_time;
  unsigned long long m_last_evict_time;
  unsigned m_last_warp_id;
  unsigned m_last_core_id;
 
  unsigned long long m_fill_time;

  unsigned m_total_accesses;  
  unsigned m_total_hits;
  unsigned m_total_evictions;
  unsigned long long m_evict_gap;
  unsigned m_rrpv;
  unsigned m_max_rrpv;
  char* m_cache_name;
  cache_block_state m_status;
  bool m_ignore_on_fill_status;
  bool m_set_modified_on_fill;
  bool m_set_readable_on_fill;
  bool m_set_byte_mask_on_fill;
  bool m_readable;
  mem_access_byte_mask_t m_dirty_byte_mask;
};

struct sector_cache_block : public cache_block_t {
  sector_cache_block() { init(); }

  void init() {
    for (unsigned i = 0; i < SECTOR_CHUNK_SIZE; ++i) {
      m_sector_alloc_time[i] = 0;
      m_sector_fill_time[i] = 0;
      m_last_sector_access_time[i] = 0;
      m_status[i] = INVALID;
      m_ignore_on_fill_status[i] = false;
      m_set_modified_on_fill[i] = false;
      m_set_readable_on_fill[i] = false;
      m_readable[i] = true;
    }
    m_line_alloc_time = 0;
    m_line_last_access_time = 0;
    m_line_last_fill_time   = 0;
    m_line_last_evict_time  = 0;
    m_last_warp_id          = (unsigned) - 1;
    m_last_core_id          = (unsigned) - 1;
    m_total_evictions       = 0;
    m_total_accesses        = 0;
         
    m_total_hits            = 0;
    m_rrpv     = get_max_rrpv();
    m_max_rrpv = get_max_rrpv();

    m_dirty_byte_mask.reset();
  }  

  virtual void allocate(new_addr_type tag, new_addr_type block_addr,
                        unsigned long long time, mem_access_sector_mask_t sector_mask) {
    allocate_line(tag, block_addr, time, sector_mask);
  }

  void allocate_line(new_addr_type tag, new_addr_type block_addr, unsigned long long time,
                     mem_access_sector_mask_t sector_mask) {
    // allocate a new line
    // assert(m_block_addr != 0 && m_block_addr != block_addr);
    init();
    m_tag = tag;
    m_block_addr = block_addr;

    unsigned sidx = get_sector_index(sector_mask);

    // set sector stats
    m_sector_alloc_time[sidx] = time;       // no-used var.
    m_last_sector_access_time[sidx] = time; // no-used var.
    m_sector_fill_time[sidx] = 0;           // no-used var.
    [[maybe_unused]] std::string prev_status = get_sector_status(sidx);
    m_status[sidx] = RESERVED;
    m_ignore_on_fill_status[sidx] = false;  // no-used var.
    m_set_modified_on_fill[sidx] = false;
    m_set_readable_on_fill[sidx] = false;
    m_set_byte_mask_on_fill = false;

    // set line stats
    m_line_alloc_time       = time;  // only set this for the first allocated sector
    m_line_last_access_time = time;
    m_rrpv = (get_max_rrpv() >> 1) + 1;

    if (DTRACE(LINE_STATUS_CHANGE)) {
      fprintf(Trace::out, "%llu allocate_line addr:%#llx "
        "m_status[%u] = {%s->%s}\n", 
        time, m_block_addr, sidx, prev_status.c_str(), get_sector_status(sidx).c_str());
    }    
  }

  void allocate_sector(unsigned long long time, mem_access_sector_mask_t sector_mask) {
    // allocate invalid sector of this allocated valid line
    assert(is_valid_line());
    unsigned sidx = get_sector_index(sector_mask);

    // set sector stats
    m_sector_alloc_time[sidx] = time;        // no-used var.
    m_last_sector_access_time[sidx] = time;  // no-used var.
    m_sector_fill_time[sidx] = 0;            // no-used var.
      // this should be the case only for fetch-on-write policy //TO DO
    if (m_status[sidx] == MODIFIED) {
      m_set_modified_on_fill[sidx] = true;
    } else {
      m_set_modified_on_fill[sidx] = false;
    }

    m_set_readable_on_fill[sidx] = false;

    [[maybe_unused]] std::string prev_status = get_sector_status(sidx);
    m_status[sidx] = RESERVED;
    m_ignore_on_fill_status[sidx] = false;
    m_readable[sidx] = true;

    // set line stats
    m_line_last_access_time = time;
    m_line_last_fill_time   = time;
    m_rrpv = (get_max_rrpv() >> 1) + 1;
    
    if (DTRACE(LINE_STATUS_CHANGE)) {
      if (m_status[sidx] == MODIFIED) {
        fprintf(Trace::out, "%llu allocate_sector addr:%#llx "
          "m_status[%u] == MODIFIED -> m_set_modified_on_fill[%u] = true. "
          "m_status[%u] = RESERVED\n", 
          time, m_block_addr /* has already been assigned in allocate_line */,
          sidx, sidx, sidx
        );
      } else {
        fprintf(Trace::out, "%llu allocate_sector addr:%#llx "
          "m_set_modified_on_fill[%u] = false. m_status[%u] = RESERVED\n", 
          time, m_block_addr /* has already been assigned in allocate_line */,
          sidx, sidx
        );
      }
    }  

    if (DTRACE(LINE_STATUS_CHANGE)) {
      fprintf(Trace::out, "%llu allocate_sector addr:%#llx "
        "m_status[%u] = {%s->%s}\n", 
        time, m_block_addr /* should has been assigned during allocate_line */, 
        sidx, prev_status.c_str(), get_sector_status(sidx).c_str() /* RESERVED */);
    }    
  }

  virtual void fill(
    unsigned long long time, 
    mem_access_sector_mask_t sector_mask, mem_access_byte_mask_t byte_mask,
    mem_fetch *mf = nullptr) {

    unsigned sidx  = get_sector_index(sector_mask);
    m_status[sidx] = m_set_modified_on_fill[sidx] ? MODIFIED : VALID;

    if (m_set_readable_on_fill[sidx]) {
      m_readable[sidx] = true;
      m_set_readable_on_fill[sidx] = false;
    }
    if (m_set_byte_mask_on_fill) {
      set_byte_mask(byte_mask);
    }

    m_sector_fill_time[sidx] = time;
    m_line_last_fill_time    = time;
    if (mf) {
      m_last_warp_id = mf->get_wid();
      m_last_core_id = mf->get_sid();
    }
    m_rrpv = (get_max_rrpv() >> 1) + 1; // for SRRIP 3'b111->3'b100
  }
  virtual bool is_invalid_line() {
    // all the sectors should be invalid 
    for (unsigned i = 0; i < SECTOR_CHUNK_SIZE; ++i) {
      if (m_status[i] != INVALID) return false;
    }
    return true;
  }
  virtual bool is_valid_line() { return !(is_invalid_line()); }
  virtual bool is_reserved_line() {
    // if any of the sector is reserved, then the line is reserved
    for (unsigned i = 0; i < SECTOR_CHUNK_SIZE; ++i) {
      if (m_status[i] == RESERVED) return true;
    }
    return false;
  }
  virtual bool is_modified_line() {
    // if any of the sector is modified, then the line is modified
    for (unsigned i = 0; i < SECTOR_CHUNK_SIZE; ++i) {
      if (m_status[i] == MODIFIED) {
        return true;
      }
    }
    return false;
  }

  virtual enum cache_block_state get_status(
      mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask);

    return m_status[sidx];
  }

  // Added return type for a chain-style calling
  virtual unsigned set_status(
    unsigned long long time, enum cache_block_state status,
    mem_access_sector_mask_t sector_mask, std::string caller = "") {    

    unsigned sidx = get_sector_index(sector_mask);
    [[maybe_unused]] std::string prev_status = get_sector_status(sidx);
    m_status[sidx] = status;

    if (DTRACE(LINE_STATUS_CHANGE)) {
      fprintf(Trace::out, "%llu %s set_status addr:%#llx "
        "m_status[%u] = {%s->%s}\n", 
        time, caller.c_str(), m_block_addr /* should has been assigned during allocate_line */, 
        sidx, prev_status.c_str(), get_sector_status(sidx).c_str());
    }

    return sidx;
  }
  virtual unsigned set_status(
    enum cache_block_state status,
    mem_access_sector_mask_t sector_mask, std::string caller = "") {    

    unsigned sidx = get_sector_index(sector_mask);
    [[maybe_unused]] std::string prev_status = get_sector_status(sidx);
    m_status[sidx] = status;

    if (DTRACE(LINE_STATUS_CHANGE)) {
      fprintf(Trace::out, "%s set_status addr:%#llx "
        "m_status[%u] = {%s->%s}\n", 
        caller.c_str(), m_block_addr /* should has been assigned during allocate_line */, 
        sidx, prev_status.c_str(), get_sector_status(sidx).c_str());
    }

    return sidx;
  }    

  virtual void set_byte_mask(mem_fetch *mf) {
    m_dirty_byte_mask = m_dirty_byte_mask | mf->get_access_byte_mask();
  }
  virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) {
    m_dirty_byte_mask = m_dirty_byte_mask | byte_mask;
  }
  virtual mem_access_byte_mask_t get_dirty_byte_mask() {
    return m_dirty_byte_mask;
  }
  virtual mem_access_sector_mask_t get_dirty_sector_mask() {
    mem_access_sector_mask_t sector_mask;
    for (unsigned i = 0; i < SECTOR_CHUNK_SIZE; i++) {
      if (m_status[i] == MODIFIED) sector_mask.set(i);
    }
    return sector_mask;
  }
  
  virtual void set_last_access_time(unsigned long long time,
                                    mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask);

    m_last_sector_access_time[sidx] = time;
    m_line_last_access_time = time;
  }
  virtual unsigned long long get_reref_gap() {
    return m_reref_gap;
  }  
  virtual unsigned long long get_last_access_time() {
    return m_line_last_access_time;
  }
  virtual void set_last_fill_time(unsigned long long time) {
    m_line_last_fill_time = time;
  }
  virtual unsigned long long get_last_fill_time() {
    return m_line_last_fill_time;
  }
  virtual unsigned long long get_last_evict_time() {
    return m_line_last_evict_time;
  }

  virtual void set_last_warp_id(unsigned warp_id) {
    m_last_warp_id = warp_id;
  }
  virtual unsigned get_last_warp_id() {
    return m_last_warp_id;
  }
  virtual void set_last_core_id(unsigned core_id) {
    m_last_core_id = core_id;
  }
  virtual unsigned get_last_core_id() {
    return m_last_core_id;
  }    

  virtual void inc_total_hits() {
    m_total_hits++;
  }
  virtual void inc_total_evictions() {
    m_total_evictions++;
  }
  virtual unsigned long long get_total_accesses() {
    return m_total_accesses;
  }
  virtual unsigned long long get_total_evictions() {
    return m_total_evictions;
  }    
  
  virtual unsigned get_total_hits() {
    return m_total_hits;
  }

  virtual unsigned long long get_evict_gap() { return m_evict_gap; }

  virtual void set_rrpv(unsigned rrpv) { m_rrpv = rrpv; }
  virtual unsigned get_rrpv() { return m_rrpv; }  
  virtual void set_max_rrpv(unsigned rrpv) { m_max_rrpv = rrpv; }
  virtual unsigned get_max_rrpv() { return m_max_rrpv; }  
  virtual void set_cache_name(char* cache_name) { m_cache_name = cache_name; }
  virtual char* get_cache_name() { return m_cache_name; }  
  virtual void inc_rrpv() { m_rrpv++; }
  virtual void dec_rrpv() { m_rrpv--; }

  virtual unsigned long long get_alloc_time() { return m_line_alloc_time; }

  virtual void set_ignore_on_fill(bool m_ignore,
                                  mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask);
    m_ignore_on_fill_status[sidx] = m_ignore;
  }

  virtual void set_modified_on_fill(bool m_modified,
                                    mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask);
    m_set_modified_on_fill[sidx] = m_modified;
  }
  virtual void set_byte_mask_on_fill(bool m_modified) {
    m_set_byte_mask_on_fill = m_modified;
  }

  virtual void set_readable_on_fill(bool readable,
                                    mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask);
    m_set_readable_on_fill[sidx] = readable;
  }
  virtual void set_m_readable(bool readable,
                              mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask);
    m_readable[sidx] = readable;
  }

  virtual bool is_readable(mem_access_sector_mask_t sector_mask) {
    unsigned sidx = get_sector_index(sector_mask);
    return m_readable[sidx];
  }
  virtual bool is_readable(unsigned sidx) {
    return m_readable[sidx];
  }

  virtual unsigned get_modified_size() {
    unsigned modified = 0;
    for (unsigned i = 0; i < SECTOR_CHUNK_SIZE; ++i) {
      if (m_status[i] == MODIFIED) modified++;
    }
    return modified * SECTOR_SIZE;
  }

  virtual void print_status() {
    printf("m_block_addr is %#llx, status = {%u %u %u %u}\n", m_block_addr,
           m_status[0], m_status[1], m_status[2], m_status[3]);
  }
  virtual void print_readable() {
    printf("m_block_addr is %#llx, readable = {%u %u %u %u}\n", m_block_addr,
           m_readable[0], m_readable[1], m_readable[2], m_readable[3]);
  }

  virtual std::string get_sector_status(unsigned sidx) {
    // INVALID = 0, RESERVED, VALID, MODIFIED,
    switch (m_status[sidx])
    {
    case INVALID:
      return "INVALID";
    case RESERVED:
      return "RESERVED";
    case VALID:
      return "VALID";
    case MODIFIED:
      return "MODIFIED";
    default:
      return "UNKNOWN";
    }
  }

 private:
  //////////////////////// no-used var. ////////////////////////
  unsigned long long m_sector_alloc_time[SECTOR_CHUNK_SIZE];
  unsigned long long m_last_sector_access_time[SECTOR_CHUNK_SIZE];  
  unsigned long long m_sector_fill_time[SECTOR_CHUNK_SIZE];
  bool m_ignore_on_fill_status[SECTOR_CHUNK_SIZE];
  //////////////////////////////////////////////////////////////

  // LRU replacement_policy related control info.  
  unsigned long long m_line_alloc_time;
  unsigned long long m_reref_gap;
  unsigned long long m_line_last_access_time;
  unsigned long long m_line_last_fill_time;
  unsigned long long m_line_last_evict_time;
  unsigned m_last_warp_id;
  unsigned m_last_core_id;  
  unsigned m_total_accesses;
  unsigned m_total_hits;
  unsigned m_total_evictions;
  // Static Re-reference Interval Prediction (SRRIP) related control info.
  unsigned m_rrpv;
  unsigned long long m_evict_gap;
  unsigned m_max_rrpv;
  char* m_cache_name;
  
  // MetaData
  cache_block_state m_status[SECTOR_CHUNK_SIZE];  
  bool m_set_modified_on_fill[SECTOR_CHUNK_SIZE];
  bool m_set_readable_on_fill[SECTOR_CHUNK_SIZE];
  bool m_set_byte_mask_on_fill;
  bool m_readable[SECTOR_CHUNK_SIZE];

  mem_access_byte_mask_t m_dirty_byte_mask;

  unsigned get_sector_index(mem_access_sector_mask_t sector_mask) {
    assert(sector_mask.count() == 1);
    for (unsigned i = 0; i < SECTOR_CHUNK_SIZE; ++i) {
      if (sector_mask.to_ulong() & (1 << i)) return i;
    }
    return SECTOR_CHUNK_SIZE;  // error
  }
};

enum set_index_function {
  LINEAR_SET_FUNCTION = 0,
  BITWISE_XORING_FUNCTION,
  HASH_IPOLY_FUNCTION,
  FERMI_HASH_SET_FUNCTION,
  CUSTOM_SET_FUNCTION,
  WARP_CORR_FUNCTION
};

enum cache_type { NORMAL = 0, SECTOR };

#define MAX_WARP_PER_SHADER 64
#define INCT_TOTAL_BUFFER 64
#define L2_TOTAL 64
#define MAX_WARP_PER_SHADER 64
#define MAX_WARP_PER_SHADER 64

typedef unsigned long long u64;
typedef unsigned u32;
class cache_config {
 public:
  cache_config() {    
    m_valid = false;
    m_disabled = false;
    m_config_string = NULL;  // set by option parser
    m_config_stringPrefL1 = NULL;
    m_config_stringPrefShared = NULL;
    m_data_port_width = 0;
    m_set_index_function = LINEAR_SET_FUNCTION;
    m_is_streaming = false;
    m_wr_percent = 0;    
  }
  void init(
    char *config, 
    char* mshr_config,     
    char* rrpv_config, 
    char* rep_enhance_config,
    FuncCache status, const char* cache_name = "") {
    cache_status = status;
    m_cache_name = cache_name;
    assert(config);
    assert(mshr_config);
    assert(rrpv_config);
    assert(rep_enhance_config);

    m_bypass_enable = 'F';
    m_max_evictions_bound = 0;
    m_trash_conf_cnt_bound = 0;

    [[maybe_unused]] int ntok_mshr = 
      sscanf(mshr_config, "%c,%c", &m_mshr_disable, &m_mshr_corr_repl);
    if (m_mshr_disable == 'T') {
      assert(m_mshr_corr_repl == 'F');
    }
    fprintf(Trace::out, 
      "----------- %s mshr_config is below -----------\n "
      "m_mshr_disable = %c m_mshr_corr_repl = %c\n",
      cache_name, m_mshr_disable, m_mshr_corr_repl);

    [[maybe_unused]] int ntok_rrpv = 
      sscanf(rrpv_config, 
            "%u,%c,%c", 
            &m_rrpv_bits, &m_combined_srrip_lru, &m_srrip_up);

    [[maybe_unused]] int ntok_rep_enhance = 
      sscanf(rep_enhance_config, "%c,%c", &m_fill_time_ascend, &m_warp_interfere_aware);
    fprintf(Trace::out, 
      "----------- %s rep_enhance_config is below -----------\n "
      "m_fill_time_ascend = %c m_warp_interfere_aware = %c\n",
      cache_name, m_fill_time_ascend, m_warp_interfere_aware);

    fprintf(Trace::out, 
      "----------- %s srrip_config is below -----------\n "
      "m_rrpv_bits = %u m_combined_srrip_lru = %c m_srrip_up = %c\n",
      cache_name, m_rrpv_bits, m_combined_srrip_lru, m_srrip_up);
    switch (m_srrip_up) {
      case 'H':
        m_srrip_update_policy = srrip_update_policy_t::HP;
        break;
      case 'F':
        m_srrip_update_policy = srrip_update_policy_t::FP;
        break;
      default:
        exit_parse_error();
    }


    char ct, rp, wp, ap, mshr_type, wap;

    //  S:32:128:24  L : B: m: L: P, A:192:4,  32:0,  32
    // %c:%u:%u:%u,  %c:%c:%c:%c:%c, %c:%u:%u, %u:%u, %u, 
    int ntok =
        sscanf(config, 
              "%c:%u:%u:%u, %c:%c:%c:%c:%c, %c:%u:%u, %u:%u, %u", 
              &ct, &m_nset, &m_line_sz, &m_assoc, 
              &rp, &wp, &ap, &wap, &m_sif,
              &mshr_type, &m_mshr_entries, &m_mshr_max_merge,
              &m_miss_queue_size, &m_result_fifo_entries,
              &m_data_port_width);
    fprintf(Trace::out, "----------- %s cache_config is below -----------\n"
      "%s\nsets = %u\nline_size = %uB\nassoc = %u\n"
      "Replacement Policy (rp) = %c\n"
      "Write Policy (wp) = %c\n"
      "Allocation Policy (ap) = %c\n"
      "Write Allocate Policy (wap) = %c\n"
      "Set Index Function (m_sif) = %c\n"
      "mshr_type = %c\n"
      "m_mshr_entries = %u\n"
      "m_mshr_max_merge = %u\n"
      "m_miss_queue_size = %u\n"
      "m_result_fifo_entries = %u\n"
      "m_data_port_width = %uB\n",
      cache_name, 
      (ct == 'S') ? "SECTOR" : "NORMAL",
      m_nset, m_line_sz, m_assoc, rp, wp, ap, wap, m_sif, mshr_type,
      m_mshr_entries, m_mshr_max_merge, 
      m_miss_queue_size, m_result_fifo_entries,
      m_data_port_width
    );
    m_sector_size = m_line_sz / SECTOR_CHUNK_SIZE;

    if (ntok < 12) {
      if (!strcmp(config, "none")) {
        m_disabled = true;
        return;
      }
      exit_parse_error();
    }

    // for debug
    std::string rp_str = "LRU";
    std::string wp_str = "READ_ONLY";
    std::string wap_str = "NO_WRITE_ALLOCATE";

    switch (ct) {
      case 'N':
        m_cache_type = NORMAL;
        break;
      case 'S':
        m_cache_type = SECTOR;
        break;
      default:
        exit_parse_error();
    }
    switch (rp) {
      case 'R':
        m_replacement_policy = SRRIP;
        rp_str = "SRRIP";
        break;      
      case 'L':
        m_replacement_policy = LRU;
        rp_str = "LRU";
        break;
      case 'F':
        m_replacement_policy = FIFO;
        rp_str = "FIFO";
        break;
      default:
        exit_parse_error();
    }
    switch (wp) {
      case 'R':
        m_write_policy = READ_ONLY;
        wp_str = "READ_ONLY";
        break;
      case 'B':
        m_write_policy = WRITE_BACK;
        wp_str = "WRITE_BACK";
        break;
      case 'T':
        m_write_policy = WRITE_THROUGH;
        wp_str = "WRITE_THROUGH";
        break;
      case 'E':
        m_write_policy = WRITE_EVICT;
        wp_str = "WRITE_EVICT";
        break;
      case 'L':
        m_write_policy = LOCAL_WB_GLOBAL_WE;
        wp_str = "LOCAL_WB_GLOBAL_WE";
        break;
      default:
        exit_parse_error();
    }
    switch (ap) {
      case 'm':
        m_alloc_policy = ON_MISS;
        break;
      case 'f':
        m_alloc_policy = ON_FILL;
        break;
      case 's':
        m_alloc_policy = STREAMING;
        break;
      default:
        exit_parse_error();
    }
    if (m_alloc_policy == STREAMING) {
      /*
      For streaming cache:
      (1) we set the alloc policy to be on-fill to remove all line_alloc_fail
      stalls. if the whole memory is allocated to the L1 cache, then make the
      allocation to be on_MISS otherwise, make it ON_FILL to eliminate line
      allocation fails. i.e. MSHR throughput is the same, independent on the L1
      cache size/associativity So, we set the allocation policy per kernel
      basis, see shader.cc, max_cta() function

      (2) We also set the MSHRs to be equal to max
      allocated cache lines. This is possible by moving TAG to be shared
      between cache line and MSHR entry (i.e. for each cache line, there is
      an MSHR rntey associated with it). This is the easiest think we can
      think of to model (mimic) L1 streaming cache in Pascal and Volta

      For more information about streaming cache, see:
      http://on-demand.gputechconf.com/gtc/2017/presentation/s7798-luke-durant-inside-volta.pdf
      https://ieeexplore.ieee.org/document/8344474/
      */
      m_is_streaming = true;
      m_alloc_policy = ON_FILL;
    }
    switch (mshr_type) {
      case 'F':
        m_mshr_type = TEX_FIFO;
        assert(ntok == 14);
        break;
      case 'T':
        m_mshr_type = SECTOR_TEX_FIFO;
        assert(ntok == 14);
        break;
      case 'A':
        m_mshr_type = ASSOC;
        break;
      case 'S':
        m_mshr_type = SECTOR_ASSOC;
        break;
      default:
        exit_parse_error();
    }
    m_line_sz_log2 = LOGB2(m_line_sz);
    m_nset_log2 = LOGB2(m_nset);
    m_valid = true;
    m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz;

// #ifdef ARISE2_L1P5
//     std::cerr << "ARISE2_L1P5 defined\n";
//     fprintf(Trace::out, "ARISE2_L1P5 defined\n");
// #else
//     std::cerr << "ARISE2_L1P5 not defined\n";
//     fprintf(Trace::out, "ARISE2_L1P5 not defined\n");
// #endif
    // std::cerr << m_cache_name << " m_atom_sz = " << m_atom_sz << 
    //   " SECTOR_SIZE = " << SECTOR_SIZE << "\n";
    fprintf(Trace::out, "%s m_atom_sz = %u B, SECTOR_SIZE = %u B\n",
      m_cache_name, m_atom_sz, SECTOR_SIZE
    );

    m_sector_sz_log2 = LOGB2(SECTOR_SIZE);
    original_m_assoc = m_assoc;

    // For more details about difference between FETCH_ON_WRITE and WRITE
    // VALIDAE policies Read: Jouppi, Norman P. "Cache write policies and
    // performance". ISCA 93. WRITE_ALLOCATE is the old write policy in
    // GPGPU-sim 3.x, that send WRITE and READ for every write request
    switch (wap) {
      case 'N':
        m_write_alloc_policy = NO_WRITE_ALLOCATE;
        wap_str = "NO_WRITE_ALLOCATE";
        break;
      case 'W':
        m_write_alloc_policy = WRITE_ALLOCATE;
        wap_str = "WRITE_ALLOCATE";
        break;
      case 'F':
        m_write_alloc_policy = FETCH_ON_WRITE;
        wap_str = "FETCH_ON_WRITE";
        break;
      case 'L':
        m_write_alloc_policy = LAZY_FETCH_ON_READ;
        wap_str = "LAZY_FETCH_ON_READ";
        break;
      default:
        exit_parse_error();
    }

    // detect invalid configuration
    if ((m_alloc_policy == ON_FILL || m_alloc_policy == STREAMING) and
        m_write_policy == WRITE_BACK) {
      // A writeback cache with allocate-on-fill policy will inevitably lead to
      // deadlock: The deadlock happens when an incoming cache-fill evicts a
      // dirty line, generating a writeback request.  If the memory subsystem is
      // congested, the interconnection network may not have sufficient buffer
      // for the writeback request.  This stalls the incoming cache-fill.  The
      // stall may propagate through the memory subsystem back to the output
      // port of the same core, creating a deadlock where the wrtieback request
      // and the incoming cache-fill are stalling each other.
      assert(0 &&
             "Invalid cache configuration: Writeback cache cannot allocate new "
             "line on fill. ");
    }

    if ((m_write_alloc_policy == FETCH_ON_WRITE ||
         m_write_alloc_policy == LAZY_FETCH_ON_READ) &&
        m_alloc_policy == ON_FILL) {
      assert(
          0 &&
          "Invalid cache configuration: FETCH_ON_WRITE and LAZY_FETCH_ON_READ "
          "cannot work properly with ON_FILL policy. Cache must be ON_MISS. ");
    }

    if (m_cache_type == SECTOR) {
      bool cond = m_line_sz / SECTOR_SIZE == SECTOR_CHUNK_SIZE &&
                  m_line_sz % SECTOR_SIZE == 0;
      if (!cond) {
        std::cerr << "assert failed! " << cache_name << 
          " (m_line_sz:" << m_line_sz << 
          " / SECTOR_SIZE:" << SECTOR_SIZE << 
          ") != SECTOR_CHUNK_SIZE:" << SECTOR_CHUNK_SIZE << "\n";

        std::cerr << "error: For sector cache, the simulator uses hard-coded "
                    "SECTOR_SIZE and SECTOR_CHUNK_SIZE. The line size "
                    "must be product of both values.\n";
        assert(0);
      }
    }

    // default: port to data array width and granularity = line size
    if (m_data_port_width == 0) {
      m_data_port_width = m_line_sz;
    }
    assert(m_line_sz % m_data_port_width == 0);

    switch (m_sif) {
      case 'H':
        m_set_index_function = FERMI_HASH_SET_FUNCTION;
        break;
      case 'P':
        m_set_index_function = HASH_IPOLY_FUNCTION;
        break;
      case 'C':
        m_set_index_function = CUSTOM_SET_FUNCTION;
        break;
      case 'L':
        m_set_index_function = LINEAR_SET_FUNCTION;
        break;
      case 'W':
        m_set_index_function = WARP_CORR_FUNCTION;
        break;
      case 'X':
        m_set_index_function = BITWISE_XORING_FUNCTION;
        break;
      default:
        exit_parse_error();
    }

    printf("Replacement Policy (rp)=%s\n"
      "Write Policy (wp)=%s\n"
      "Write Allocate Policy (wap)=%s\n",
      rp_str.c_str(), wp_str.c_str(), wap_str.c_str()
    );
    if (DTRACE(CACHE_CONFIG)) {
      fprintf(Trace::out, "Replacement Policy (rp)=%s\n"
        "Write Policy (wp)=%s\n"
        "Write Allocate Policy (wap)=%s\n",
        rp_str.c_str(), wp_str.c_str(), wap_str.c_str()
      );
    }
  }

  bool disabled() const { return m_disabled; }
  unsigned get_line_sz() const {
    assert(m_valid);
    return m_line_sz;
  }
  unsigned get_atom_sz() const {
    assert(m_valid);
    return m_atom_sz;
  }
  unsigned get_num_lines() const {
    assert(m_valid);
    return m_nset * m_assoc;
  }
  unsigned get_max_num_lines() const {
    assert(m_valid);
    return get_max_cache_multiplier() * m_nset * original_m_assoc;
  }
  unsigned get_max_assoc() const {
    assert(m_valid);
    return get_max_cache_multiplier() * original_m_assoc;
  }
  void print(FILE *fp) const {
    fprintf(fp, "Size = %d B (%d Set x %d-way x %d byte line)\n",
            m_line_sz * m_nset * m_assoc, m_nset, m_assoc, m_line_sz);
  }

  unsigned get_line_bits() const {
    return m_line_sz_log2;
  }
  unsigned get_set_bits() const {
    return m_nset_log2;
  }

  virtual unsigned set_index(
    new_addr_type addr, unsigned warp_id = (unsigned) - 1) const;
  virtual std::pair<unsigned, unsigned> set_index_pairs(
    new_addr_type addr, unsigned warp_id = (unsigned) - 1) const;

  virtual unsigned recalc_orig_addr(new_addr_type tag, unsigned set_index) const;

  virtual unsigned get_max_cache_multiplier() const {
    return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER;
  }

  unsigned hash_function(new_addr_type addr, unsigned m_nset,
                         unsigned m_line_sz_log2, unsigned m_nset_log2,
                         unsigned m_index_function, 
                         unsigned warp_id = (unsigned) - 1) const;

  new_addr_type tag(new_addr_type addr) const {
    // For generality, the tag includes both index and tag. This allows for more
    // complex set index calculations that can result in different indexes
    // mapping to the same set, thus the full tag + index is required to check
    // for hit/miss. Tag is now identical to the block address.  

    // return (m_sif == 'L') ? (addr >> (m_line_sz_log2 + m_nset_log2)) :
    //   (addr & ~(new_addr_type)(m_line_sz - 1));

    // return addr >> (m_line_sz_log2 + m_nset_log2);
    return addr & ~(new_addr_type)(m_line_sz - 1);
  }
  new_addr_type block_addr(new_addr_type addr) const {
    return addr & ~(new_addr_type)(m_line_sz - 1);
  }
  new_addr_type mshr_addr(new_addr_type addr) const {
    return addr & ~(new_addr_type)(m_atom_sz - 1);
  }
  enum mshr_config_t get_mshr_type() const { return m_mshr_type; }
  void set_assoc(unsigned n) {
    // set new assoc. L1 cache dynamically resized in Volta
    m_assoc = n;
  }
  unsigned get_nset() const {
    assert(m_valid);
    return m_nset;
  }
  unsigned get_total_size_inKB() const {
    assert(m_valid);
    return (m_assoc * m_nset * m_line_sz) / 1024;
  }
  bool is_streaming() { return m_is_streaming; }
  FuncCache get_cache_status() { return cache_status; }
  void set_allocation_policy(enum allocation_policy_t alloc) {
    m_alloc_policy = alloc;
  }
  char *m_config_string;
  char *m_mshr_config_string;
  char *m_rrpv_config_string;
  char *m_rep_enhance_string;
  char *m_bypass_config_string;
  char *m_config_stringPrefL1;
  char *m_config_stringPrefShared;

  char m_bypass_enable;
  char m_total_evictions_aware;
  u32 m_max_evictions_bound;
  u32 m_trash_conf_cnt_bound;

  FuncCache cache_status;
  unsigned m_wr_percent;
  write_allocate_policy_t get_write_allocate_policy() {
    return m_write_alloc_policy;
  }
  write_policy_t get_write_policy() { return m_write_policy; }

  const char* get_cache_name() const { return m_cache_name; }
  const unsigned getSectorSize() const { return m_sector_size; }
  const unsigned get_sub_partition() const { return m_sub_partition; }
  const unsigned get_mshr_max_merge() const { return m_mshr_max_merge; }
  const unsigned get_mshr_entries() const { return m_mshr_entries; }
  const char get_sif() const { return m_sif; }
  const char get_mshr_disable() const { return m_mshr_disable; }

 protected:
  void exit_parse_error() {
    printf("GPGPU-Sim uArch: cache configuration parsing error (%s)\n",
           m_config_string);
    abort();
  }

  const char* m_cache_name;
  unsigned m_sector_size; // Globallly replace the hard-coded "SECTOR_SIZE"
  unsigned m_sub_partition;
  unsigned m_num_cores; // passed from m_shader_config->n_simt_cores_per_cluster
  bool m_valid;
  bool m_disabled;
  unsigned m_line_sz;
  unsigned m_line_sz_log2;
  unsigned m_nset;
  unsigned m_nset_log2;
  unsigned m_assoc;
  unsigned m_atom_sz;
  unsigned m_sector_sz_log2;
  char m_sif;
  unsigned original_m_assoc;
  bool m_is_streaming;

  enum replacement_policy_t m_replacement_policy;  // 'L' = LRU, 'F' = FIFO
  enum write_policy_t
      m_write_policy;  // 'T' = write through, 'B' = write back, 'R' = read only
  enum allocation_policy_t
      m_alloc_policy;  // 'm' = allocate on miss, 'f' = allocate on fill
  enum mshr_config_t m_mshr_type;
  enum cache_type m_cache_type;

  write_allocate_policy_t
      m_write_alloc_policy;  // 'W' = Write allocate, 'N' = No write allocate

  char m_mshr_disable;
  char m_mshr_corr_repl;
  char m_fill_time_ascend;
  char m_warp_interfere_aware;

  enum srrip_update_policy_t m_srrip_update_policy;
  unsigned m_rrpv_bits;
  char m_combined_srrip_lru;
  char m_srrip_up;

  union {
    unsigned m_mshr_entries;
    unsigned m_fragment_fifo_entries;
  };
  union {
    unsigned m_mshr_max_merge;
    unsigned m_request_fifo_entries;
  };
  union {
    unsigned m_miss_queue_size;
    unsigned m_rob_entries;
  };
  unsigned m_result_fifo_entries;
  unsigned m_data_port_width;  //< number of byte the cache can access per cycle
  enum set_index_function
      m_set_index_function;  // Hash, linear, or custom set index function

  friend class tag_array;
  friend class baseline_cache;
  friend class read_only_cache;
  friend class tex_cache;
  friend class data_cache;
  friend class l1_cache;
  friend class l2_cache;
  friend class memory_sub_partition;
};

class l1d_cache_config : public cache_config {
 public:
  l1d_cache_config() : cache_config() {
  }
  unsigned set_bank(new_addr_type addr) const;
  void init(
    char *config, char *mshr_config, char* rrpv_config, 
    char* rep_enhance_config,
    FuncCache status, const char* cache_name = "L1D") {
    l1_banks_byte_interleaving_log2 = LOGB2(l1_banks_byte_interleaving);
    l1_banks_log2 = LOGB2(l1_banks);
    cache_config::init(config, mshr_config, rrpv_config, 
      rep_enhance_config, status, cache_name);
  }

  unsigned m_shader_cores;
  unsigned l1_latency;
  unsigned l1_banks;
  unsigned l1_banks_log2;
  unsigned l1_banks_byte_interleaving;
  unsigned l1_banks_byte_interleaving_log2;
  unsigned l1_banks_hashing_function;
  unsigned m_unified_cache_size;

  void extra_config(char* bypass_config);
  virtual unsigned get_max_cache_multiplier() const;
};

class l2_cache_config : public cache_config {
 public:
  l2_cache_config() : cache_config() {
    cache_name = "L2";
  }
  void init(linear_to_raw_address_translation *address_mapping);
  virtual unsigned set_index(
    new_addr_type addr, unsigned warp_id = (unsigned) - 1) const;
  bool m_disable_wr_merge;
 private:
  const char* cache_name;
  linear_to_raw_address_translation *m_address_mapping;
};

class prefetcher {
  public:
    prefetcher();
    ~prefetcher();
};

enum LINE_RECENCY_ITEMS {
  AVG_EVICT_INTERVAL = 0,
  TOTAL_HITS,
  LAST_ACCESS_TIME,
  UNFOLDED_IDX
};

struct WARP_INTERFERE_RECORD {
  unsigned last_warp_id;
  unsigned curr_warp_id;
  WARP_INTERFERE_RECORD(
    unsigned last_warp_id_,
    unsigned curr_warp_id_
  ) : 
  last_warp_id(last_warp_id_),
  curr_warp_id(curr_warp_id_) {}
};

struct LINE_RECENCY {
  bool is_valid;
  bool was_recorded_in_mshr;
  unsigned long long reref_gap;
  unsigned long long last_access_time;
  unsigned long long last_fill_time;
  unsigned long long last_evict_time;
  unsigned long long evict_gap;
  unsigned rrpv;
  unsigned max_rrpv;
  unsigned total_hits;
  unsigned total_evictions;
  unsigned total_accesses;
  unsigned warp_id;
  unsigned core_id;
  LINE_RECENCY(
    bool is_valid_,
    bool was_recorded_in_mshr_,
    unsigned long long reref_gap_,
    unsigned long long last_access_time_,
    unsigned long long last_fill_time_,
    unsigned long long last_evict_time_,
    unsigned long long evict_gap_,
    unsigned rrpv_,    
    unsigned max_rrpv_,
    unsigned total_hits_,
    unsigned total_evictions_,
    unsigned total_accesses_,
    unsigned warp_id_,
    unsigned core_id_
  ) : 
  is_valid(is_valid_),
  was_recorded_in_mshr(was_recorded_in_mshr_),
  reref_gap(reref_gap_),
  last_access_time(last_access_time_),
  last_fill_time(last_fill_time_),
  last_evict_time(last_evict_time_),
  evict_gap(evict_gap_),
  rrpv(rrpv_),
  max_rrpv(max_rrpv_),
  total_hits(total_hits_),
  total_evictions(total_evictions_),
  total_accesses(total_accesses_),
  warp_id(warp_id_),
  core_id(core_id_)
  {}

  void init() {
    is_valid = false;
    was_recorded_in_mshr = false;
    reref_gap        = (unsigned long long) - 1;
    last_access_time = (unsigned long long) - 1;
    last_fill_time   = (unsigned long long) - 1;
    last_evict_time  = (unsigned long long) - 1;
    evict_gap        = (unsigned long long) - 1;
    rrpv                   = (unsigned) - 1;
    max_rrpv               = (unsigned) - 1;
    total_hits             = (unsigned) - 1;
    total_evictions        = (unsigned) - 1;
    total_accesses         = (unsigned) - 1;
    warp_id = (unsigned) - 1;
    core_id = (unsigned) - 1;
  }
};

struct LINE_LOCALITY
{
  unsigned core_id;
  new_addr_type addr;
  unsigned total_evictions;
  LINE_LOCALITY() : 
    core_id((unsigned) - 1), addr((new_addr_type) - 1), total_evictions(0) {}
  LINE_LOCALITY(
    unsigned core_id_, new_addr_type addr_, unsigned total_evictions_) :
    core_id(core_id_), addr(addr_), total_evictions(total_evictions_) {}
};
struct REQ_PKT
{
  unsigned uid;
  new_addr_type addr;
  REQ_PKT() : uid((unsigned) - 1), addr((new_addr_type) - 1) {}
  REQ_PKT(unsigned uid_, new_addr_type addr_) :
    uid(uid_), addr(addr_) {}

  bool operator<(const REQ_PKT& o) const {
    if (uid != o.uid) {
      return uid < o.uid;
    }
    return addr < o.addr;
  }
};

typedef unsigned long long u64;
typedef unsigned u32;

struct LOCALITY_KEY
{
  u64 stream_id;
  u32 kernel;
  LOCALITY_KEY() : 
    stream_id((u64) - 1), kernel((u32) - 1) {}
  LOCALITY_KEY(
    u64 stream_id_, u32 kernel_) :
    stream_id(stream_id_), kernel(kernel_) {}

  bool operator<(const LOCALITY_KEY& other) const {
    if (stream_id != other.stream_id) return stream_id < other.stream_id;
    if (kernel != other.kernel) return kernel < other.kernel;
    return false;
  }

  bool operator==(const LOCALITY_KEY& other) const {
    return stream_id == other.stream_id && kernel == other.kernel;
  }
};
struct LOCALITY_KEY_HASH {
  std::size_t operator()(const LOCALITY_KEY& key) const {
    const std::size_t h1 = std::hash<u64>{}(key.stream_id);
    const std::size_t h2 = std::hash<u32>{}(key.kernel);
    return h1 ^ (h2 << 1);
  }
};

struct BYPASS_KEY
{
  u64 stream_id;
  u32 kernel;
  u64 block_addr;
  u64 sector_addr;
  BYPASS_KEY() : 
    stream_id((u64) - 1), kernel((u32) - 1), 
    block_addr((u64) - 1), sector_addr((u64) - 1) {}
  BYPASS_KEY(
    u64 stream_id_, u32 kernel_, u64 block_addr_, u64 sector_addr_) :
    stream_id(stream_id_), kernel(kernel_), 
    block_addr(block_addr_), sector_addr(sector_addr_) {} 

  bool operator<(const BYPASS_KEY& other) const {
    if (stream_id != other.stream_id) return stream_id < other.stream_id;
    if (kernel != other.kernel) return kernel < other.kernel;
    if (block_addr != other.block_addr) return block_addr < other.block_addr;
    if (sector_addr != other.sector_addr) return sector_addr < other.sector_addr;
    return false;
  }

  bool operator==(const BYPASS_KEY& other) const {
    return stream_id == other.stream_id &&
      kernel == other.kernel &&
      block_addr == other.block_addr &&
      sector_addr == other.sector_addr;
  }
};

struct BYPASS_KEY_REQ_HASH {
  std::size_t operator()(const BYPASS_KEY& key) const {
    const std::size_t h1 = std::hash<u64>{}(key.stream_id);
    const std::size_t h2 = std::hash<u32>{}(key.kernel);
    const std::size_t h3 = std::hash<u64>{}(key.block_addr);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};
struct BYPASS_KEY_REQ_EQ {
  bool operator()(const BYPASS_KEY& a, const BYPASS_KEY& b) const {
    return a.stream_id == b.stream_id &&
           a.kernel == b.kernel &&
           a.block_addr == b.block_addr;
  }
};
struct BYPASS_KEY_REQ_LESS {
  bool operator()(const BYPASS_KEY& a, const BYPASS_KEY& b) const {
    if (a.stream_id != b.stream_id) return a.stream_id < b.stream_id;
    if (a.kernel != b.kernel) return a.kernel < b.kernel;
    return a.block_addr < b.block_addr;
  }
};

struct BYPASS_KEY_RESP_HASH {
  std::size_t operator()(const BYPASS_KEY& key) const {
    const std::size_t h1 = std::hash<u64>{}(key.stream_id);
    const std::size_t h2 = std::hash<u32>{}(key.kernel);
    const std::size_t h3 = std::hash<u64>{}(key.sector_addr);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};
struct BYPASS_KEY_RESP_EQ {
  bool operator()(const BYPASS_KEY& a, const BYPASS_KEY& b) const {
    return a.stream_id == b.stream_id &&
           a.kernel == b.kernel &&
           a.sector_addr == b.sector_addr;
  }
};

class tag_array {
  friend class baseline_cache;
  friend class data_cache;
  friend class cache_config;  
 public:
  // Use this constructor
  tag_array(gpgpu_sim *gpu, cache_config &config, int core_id, int type_id);
  ~tag_array();

  unsigned get_avg_l1d_byp_activated_times() {
    unsigned total_times = 0;
    for (const auto& entry : m_l1d_rd_byp_activated_times) {
      total_times += entry.second;
    }
    float avg_times = total_times / (float)m_l1d_rd_byp_activated_times.size();
    return static_cast<unsigned>(avg_times);
  }

  std::vector<std::set<new_addr_type>> get_l1d_unique_lines() {
    return m_l1d_unique_lines;
  }

  void set_l1d_rd_fill_time(BYPASS_KEY key, u64 time) {
    m_l1d_rd_fill_time[key] = time;
  }
  void set_l1d_evict_time(BYPASS_KEY key, u64 time) {
    m_l1d_evict_time[key] = time;
  }
  void set_l1d_rd_fill_to_evict_gap(BYPASS_KEY key, u64 gap) {
    m_l1d_rd_fill_to_evict_gap[key] = gap;
  }
  u64 get_l1d_evict_time(BYPASS_KEY key) {
    return m_l1d_evict_time[key];
  }

  void average_l1d_rd_fill_to_evict_gap(BYPASS_KEY key) {
    assert(m_l1d_rd_fill_to_evict_gap.find(key) != m_l1d_rd_fill_to_evict_gap.end());
    u64 this_gap = m_l1d_rd_fill_to_evict_gap[key];
    if (m_avg_l1d_rd_fill_to_evict_gap.find(key) == m_avg_l1d_rd_fill_to_evict_gap.end()) {
      m_avg_l1d_rd_fill_to_evict_gap[key] = this_gap;
    } else {
      m_avg_l1d_rd_fill_to_evict_gap[key] = 
        (m_avg_l1d_rd_fill_to_evict_gap[key] + this_gap) >> 1;
    }
  }

  u64 get_l1d_rd_fill_time(BYPASS_KEY key) {
    return m_l1d_rd_fill_time[key];
  }
  u64 get_l1d_rd_fill_to_evict_gap(BYPASS_KEY key) {
    return m_l1d_rd_fill_to_evict_gap[key];
  }  
  u64 get_avg_l1d_rd_fill_to_evict_gap(BYPASS_KEY key) {
    return m_avg_l1d_rd_fill_to_evict_gap[key];
  }

  bool hit_l1d_bypassed_item(BYPASS_KEY key, mem_fetch* mf) {
    if (m_trashed_reqs.find(key) != m_trashed_reqs.end() &&
      !mf->is_write() && !mf->isatomic()) {
      return true;
    }
    return false;
  }
  bool hit_l1d_byp_on_req_path(BYPASS_KEY key, mem_fetch* mf) {
    if (m_trashed_reqs.find(key) != m_trashed_reqs.end() &&
      !mf->is_write() && !mf->isatomic()) {
      return true;
    }
    return false;
  }  

  static bool cmpForSmallerTimestamp(
    const std::pair<unsigned, LINE_RECENCY>& a, 
    const std::pair<unsigned, LINE_RECENCY>& b) {
    return a.second.last_access_time < b.second.last_access_time;
  }
  static bool cmpForSmallerTotalHits(
    const std::pair<unsigned, LINE_RECENCY>& a, 
    const std::pair<unsigned, LINE_RECENCY>& b) {
    return a.second.total_hits < b.second.total_hits;
  }  
  static bool cmpForSmallerFillTime(
    const std::pair<unsigned, LINE_RECENCY>& a, 
    const std::pair<unsigned, LINE_RECENCY>& b) {
    return a.second.last_fill_time < b.second.last_fill_time;
  }

  void gather_rep_candidates(
    unsigned long long time,
    mem_fetch* mf, cache_block_t* line, const unsigned& set_index, const unsigned& index,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates);

  // Update valid_line with index parsed from [set_index][way]
  void lru_pick(
    cache_block_t* line, unsigned long long& valid_timestamp, 
    unsigned& valid_line, const unsigned& index, 
    unsigned& warp_id, unsigned& core_id,
    bool& lru_has_picked);
  void fill_time_pick(
    cache_block_t* line, unsigned long long& valid_timestamp, 
    unsigned& valid_line, const unsigned& index);

  void warp_interfere_aware_pick(
    unsigned long long time,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
    bool& has_interfered,
    unsigned& valid_line,
    unsigned& warp_id, unsigned& core_id,
    mem_fetch *mf
  );
  void pick_with_lru(
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
    unsigned& valid_line,
    unsigned& warp_id, unsigned& core_id,
    unsigned long long& smallest_access_time,
    unsigned& lru_picked_total_hits,
    unsigned long long& lru_picked_avg_evict_interval
  );
  void pick_modified_by_total_hits_ascend(
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
    unsigned& valid_line, const unsigned& lru_picked_total_hits
  );
  void fill_time_awared_modification_for_lru(
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
    unsigned& valid_line,
    const unsigned long long& smallest_last_access_time);

  void fill_time_awared_modification_for_srrip(
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
    unsigned& valid_line);
  void mshr_awared_modification_for_srrip(
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_no_record_in_mshr,
    std::vector<std::pair<unsigned, LINE_RECENCY>>& hybrid_rep_candidates_recorded_in_mshr,
    unsigned& valid_line);  

  void reset_cnt_for_abort_bypass(int index);
  
  enum cache_request_status probe(const std::string& caller,
                                  bool early_return /* L1D bypass required */,
                                  new_addr_type raw_addr,
                                  new_addr_type addr /* block_addr */, unsigned &idx,
                                  mem_fetch *mf, bool is_write,
                                  unsigned long long time,
                                  bool& inter_warp_has_interference, 
                                  WARP_INTERFERE_RECORD& warp_interfere_record,
                                  
                                  bool probe_mode = false);
  enum cache_request_status probe(const std::string& caller,
                                  bool early_return /* L1D bypass required */,
                                  new_addr_type raw_addr,
                                  new_addr_type addr /* block_addr */, unsigned &idx,
                                  mem_access_sector_mask_t mask, bool is_write,
                                  unsigned long long time,
                                  bool probe_mode,
                                  bool& inter_warp_has_interference, 
                                  WARP_INTERFERE_RECORD& warp_interfere_record,                                  
                                  mem_fetch *mf = NULL);
  enum cache_request_status access(new_addr_type raw_addr, 
                                   new_addr_type addr /* block_addr */, unsigned long long time,
                                   unsigned &idx, mem_fetch *mf);
  enum cache_request_status access(new_addr_type raw_addr,
                                   new_addr_type addr /* block_addr */, unsigned long long time,
                                   unsigned &idx, bool &wb,
                                   evicted_block_info &evicted, mem_fetch *mf);
  void inc_rrpv_for_one_set(unsigned set_index);
  bool already_has_max_rrpv_in_one_set(unsigned set_index);

  void fill(new_addr_type addr, unsigned long long time, mem_fetch *mf, bool is_write);
  void fill(unsigned index, unsigned long long time, mem_fetch *mf);
  void fill(new_addr_type addr, unsigned long long time, mem_access_sector_mask_t mask,
            mem_access_byte_mask_t byte_mask, bool is_write,
            mem_fetch *mf = NULL);
  void reset_record_in_mshr(unsigned index);
  void set_recorded_in_mshr(unsigned index);

  unsigned size() const { return m_config.get_num_lines(); }
  cache_block_t *get_block(unsigned idx) { return m_lines[idx]; }

  void flush();       // flush all written entries
  void invalidate();  // invalidate all entries
  void new_window();

  void print(FILE *stream, unsigned &total_access,
             unsigned &total_misses) const;
  float windowed_miss_rate() const;
  void get_stats(unsigned &total_access, unsigned &total_misses,
                 unsigned &total_hit_res, unsigned &total_res_fail) const;

  void update_cache_parameters(cache_config &config);
  void add_pending_line(mem_fetch *mf);
  void remove_pending_line(mem_fetch *mf);
  void inc_dirty() { m_dirty++; }

  u32 get_l1d_evictions(const BYPASS_KEY& byp_key) {
    return m_l1d_evictions[byp_key];
  }

 protected:
  // This constructor is intended for use only from derived classes that wish to
  // avoid unnecessary memory allocation that takes place in the
  // other tag_array constructor
  tag_array(cache_config &config, int core_id, int type_id,
            cache_block_t **new_lines);
  void init(int core_id, int type_id);

 protected:
  gpgpu_sim *m_gpu;
  cache_config &m_config;

  cache_block_t **m_lines; /* nbanks x nset x assoc lines in total */
  unsigned m_total_records_in_mshr;

  bool m_is_l1d;
  bool m_is_l2;
  unsigned m_access;
  unsigned m_reads;
  unsigned m_writes;
  unsigned m_rd_miss;
  unsigned m_wr_miss;
  unsigned m_miss;
  unsigned m_pending_hit;  // number of cache miss that hit a line that is
                           // allocated but not filled
  unsigned m_res_fail;
  unsigned m_rd_sector_miss;
  unsigned m_wr_sector_miss;
  unsigned m_sector_miss;
  unsigned m_dirty;

  std::map<new_addr_type, bool> m_mshr_recorded_block_addresses;  

  // performance counters for calculating the amount of misses within a time
  // window
  unsigned m_prev_snapshot_access;
  unsigned m_prev_snapshot_miss;
  unsigned m_prev_snapshot_pending_hit;

  int m_core_id;  // which shader core is using this
  int m_type_id;  // what kind of cache is this (normal, texture, constant)

  bool is_used;  // a flag if the whole cache has ever been accessed before

  std::unordered_set<BYPASS_KEY, BYPASS_KEY_REQ_HASH, BYPASS_KEY_REQ_EQ> m_trashed_reqs;

  typedef tr1_hash_map<new_addr_type, u32> line_table;
  line_table pending_lines;
  line_table lines_locality;
  std::vector<std::set<new_addr_type>> m_l1d_unique_lines;
  std::vector<std::vector<std::pair<new_addr_type, u64>>> m_reref_gap;  
  std::vector<u64> m_avg_reref_gap;
  std::vector<u32> m_l1d_max_evicts;
  std::vector<u32> m_l1d_avg_evicts;
  std::map<BYPASS_KEY, u64 /* cycles */> m_l1d_rd_fill_time;
  std::map<BYPASS_KEY, u64 /* cycles */> m_l1d_evict_time;
  std::map<BYPASS_KEY, u64 /* cycles */> m_l1d_rd_fill_to_evict_gap;
  std::map<BYPASS_KEY, u32 /* evictions */> m_l1d_evictions;
  std::map<BYPASS_KEY, u32> m_l1d_rd_byp_activated_times;
  std::map<BYPASS_KEY, u32> m_l1d_rd_byp_deactivated_times;
  std::map<BYPASS_KEY, int> m_l1d_rd_bypass_confidence;
  std::map<BYPASS_KEY, bool> m_l1d_rd_bypass_activated;
  std::map<BYPASS_KEY, u64 /* cycles */> m_avg_l1d_rd_fill_to_evict_gap;
  std::set<new_addr_type> m_l1d_trashed_lines;
  float m_l1d_mpki;

  void inc_conf_cnt(int& conf, const int upper_bound, const int step);
  void dec_conf_cnt(int& conf, const int lower_bound, const int step);  
};

class mshr_table {
 public:
  mshr_table(unsigned num_entries, unsigned max_merged)
      : m_num_entries(num_entries),
        m_max_merged(max_merged)
#if (tr1_hash_map_ismap == 0)
        ,
        m_data(2 * num_entries)
#endif
  {
  }

  /// Get MSHR occupancy
  unsigned occupied_entries() const { return static_cast<unsigned>(m_data.size()); }
  unsigned merged_slots(new_addr_type block_addr) { return m_data[block_addr].m_list.size(); }
  /// Checks if there is a pending request to the lower memory level already
  bool probe(new_addr_type block_addr) const;
  /// Checks if there is space for tracking a new memory access
  bool full(new_addr_type block_addr) const;
  /// Add or merge this access
  void add(new_addr_type mshr_addr, mem_fetch *mf, bool& is_new_entry, const char* cache_name="");
  unsigned occupied_slots(new_addr_type mshr_addr);
  /// Returns true if cannot accept new fill responses
  bool busy() const { return false; }
  /// Accept a new cache fill response: mark entry ready for processing
  void mark_ready(const char* cache_name, new_addr_type block_addr, bool &has_atomic, unsigned long long cycle);
  /// Returns true if ready accesses exist
  bool access_ready() const { return !m_lfb.empty(); }
  size_t num_pending_responses() const { return m_lfb.size(); }
  /// Returns next ready access
  mem_fetch *next_access(const char* cache_name, unsigned long long cycle = 0);
  void display(FILE *fp, const char* cache_name= "") const;
  void display_resp_q(FILE *fp, const char* cache_name= "") const;
  // Returns true if there is a pending read after write
  bool is_read_after_write_pending(new_addr_type block_addr);

  void check_mshr_parameters(unsigned num_entries, unsigned max_merged) {
    assert(m_num_entries == num_entries &&
           "Change of MSHR parameters between kernels is not allowed");
    assert(m_max_merged == max_merged &&
           "Change of MSHR parameters between kernels is not allowed");
  }
  unsigned get_max_merged() const { return m_max_merged; }

 private:
  // finite sized, fully associative table, with a finite maximum number of
  // merged requests
  const unsigned m_num_entries;
  const unsigned m_max_merged;

  struct mshr_entry {
    std::list<mem_fetch *> m_list;
    bool m_has_atomic;
    mshr_entry() : m_has_atomic(false) {}
  };
  typedef tr1_hash_map<new_addr_type, mshr_entry> table;
  typedef tr1_hash_map<new_addr_type, mshr_entry> line_table;
  table m_data;
  line_table pending_lines;

  // it may take several cycles to process the merged requests
  bool m_current_response_ready;
  std::list<new_addr_type> m_lfb;
};

/***************************************************************** Caches
 * *****************************************************************/
///
/// Simple struct to maintain cache accesses, misses, pending hits, and
/// reservation fails.
///
// Accumulated stats of all kernels
typedef unsigned long long u64;
typedef unsigned u32;
struct cache_sub_stats {
  unsigned long long accesses;
  unsigned long long reads;
  unsigned long long writes;
  unsigned long long misses;
  unsigned long long rd_misses;
  unsigned long long wr_misses;  
  unsigned long long sector_misses;
  unsigned long long sector_rd_misses;
  unsigned long long sector_wr_misses;
  unsigned long long pending_hits;
  unsigned long long res_fails;

  // different from others' overloading of "+" and "+="
  // Here actually implements average
  unsigned long long avg_evict_interval;
  unsigned long long avg_rd_byp_activates;
  unsigned long long avg_rd_byp_deactivates;

  unsigned long long port_available_cycles;
  unsigned long long data_port_busy_cycles;
  unsigned long long fill_port_busy_cycles;

  u32 n_bypassed; 

  cache_sub_stats() { clear(); }
  void clear() {
    accesses = 0;
    reads = 0;
    writes = 0;
    misses = 0;
    rd_misses = 0;
    wr_misses = 0;
    sector_misses = 0;
    sector_rd_misses = 0;
    sector_wr_misses = 0;
    pending_hits = 0;
    res_fails = 0;
    avg_evict_interval = 0;
    avg_rd_byp_activates   = 0;
    avg_rd_byp_deactivates = 0;
    port_available_cycles  = 0;
    data_port_busy_cycles  = 0;
    fill_port_busy_cycles  = 0;
    n_bypassed = 0;
  }
  cache_sub_stats &operator+=(const cache_sub_stats &css) {
    ///
    /// Overloading += operator to easily accumulate stats
    ///
    accesses += css.accesses;
    reads += css.reads;
    writes += css.writes;
    misses += css.misses;
    rd_misses += css.rd_misses;
    wr_misses += css.wr_misses;
    sector_misses += css.sector_misses;
    sector_rd_misses += css.sector_rd_misses;
    sector_wr_misses += css.sector_wr_misses;
    pending_hits += css.pending_hits;
    res_fails += css.res_fails;
    avg_evict_interval     = (avg_evict_interval + css.avg_evict_interval) >> 1;
    avg_rd_byp_activates   = (avg_rd_byp_activates + css.avg_rd_byp_activates) >> 1;
    avg_rd_byp_deactivates = (avg_rd_byp_deactivates + css.avg_rd_byp_deactivates) >> 1;
    port_available_cycles += css.port_available_cycles;
    data_port_busy_cycles += css.data_port_busy_cycles;
    fill_port_busy_cycles += css.fill_port_busy_cycles;
    n_bypassed += css.n_bypassed;
    return *this;
  }

  cache_sub_stats operator+(const cache_sub_stats &cs) {
    ///
    /// Overloading + operator to easily accumulate stats
    ///
    cache_sub_stats ret;
    ret.accesses = accesses + cs.accesses;
    ret.reads = reads + cs.reads;
    ret.writes = writes + cs.writes;
    ret.misses = misses + cs.misses;
    ret.rd_misses = rd_misses + cs.rd_misses;
    ret.wr_misses = wr_misses + cs.wr_misses;
    ret.sector_misses = sector_misses + cs.sector_misses;
    ret.pending_hits = pending_hits + cs.pending_hits;
    ret.res_fails = res_fails + cs.res_fails;
    ret.avg_evict_interval     = (avg_evict_interval + cs.avg_evict_interval) >> 1;
    ret.avg_rd_byp_activates   = (avg_rd_byp_activates + cs.avg_rd_byp_activates) >> 1;
    ret.avg_rd_byp_deactivates = (avg_rd_byp_deactivates + cs.avg_rd_byp_deactivates) >> 1;
    ret.port_available_cycles =
        port_available_cycles + cs.port_available_cycles;
    ret.data_port_busy_cycles =
        data_port_busy_cycles + cs.data_port_busy_cycles;
    ret.fill_port_busy_cycles =
        fill_port_busy_cycles + cs.fill_port_busy_cycles;
    ret.n_bypassed = n_bypassed + cs.n_bypassed;
    return ret;
  }

  void print_port_stats(FILE *fout, const char *cache_name) const;
};

// Used for collecting AerialVision per-window statistics
struct cache_sub_stats_pw {
  unsigned accesses;
  unsigned write_misses;
  unsigned write_hits;
  unsigned write_pending_hits;
  unsigned write_res_fails;

  unsigned read_misses;
  unsigned read_hits;
  unsigned read_pending_hits;
  unsigned read_res_fails;

  cache_sub_stats_pw() { clear(); }
  void clear() {
    accesses = 0;
    write_misses = 0;
    write_hits = 0;
    write_pending_hits = 0;
    write_res_fails = 0;
    read_misses = 0;
    read_hits = 0;
    read_pending_hits = 0;
    read_res_fails = 0;
  }
  cache_sub_stats_pw &operator+=(const cache_sub_stats_pw &css) {
    ///
    /// Overloading += operator to easily accumulate stats
    ///
    accesses += css.accesses;
    write_misses += css.write_misses;
    read_misses += css.read_misses;
    write_pending_hits += css.write_pending_hits;
    read_pending_hits += css.read_pending_hits;
    write_res_fails += css.write_res_fails;
    read_res_fails += css.read_res_fails;
    return *this;
  }

  cache_sub_stats_pw operator+(const cache_sub_stats_pw &cs) {
    ///
    /// Overloading + operator to easily accumulate stats
    ///
    cache_sub_stats_pw ret;
    ret.accesses = accesses + cs.accesses;
    ret.write_misses = write_misses + cs.write_misses;
    ret.read_misses = read_misses + cs.read_misses;
    ret.write_pending_hits = write_pending_hits + cs.write_pending_hits;
    ret.read_pending_hits = read_pending_hits + cs.read_pending_hits;
    ret.write_res_fails = write_res_fails + cs.write_res_fails;
    ret.read_res_fails = read_res_fails + cs.read_res_fails;
    return ret;
  }
};

///
/// Cache_stats
/// Used to record statistics for each cache.
/// Maintains a record of every 'mem_access_type' and its resulting
/// 'cache_request_status' : [mem_access_type][cache_request_status]
///
typedef unsigned long long u64;
typedef unsigned u32;

// per kernel independent
class cache_stats {
 public:
  cache_stats();
  void clear();
  // Clear AerialVision cache stats after each window
  void clear_pw();
  u32 get_mshr_merge_dist_cnt(u64 streamID, u32 sm_id, u32 warp_id);

  void overall_average_l1d_rd_fill_to_evict_gap(u64 streamID, u64 served_cycles);  
  void avg_l1d_rd_miss_served_cycles(u64 streamID, u64 served_cycles);
  void avg_l1d_wr_miss_served_cycles(u64 streamID, u64 served_cycles);
  
  void inc_l1d_accesses(u64 streamID, u32 kernel);
  void inc_l1d_reads(u64 streamID, u32 kernel);
  void inc_l1d_rd_misses(u64 streamID, u32 kernel);
  void inc_l1d_writes(u64 streamID, u32 kernel);
  void inc_l1d_wr_misses(u64 streamID, u32 kernel);
  void update_l1d_max_evictions(const LOCALITY_KEY& loc_key, u32 n_evictions);
  void update_l1d_avg_evictions(const LOCALITY_KEY& loc_key, u32 n_evictions);
  void update_n_l1d_fill_to_evict(const LOCALITY_KEY& loc_key, u32 n_lines);
  
  void inc_mshr_stats(u64 streamID, u32 sm_id, u32 warp_id);
  void inc_accu_l2_dram_queue_size(u64 streamID, u32 l2_sub, u32 size);
  void inc_accu_l2_icnt_queue_size(u64 streamID, u32 l2_sub, u32 size);  
  void inc_l2_dram_q_accesses(u64 streamID, u32 l2_sub);
  void inc_l2_icnt_q_accesses(u64 streamID, u32 l2_sub);
  void inc_l2_mshr_slots_fills(u64 streamID, u32 l2_sub);
  void inc_l2_sub_miss_served_cycles(
    u64 streamID, u32 l2_sub,
    u64 served_cycles);
  void inc_l2_sub_misses(u64 streamID, u32 l2_sub);

  void inc_l2_miss_q_pops();
  void gather_lines_stats(u64 streamID, u32 unfolded_index);
  void inc_stats(int access_type, int access_outcome, u64 streamID);  
  void update_evict_stats(u64 streamID, u64 victim_avg_evict_interval);
  // Increment AerialVision cache stats
  void inc_stats_pw(int access_type, int access_outcome, u64 streamID);
  void inc_fail_stats(int access_type, int fail_outcome,
                      u64 streamID, int fail_driver = -1);
  enum cache_request_status select_stats_status(
      enum cache_request_status probe, enum cache_request_status access) const;
  u64 &operator()(int access_type, int access_outcome,
                                 bool fail_outcome,
                                 u64 streamID);
  u64 operator()(int access_type, int access_outcome,
                                bool fail_outcome,
                                u64 streamID) const;
  // for m_mshr_occupancy_stats
  u64 operator()(u32 sm, u32 warp, u64 streamID) const;

  // for m_accu_l2_dram_queue_size, m_accu_l2_icnt_queue_size, m_l2_dram_q_accesses
  u32 operator()(u32 l2_sub, u64 streamID) const;

  // l1d
  u64 operator() (u64 streamID, const char* tgt_name) const;
  u32 getU32(u64 streamID, u32 kernel, const char* tgt_name) const;

  u64 operator()(u32 l2_sub, u64 streamID, const char* tgt_name) const;

  u64 operator()(int access_type, int access_outcome,
                                bool is_fail_outcome,
                                int fail_driver,
                                u64 streamID) const;

  cache_stats operator+(const cache_stats &cs);
  cache_stats &operator+=(const cache_stats &cs);
  void print_stats(FILE *fout, u64 streamID,
                   const char *cache_info = "Cache_stats") const;
  void print_fail_stats(FILE *fout, u64 streamID,
                        const char *cache_info = "Cache_fail_stats") const;
  void print_mshr_stats(FILE *fout, u64 streamID,
                        const char *cache_info = "mshr_stats") const;
  void print_l2_dram_queue_stats(
    FILE *fout, u32 l2_dram_q_capacity, u64 streamID, const char *info = "") const;
  void print_l2_icnt_queue_stats(
    FILE *fout, u32 l2_icnt_q_capacity, u64 streamID, const char *info = "") const;

  void print_l1d_accesses(FILE* fout, u64 streamID, u32 kernel) const;
  void print_l1d_wr_misses(FILE* fout, u64 streamID, u32 kernel) const;
  void print_l1d_writes(FILE* fout, u64 streamID, u32 kernel) const;
  u32 print_l1d_rd_misses(FILE* fout, u64 streamID, u32 kernel, u64 cycle = (u64) - 1) const;
  u32 print_l1d_reads(FILE* fout, u64 streamID, u32 kernel, u64 cycle = (u64) - 1) const;
  void print_l1d_rd_miss_rate(
    FILE* fout, u64 streamID, u32 kernel, u32 misses, u32 reads, u64 cycles = (u64) - 1) const;

  u32 print_l1d_max_evictions(FILE* fout, LOCALITY_KEY& loc_key, u64 cycles) const;
  u32 print_l1d_avg_evictions(FILE* fout, LOCALITY_KEY& loc_key, u64 cycles) const;

  void print_l1d_n_fill_to_evict_lines(FILE* fout, u64 streamID, u32 kernel) const;
  void print_l1d_avg_rd_byp_activates(FILE* fout, u64 streamID, u32 kernel) const;
  void print_l1d_avg_rd_byp_deactivates(FILE* fout, u64 streamID, u32 kernel) const;
  void print_l1d_avg_rd_byp_act_rate(FILE* fout, u64 streamID, u32 kernel) const;
  void print_avg_core_cache_miss_served_cycles(FILE* fout, u64 streamID, u32 kernel) const;
  void print_avg_l1d_rd_fill_to_evict_gap(FILE* fout, u64 streamID) const;
  void print_avg_l1d_rd_miss_served_cycles(FILE* fout, u64 streamID) const;  
  void print_avg_l1d_wr_miss_served_cycles(FILE* fout, u64 streamID) const;
  void print_avg_l2_miss_served_cycles(FILE* fout, u64 streamID) const;
  void print_l2_mshr_slots_stats(
    FILE *fout, u32 l2_mshr_allocated_slots, 
    u64 streamID, const char *info) const;

  void print_l2_miss_q_pops(FILE *fout, const char *info = "") const;

  u64 get_stats(enum mem_access_type *access_type,
                               u32 num_access_type,
                               enum cache_request_status *access_status,
                               u32 num_access_status) const;
  void get_sub_stats(
    struct cache_sub_stats &css, const char* cache_name, 
    unsigned long long time,
    unsigned kernel = ((unsigned) - 1)) const;

  // Get per-window cache stats for AerialVision
  void get_sub_stats_pw(struct cache_sub_stats_pw &css) const;

  void sample_cache_port_utility(bool data_port_busy, bool fill_port_busy);

  void set_cache_name(const char* cache_name) { m_cache_name = cache_name; }
  const char* get_cache_name() const { return m_cache_name; }  
  void set_sub_partition(const u32 sub_partition) { m_sub_partition = sub_partition; }
  void set_sub_partitions(const u32 sub_partitions) { m_sub_partitions = sub_partitions; }
  void set_l2_dram_queue_capacity(const u32 capacity) { m_l2_dram_queue_capacity = capacity; }
  void set_l2_icnt_queue_capacity(const u32 capacity) { m_l2_icnt_queue_capacity = capacity; }
  const u32 get_l2_dram_queue_capacity() const { return m_l2_dram_queue_capacity; }
  const u32 get_l2_icnt_queue_capacity() const { return m_l2_icnt_queue_capacity; }
  const u32 get_sub_partition() const { return m_sub_partition; }  
  const u32 get_sub_partitions() const { return m_sub_partitions; }

  void update_l1d_rd_byp_act(bool en, u64 block_addr, u32 activates, u64 streamID);
  void update_l1d_rd_byp_deact(bool en, u64 block_addr, u32 deactivates, u64 streamID);

  u64 get_overall_avg_l1d_rd_fill_to_evict_gap(u64 streamID) const { return m_overall_avg_l1d_rd_fill_to_evict_gap.at(streamID); }
  u64 get_l1d_rd_miss_served_cycles(u64 streamID) const { return m_l1d_rd_miss_served_cycles.at(streamID); }
  u64 get_l1d_wr_miss_served_cycles(u64 streamID) const { return m_l1d_wr_miss_served_cycles.at(streamID); }

  u32 get_l1d_accesses(u64 streamID, u32 kernel) const {
    auto stream_it = m_l1d_accesses.find(streamID);
    if (stream_it == m_l1d_accesses.end()) {
      return 0;
    } else {
      std::map<u32 /* kernel */, u32> record = stream_it->second;
      auto kernel_it = record.find(kernel);
      if (kernel_it == record.end()) {
        return 0;
      } else {
        return record[kernel];
      }
    }
  }
  u32 get_l1d_reads(u64 streamID, u32 kernel) const {
    auto stream_it = m_l1d_reads.find(streamID);
    if (stream_it == m_l1d_reads.end()) {
      return 0;
    } else {
      std::map<u32 /* kernel */, u32> record = stream_it->second;
      auto kernel_it = record.find(kernel);
      if (kernel_it == record.end()) {
        return 0;
      } else {
        return record[kernel];
      }
    }
  }
  u32 get_l1d_writes(u64 streamID, u32 kernel) const {
    auto stream_it = m_l1d_writes.find(streamID);
    if (stream_it == m_l1d_writes.end()) {
      return 0;
    } else {
      std::map<u32 /* kernel */, u32> record = stream_it->second;
      auto kernel_it = record.find(kernel);
      if (kernel_it == record.end()) {
        return 0;
      } else {
        return record[kernel];
      }
    }
  }
  u32 get_l1d_rd_misses(u64 streamID, u32 kernel) const {
    auto stream_it = m_l1d_rd_misses.find(streamID);
    if (stream_it == m_l1d_rd_misses.end()) {
      return 0;
    } else {
      std::map<u32 /* kernel */, u32> record = stream_it->second;
      auto kernel_it = record.find(kernel);
      if (kernel_it == record.end()) {
        return 0;
      } else {
        return record[kernel];
      }
    }    
  }
  u32 get_l1d_wr_misses(u64 streamID, u32 kernel) const {
    auto stream_it = m_l1d_wr_misses.find(streamID);
    if (stream_it == m_l1d_wr_misses.end()) {
      return 0;
    } else {
      std::map<u32 /* kernel */, u32> record = stream_it->second;
      auto kernel_it = record.find(kernel);
      if (kernel_it == record.end()) {
        return 0;
      } else {
        return record[kernel];
      }
    }     
  }

 private:
  const char* m_cache_name;
  u32 m_sub_partition;
  u32 m_sub_partitions;
  u32 m_l2_dram_queue_capacity;
  u32 m_l2_icnt_queue_capacity;
  bool check_valid(int type, int status) const;
  bool check_valid(u32 sm, u32 warp) const;
  bool check_fail_valid(int type, int fail) const;

  void accu_single_stat(const char* tgt_item, const cache_stats &cs);

  // CUDA streamID -> cache stats[NUM_MEM_ACCESS_TYPE]
  std::map<u64, std::vector<std::vector<u64>>> m_stats;
  std::map<u64, u32 /* avg_evict_interval */> m_evict_stats;
  std::map<u64 /* streamID */, u64> m_overall_avg_l1d_rd_fill_to_evict_gap;
  std::map<u64 /* streamID */, u64> m_l1d_rd_miss_served_cycles; // done accu
  std::map<u64 /* streamID */, u64> m_l1d_wr_miss_served_cycles; // done accu

  std::map<LOCALITY_KEY, u32> m_n_l1d_fill_to_evict_lines;
  std::map<u64 /* streamID */, std::map<u64, u32>> m_l1d_rd_byp_activates; // done accu
  std::map<u64 /* streamID */, std::map<u64, u32>> m_l1d_rd_byp_deactivates; // done accu
  // SM is not differentiated in following stats
  std::map<u64 /* streamID */, std::map<u32 /* kernel */, u32>> m_l1d_accesses;  // done accu
  std::map<u64 /* streamID */, std::map<u32 /* kernel */, u32>> m_l1d_misses;    // done accu
  std::map<u64 /* streamID */, std::map<u32 /* kernel */, u32>> m_l1d_reads;     // done accu
  std::map<u64 /* streamID */, std::map<u32 /* kernel */, u32>> m_l1d_writes;    // done accu 
  std::map<u64 /* streamID */, std::map<u32 /* kernel */, u32>> m_l1d_rd_misses; // done accu
  std::map<u64 /* streamID */, std::map<u32 /* kernel */, u32>> m_l1d_wr_misses; // done accu
  std::map<LOCALITY_KEY, u32> m_l1d_max_evictions;
  std::map<LOCALITY_KEY, u32> m_l1d_avg_evictions;

  std::map<u64 /* streamID */, std::vector<u64>> m_l2_sub_miss_served_cycles;
  std::map<u64 /* streamID */, std::vector<u32>> m_l2_sub_misses;

  // AerialVision cache stats (per-window)
  std::map<u64, std::vector<std::vector<u64>>> m_stats_pw;
  std::map<u64, std::vector<std::vector<u64>>> m_fail_stats;
  std::map<u64, std::vector<std::vector<u64>>> m_line_alloc_fail;
  std::map<u64, std::vector<std::vector<u64>>> m_mshr_entry_fail;
  std::map<u64, std::vector<std::vector<u64>>> m_miss_q_full;
  std::map<u64, std::vector<std::vector<u64>>> m_mshr_merge_entry_fail;
  std::map<u64, std::vector<u64>> m_fail_stats_total;

  std::map<u64 /* streamID */, std::vector<u32 /* unfolded_index */>> m_lines_evictons;
  std::map<u64 /* streamID */, std::vector<u32 /* unfolded_index */>> m_lines_accesses;

  std::map<u64 /* streamID */, 
    std::vector< /* SMs */std::vector<u32 /* WARPs per SM */>>> m_mshr_occupancy_stats;
  std::map<u64 /* streamID */, std::vector<u32 /* L2 Sub */>> m_accu_l2_dram_queue_size;
  std::map<u64 /* streamID */, std::vector<u32 /* L2 Sub */>> m_accu_l2_icnt_queue_size;
  std::map<u64 /* streamID */, std::vector<u32 /* L2 Sub */>> m_l2_dram_q_accesses;
  std::map<u64 /* streamID */, std::vector<u32 /* L2 Sub */>> m_l2_icnt_q_accesses;
  std::map<u64 /* streamID */, std::vector<u32 /* L2 Sub */>> m_l2_mshr_slots_fills;
  u64 m_l2_miss_q_pops;

  u64 m_cache_port_available_cycles;
  u64 m_cache_data_port_busy_cycles;
  u64 m_cache_fill_port_busy_cycles;  
};

class cache_t {
 public:
  virtual ~cache_t() {}
  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned long long time,
                                           std::list<cache_event> &events) = 0;

  // accessors for cache bandwidth availability
  virtual bool data_port_free() const = 0;
  virtual bool fill_port_free() const = 0;
};

bool was_write_sent(const std::list<cache_event> &events);
bool was_read_sent(const std::list<cache_event> &events);
bool was_writeallocate_sent(const std::list<cache_event> &events);

/// Baseline cache
/// Implements common functions for read_only_cache and data_cache
/// Each subclass implements its own 'access' function
class baseline_cache : public cache_t {
  friend class tag_array;
  friend class tex_cache;
  friend class shader_core_ctx;
  friend class ldst_unit;
  friend class memory_sub_partition;
  /// Sub-class containing all metadata for port bandwidth management
  class bandwidth_management {
    public:
      bandwidth_management(cache_config &config);

      /// use the data port based on the outcome and events generated by the
      /// mem_fetch request
      void use_data_port(mem_fetch *mf, enum cache_request_status outcome,
                        const std::list<cache_event> &events, bool is_wr = false);

      /// use the fill port
      void use_fill_port(mem_fetch *mf);

      /// called every cache cycle to free up the ports
      void replenish_port_bandwidth();

      /// query for data port availability
      bool data_port_free() const;
      /// query for fill port availability
      bool fill_port_free() const;

    protected:
      const cache_config &m_config;

      int m_data_port_occupied_cycles;  //< Number of cycle that the data port
                                        // remains used
      int m_fill_port_occupied_cycles;  //< Number of cycle that the fill port
                                        // remains used
  };

 public:
  baseline_cache(const char *name, cache_config &config, int core_id,
                 int type_id, mem_fetch_interface *memport,
                 enum mem_fetch_status status, enum cache_gpu_level level,
                 gpgpu_sim *gpu)
      : m_config(config),
        m_tag_array(new tag_array(gpu, config, core_id, type_id)),
        m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge),        
        m_level(level),
        m_gpu(gpu),
        m_bandwidth_management(config) {
    // for debug
    printf("baseline_cache %s {mshrs_entries = %u, mshr_max_merge = %u}\n", 
      config.get_cache_name(), config.m_mshr_entries, config.m_mshr_max_merge);

    init(name, config, memport, status);
  }

  void init(const char *name, const cache_config &config,
            mem_fetch_interface *memport, enum mem_fetch_status status) {
    m_name   = name;
    m_is_l1d = false;
    m_is_l2  = false;
    if (!strcmp(config.get_cache_name(), "L1D")) {
      m_is_l1d = true;        
    } else if (!strcmp(config.get_cache_name(), "L2")) {
      m_is_l2 = true;
    }
    m_tag_array->m_is_l1d = m_is_l1d;
    m_tag_array->m_is_l2  = m_is_l2;
    fprintf(Trace::out, "init cache: %s\n", m_name.c_str());

    m_l1d_rd_miss_addresses.clear();

    assert(config.m_mshr_type == ASSOC || config.m_mshr_type == SECTOR_ASSOC);
    m_memport = memport;
    m_miss_queue_status = status;
    m_lfb.clear();
  }

  virtual ~baseline_cache() { delete m_tag_array; }

  void update_cache_parameters(cache_config &config) {
    m_config = config;
    m_tag_array->update_cache_parameters(config);
    m_mshrs.check_mshr_parameters(config.m_mshr_entries,
                                  config.m_mshr_max_merge);
  }

  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned long long time,
                                           std::list<cache_event> &events) = 0;
  /// Sends next request to lower level of memory (return)
  void cycle();
  /// Interface for response from lower memory level (model bandwidth
  /// restictions in caller)
  void fill(mem_fetch *mf, unsigned long long time);
  /// Checks if mf is waiting to be filled by lower memory level
  bool waiting_for_fill(mem_fetch *mf);

  /// Are any (accepted) accesses that had to wait for memory now ready? (does
  /// not include accesses that "HIT")
  virtual bool access_ready() const {
    if (m_config.m_mshr_disable == 'T') {    
      return !m_lfb.empty();
    } else {
      return m_mshrs.access_ready();
    }
  }
  /// 
  size_t num_pending_responses() const {
    if (m_config.m_mshr_disable == 'T') {
      return m_lfb.size();  
    } else {
      return m_mshrs.num_pending_responses();  
    }
  }
  /// Pop next ready access (does not include accesses that "HIT")
  mem_fetch *next_access(const char* cache_name, unsigned long long cycle = 0) {
    if (m_config.m_mshr_disable == 'T') {
      (void)cache_name;
      (void)cycle;
      if (m_lfb.empty()) {
        return NULL;
      }
      mem_fetch *mf = m_lfb.front();
      m_lfb.pop_front();
      if (DTRACE(CACHE_REFILL_QUEUE)) {
        fprintf(Trace::out, "%llu %s_sub[%d] m_lfb added %#llx\n", 
          cycle, cache_name, mf->get_sub_partition(), mf->get_addr());
      }
      return mf;
    } else {
      return m_mshrs.next_access(cache_name, cycle);
    }
  }
  // flash invalidate all entries in cache
  void flush() { m_tag_array->flush(); }
  void invalidate() { m_tag_array->invalidate(); }
  void print(FILE *fp, unsigned &accesses, unsigned &misses) const;
  void display_state(FILE *fp) const;
  virtual void dumpCacheEvent(unsigned long long time, const char* stage, const char* event, mem_fetch *mf);
  virtual void dumpMSHREvent(unsigned long long time, mem_fetch *mf, new_addr_type mshr_addr, bool is_new_entry);
  virtual void dumpMissQueue(unsigned long long time, const char* stage, const char* event, mem_fetch *mf);

  // Stat collection
  const cache_stats &get_stats() const { return m_stats; }
  unsigned get_stats(enum mem_access_type *access_type,
                     unsigned num_access_type,
                     enum cache_request_status *access_status,
                     unsigned num_access_status) const {
    return m_stats.get_stats(access_type, num_access_type, access_status,
                             num_access_status);
  }
  void get_sub_stats(
    struct cache_sub_stats &css, const char* cache_name, 
    unsigned long long time, unsigned kernel) const {
    m_stats.get_sub_stats(css, cache_name, time, kernel);
  }
  // Clear per-window stats for AerialVision support
  void clear_pw() { m_stats.clear_pw(); }
  // Per-window sub stats for AerialVision support
  void get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
    m_stats.get_sub_stats_pw(css);
  }

  // accessors for cache bandwidth availability
  bool data_port_free() const {
    return m_bandwidth_management.data_port_free();
  }
  bool fill_port_free() const {
    return m_bandwidth_management.fill_port_free();
  }
  void inc_aggregated_stats(cache_request_status status,
                            cache_request_status cache_status, mem_fetch *mf,
                            enum cache_gpu_level level);
  void inc_aggregated_fail_stats(cache_request_status status,
                                 cache_request_status cache_status,
                                 mem_fetch *mf, enum cache_gpu_level level);
  void inc_aggregated_stats_pw(cache_request_status status,
                               cache_request_status cache_status, mem_fetch *mf,
                               enum cache_gpu_level level);

  // This is a gapping hole we are poking in the system to quickly handle
  // filling the cache on cudamemcopies. We don't care about anything other than
  // L2 state after the memcopy - so just force the tag array to act as though
  // something is read or written without doing anything else.
  void force_tag_access(new_addr_type addr, unsigned long long time,
                        mem_access_sector_mask_t mask) {
    mem_access_byte_mask_t byte_mask;
    m_tag_array->fill(addr, time, mask, byte_mask, true);
  }

 protected:
  // Constructor that can be used by derived classes with custom tag arrays
  baseline_cache(const char *name, cache_config &config, int core_id,
                 int type_id, mem_fetch_interface *memport,
                 enum mem_fetch_status status, tag_array *new_tag_array)
      : m_config(config),
        m_tag_array(new_tag_array),
        m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge),
        m_bandwidth_management(config) {
    init(name, config, memport, status);
  }

 protected:  
  std::string m_name;
  bool m_is_l1d;
  bool m_is_l2;
  std::vector<new_addr_type> m_l1d_rd_miss_addresses;
  cache_config &m_config;
  tag_array *m_tag_array;
  mshr_table m_mshrs;  
  std::list<mem_fetch *> m_miss_queue;
  enum mem_fetch_status m_miss_queue_status;
  mem_fetch_interface *m_memport;
  cache_gpu_level m_level;
  gpgpu_sim *m_gpu;

  struct extra_mf_fields {
    extra_mf_fields() { m_valid = false; }
    extra_mf_fields(
      new_addr_type a /* mshr_addr, i.e., sector_addr */, 
      new_addr_type ad /* mf->get_addr() */, 
      unsigned i, unsigned d,
      const cache_config &m_config,
      bool l1d_bypass_noalloc) { // case can pass ?

      m_valid = true;
      m_block_addr = a;
      m_addr = ad;
      m_cache_index = i;
      m_data_size = d;
      pending_read = m_config.m_mshr_type == SECTOR_ASSOC
                         ? m_config.m_line_sz / SECTOR_SIZE
                         : 0;
      m_l1d_bypass_noalloc = l1d_bypass_noalloc;
    }
    bool m_valid;
    new_addr_type m_block_addr;
    new_addr_type m_addr;
    unsigned m_cache_index;
    unsigned m_data_size;
    // this variable is used when a load request generates multiple load
    // transactions For example, a read request from non-sector L1 request sends
    // a request to sector L2
    unsigned pending_read;
    bool m_l1d_bypass_noalloc;
  };

  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;

  extra_mf_fields_lookup m_extra_mf_fields;

  cache_stats m_stats;

  // Line Fill Buffer (LFB) to buffer response from downstream when MSHR is disabled
  std::list<mem_fetch *> m_lfb;
  
  std::list<cache_block_t* > m_victim_cache; // Assume perfect

  /// Checks whether this request can be handled on this cycle. num_miss equals
  /// max # of misses to be handled on this cycle
  bool miss_queue_full(unsigned num_miss, std::string caller = nullptr) {
    if (DTRACE(CACHE_STALLED)) {
      std::string cache_name = m_is_l1d ? "L1D" : (m_is_l2 ? "L2" : "other L1");
      fprintf(Trace::out, "Inside %s, miss_queue_full at %s\n", 
        caller.c_str(), cache_name.c_str());
    }
    return ((m_miss_queue.size() + num_miss) >= m_config.m_miss_queue_size);
  }
  /// Read miss handler without writeback
  void send_read_request(new_addr_type raw_addr, new_addr_type block_addr,
                         unsigned cache_index, mem_fetch *mf, unsigned long long time,
                         bool &do_miss, std::list<cache_event> &events,
                         bool read_only, bool wa);
  /// Read miss handler. Check MSHR hit or MSHR available
  void send_read_request(new_addr_type raw_addr, new_addr_type block_addr,
                         unsigned cache_index, mem_fetch *mf, unsigned long long time,
                         bool &do_miss, bool &wb, evicted_block_info &evicted,
                         std::list<cache_event> &events, bool read_only,
                         bool wa);

  virtual void dump_cache_access_info(
    const char* caller,
    new_addr_type addr, mem_fetch *mf, unsigned long long time, 
    enum cache_request_status status,
    bool dump_inst_str = false);                     

  virtual void dump_cache_fill_info(
    const char* caller,
    new_addr_type addr, mem_fetch *mf, unsigned long long time, 
    bool dump_inst_str = false);     

  bandwidth_management m_bandwidth_management;
};

/// Read only cache
class read_only_cache : public baseline_cache {
 public:
  read_only_cache(const char *name, cache_config &config, int core_id,
                  int type_id, mem_fetch_interface *memport,
                  enum mem_fetch_status status, enum cache_gpu_level level,
                  gpgpu_sim *gpu)
      : baseline_cache(name, config, core_id, type_id, memport, status, level,
                       gpu) {}

  /// Access cache for read_only_cache: returns RESERVATION_FAIL if request
  /// could not be accepted (for any reason)
  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned long long time,
                                           std::list<cache_event> &events);

  virtual ~read_only_cache() {}

 protected:
  read_only_cache(gpgpu_sim *gpu, const char *name, cache_config &config, int core_id,
                  int type_id, mem_fetch_interface *memport,
                  enum mem_fetch_status status, tag_array *new_tag_array)
      : baseline_cache(name, config, core_id, type_id, memport, status,
                       new_tag_array) {}
};

/// Data cache - Implements common functions for L1 and L2 data cache
class data_cache : public baseline_cache {
 public:
  data_cache(const char *name, cache_config &config, int core_id, int type_id,
             mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
             enum mem_fetch_status status, mem_access_type wr_alloc_type,
             mem_access_type wrbk_type, class gpgpu_sim *gpu,
             enum cache_gpu_level level)
      : baseline_cache(name, config, core_id, type_id, memport, status, level,
                       gpu) {
    init(mfcreator);
    m_wr_alloc_type = wr_alloc_type;
    m_wrbk_type = wrbk_type;
    m_gpu = gpu;
  }

  virtual ~data_cache() {}

  virtual void init(mem_fetch_allocator *mfcreator) {
    m_memfetch_creator = mfcreator;

    // Set read hit function
    m_rd_hit = &data_cache::rd_hit_base;

    // Set read miss function
    m_rd_miss = &data_cache::rd_miss_base;

    // Set write hit function
    switch (m_config.m_write_policy) {
      // READ_ONLY is now a separate cache class, config is deprecated
      case READ_ONLY:
        assert(0 && "Error: Writable Data_cache set as READ_ONLY\n");
        break;
      case WRITE_BACK:
        m_wr_hit = &data_cache::wr_hit_wb;
        break;
      case WRITE_THROUGH:
        m_wr_hit = &data_cache::wr_hit_wt;
        break;
      case WRITE_EVICT:
        m_wr_hit = &data_cache::wr_hit_we;
        break;
      case LOCAL_WB_GLOBAL_WE:
        m_wr_hit = &data_cache::wr_hit_global_we_local_wb;
        break;
      default:
        assert(0 && "Error: Must set valid cache write policy\n");
        break;  // Need to set a write hit function
    }

    // Set write miss function
    switch (m_config.m_write_alloc_policy) {
      case NO_WRITE_ALLOCATE:
        m_wr_miss = &data_cache::wr_miss_no_wa;
        break;
      case WRITE_ALLOCATE:
        m_wr_miss = &data_cache::wr_miss_wa_naive;
        break;
      case FETCH_ON_WRITE:
        m_wr_miss = &data_cache::wr_miss_wa_fetch_on_write;
        break;
      case LAZY_FETCH_ON_READ:
        m_wr_miss = &data_cache::wr_miss_wa_lazy_fetch_on_read;
        break;
      default:
        assert(0 && "Error: Must set valid cache write miss policy\n");
        break;  // Need to set a write miss function
    }
  }

  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned long long time,
                                           std::list<cache_event> &events);

 protected:
  data_cache(const char *name, cache_config &config, int core_id, int type_id,
             mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
             enum mem_fetch_status status, tag_array *new_tag_array,
             mem_access_type wr_alloc_type, mem_access_type wrbk_type,
             class gpgpu_sim *gpu)
      : baseline_cache(name, config, core_id, type_id, memport, status,
                       new_tag_array) {
    init(mfcreator);
    m_wr_alloc_type = wr_alloc_type;
    m_wrbk_type = wrbk_type;
    m_gpu = gpu;
  }

  mem_access_type m_wr_alloc_type; // Specifies type of write allocate request (e.g., L1 or L2)
  mem_access_type m_wrbk_type;     // Specifies type of writeback request (e.g., L1 or L2)
  class gpgpu_sim *m_gpu;

  //! A general function that takes the result of a tag_array probe
  //  and performs the correspding functions based on the cache configuration
  //  The access fucntion calls this function
  enum cache_request_status process_tag_probe(bool wr,
                                              enum cache_request_status status,
                                              new_addr_type addr /* raw_addr */,
                                              unsigned cache_index,
                                              mem_fetch *mf, unsigned long long time,
                                              std::list<cache_event> &events);

 protected:
  mem_fetch_allocator *m_memfetch_creator;

  // Functions for data cache access
  /// Sends write request to lower level memory (write or writeback)
  void send_write_request(
    std::string caller, mem_fetch *mf, 
    cache_event request, unsigned long long time, std::list<cache_event> &events);
  void update_m_readable(mem_fetch *mf, unsigned cache_index);
  // Member Function pointers - Set by configuration options
  // to the functions below each grouping
  /******* Write-hit configs *******/
  enum cache_request_status (data_cache::*m_wr_hit)(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned long long time,
      std::list<cache_event> &events, enum cache_request_status status);
  /// Marks block as MODIFIED and updates block LRU
  enum cache_request_status wr_hit_wb(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned long long time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-back
  enum cache_request_status wr_hit_wt(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned long long time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-through

  /// Marks block as INVALID and sends write request to lower level memory
  enum cache_request_status wr_hit_we(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned long long time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-evict
  enum cache_request_status wr_hit_global_we_local_wb(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned long long time,
      std::list<cache_event> &events, enum cache_request_status status);
  // global write-evict, local write-back

  /******* Write-miss configs *******/
  enum cache_request_status (data_cache::*m_wr_miss)(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned long long time,
      std::list<cache_event> &events, enum cache_request_status status);
  /// Sends read request, and possible write-back request,
  //  to lower level memory for a write miss with write-allocate
  enum cache_request_status wr_miss_wa_naive(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned long long time,
      std::list<cache_event> &events,
      enum cache_request_status
          status);  // write-allocate-send-write-and-read-request
  enum cache_request_status wr_miss_wa_fetch_on_write(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned long long time,
      std::list<cache_event> &events,
      enum cache_request_status
          status);  // write-allocate with fetch-on-every-write
  enum cache_request_status wr_miss_wa_lazy_fetch_on_read(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned long long time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-allocate with read-fetch-only
  enum cache_request_status wr_miss_wa_write_validate(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned long long time,
      std::list<cache_event> &events,
      enum cache_request_status
          status);  // write-allocate that writes with no read fetch
  enum cache_request_status wr_miss_no_wa(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned long long time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // no write-allocate

  // Currently no separate functions for reads
  /******* Read-hit configs *******/
  enum cache_request_status (data_cache::*m_rd_hit)(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, 
      unsigned long long time,
      std::list<cache_event> &events, enum cache_request_status status);
  enum cache_request_status rd_hit_base(new_addr_type addr /* raw_addr */,
                                        unsigned cache_index, mem_fetch *mf,
                                        unsigned long long time,
                                        std::list<cache_event> &events,
                                        enum cache_request_status status);

  /******* Read-miss configs *******/
  enum cache_request_status (data_cache::*m_rd_miss)(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, 
      unsigned long long time,
      std::list<cache_event> &events, enum cache_request_status status);
  enum cache_request_status rd_miss_base(new_addr_type addr,
                                         unsigned cache_index, mem_fetch *mf,
                                         unsigned long long time,
                                         std::list<cache_event> &events,
                                         enum cache_request_status status);
};

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at
/// the granularity of individual blocks
/// (the policy used in fermi according to the CUDA manual)
class l1_cache : public data_cache {
 public:
  l1_cache(const char *name, cache_config &config, 
          int core_id, 
          int type_id,
           mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
           enum mem_fetch_status status, class gpgpu_sim *gpu,
           enum cache_gpu_level level)
      : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                   L1_WR_ALLOC_R, L1_WRBK_ACC, gpu, level) {}

  virtual ~l1_cache() {}

  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned long long time,
                                           std::list<cache_event> &events);

 protected:
  l1_cache(const char *name, cache_config &config, int core_id, int type_id,
           mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
           enum mem_fetch_status status, tag_array *new_tag_array,
           class gpgpu_sim *gpu)
      : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                   new_tag_array, L1_WR_ALLOC_R, L1_WRBK_ACC, gpu) {}
};

/// Models second level shared cache with global write-back
/// and write-allocate policies
class l2_cache : public data_cache {
  friend class memory_sub_partition;
 public:
  l2_cache(const char *name, cache_config &config, int core_id, int type_id,
           mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
           enum mem_fetch_status status, class gpgpu_sim *gpu,
           enum cache_gpu_level level)
      : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                   L2_WR_ALLOC_R, L2_WRBK_ACC, gpu, level) {
  }

  virtual ~l2_cache() {}

  virtual enum cache_request_status access(
    new_addr_type addr, mem_fetch *mf, unsigned long long time,
    std::list<cache_event> &events);

  private:
  // unsigned getSubPartitionID() const { return m_sub_partition_id; }

  // protected:
  //   unsigned m_sub_partition_id;
  //   void setSubPartitionID(unsigned id) { m_sub_partition_id = id; }    
};

/*****************************************************************************/

// See the following paper to understand this cache model:
//
// Igehy, et al., Prefetching in a Texture Cache Architecture,
// Proceedings of the 1998 Eurographics/SIGGRAPH Workshop on Graphics Hardware
// http://www-graphics.stanford.edu/papers/texture_prefetch/
class tex_cache : public cache_t {
 public:
  tex_cache(gpgpu_sim *gpu, const char *name, cache_config &config, int core_id, int type_id,
            mem_fetch_interface *memport, enum mem_fetch_status request_status,
            enum mem_fetch_status rob_status)
      : m_gpu(gpu),
        m_config(config),
        m_tags(gpu, config, core_id, type_id),
        m_fragment_fifo(config.m_fragment_fifo_entries),
        m_request_fifo(config.m_request_fifo_entries),
        m_rob(config.m_rob_entries),
        m_result_fifo(config.m_result_fifo_entries) {
    m_name = name;
    assert(config.m_mshr_type == TEX_FIFO ||
           config.m_mshr_type == SECTOR_TEX_FIFO);
    assert(config.m_write_policy == READ_ONLY);
    assert(config.m_alloc_policy == ON_MISS);
    m_memport = memport;
    m_cache = new data_block[config.get_num_lines()];
    m_request_queue_status = request_status;
    m_rob_status = rob_status;
  }

  /// Access function for tex_cache
  /// return values: RESERVATION_FAIL if request could not be accepted
  /// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
  /// since unlike a normal CPU cache, a "HIT" in texture cache does not
  /// mean the data is ready (still need to get through fragment fifo)
  enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                   unsigned long long time,
                                   std::list<cache_event> &events);
  void cycle();
  /// Place returning cache block into reorder buffer
  void fill(mem_fetch *mf, unsigned time);
  /// Are any (accepted) accesses that had to wait for memory now ready? (does
  /// not include accesses that "HIT")
  bool access_ready() const { return !m_result_fifo.empty(); }
  /// Pop next ready access (includes both accesses that "HIT" and those that
  /// "MISS")
  mem_fetch *next_access() { return m_result_fifo.pop(); }
  void display_state(FILE *fp) const;

  // accessors for cache bandwidth availability - stubs for now
  bool data_port_free() const { return true; }
  bool fill_port_free() const { return true; }

  // Stat collection
  const cache_stats &get_stats() const { return m_stats; }
  unsigned get_stats(enum mem_access_type *access_type,
                     unsigned num_access_type,
                     enum cache_request_status *access_status,
                     unsigned num_access_status) const {
    return m_stats.get_stats(access_type, num_access_type, access_status,
                             num_access_status);
  }

  void get_sub_stats(
    struct cache_sub_stats &css, const char* cache_name, 
    unsigned long long time, unsigned kernel) const {
    m_stats.get_sub_stats(css, cache_name, time, kernel);
  }

 private:
  std::string m_name;
  gpgpu_sim *m_gpu;
  const cache_config &m_config;

  struct fragment_entry {
    fragment_entry() {}
    fragment_entry(mem_fetch *mf, unsigned idx, bool m, unsigned d) {
      m_request = mf;
      m_cache_index = idx;
      m_miss = m;
      m_data_size = d;
    }
    mem_fetch *m_request;    // request information
    unsigned m_cache_index;  // where to look for data
    bool m_miss;             // true if sent memory request
    unsigned m_data_size;
  };

  struct rob_entry {
    rob_entry() {
      m_ready = false;
      m_time = 0;
      m_request = NULL;
    }
    rob_entry(unsigned i, mem_fetch *mf, new_addr_type a) {
      m_ready = false;
      m_index = i;
      m_time = 0;
      m_request = mf;
      m_block_addr = a;
    }
    bool m_ready;
    unsigned m_time;   // which cycle did this entry become ready?
    unsigned m_index;  // where in cache should block be placed?
    mem_fetch *m_request;
    new_addr_type m_block_addr;
  };

  struct data_block {
    data_block() { m_valid = false; }
    bool m_valid;
    new_addr_type m_block_addr;
  };

  // TODO: replace fifo_pipeline with this?
  template <class T>
  class fifo {
   public:
    fifo(unsigned size) {
      m_size = size;
      m_num = 0;
      m_head = 0;
      m_tail = 0;
      m_data = new T[size];
    }
    bool full() const { return m_num == m_size; }
    bool empty() const { return m_num == 0; }
    unsigned size() const { return m_num; }
    unsigned capacity() const { return m_size; }
    unsigned push(const T &e) {
      assert(!full());
      m_data[m_head] = e;
      unsigned result = m_head;
      inc_head();
      return result;
    }
    T pop() {
      assert(!empty());
      T result = m_data[m_tail];
      inc_tail();
      return result;
    }
    const T &peek(unsigned index) const {
      assert(index < m_size);
      return m_data[index];
    }
    T &peek(unsigned index) {
      assert(index < m_size);
      return m_data[index];
    }
    T &peek() const { return m_data[m_tail]; }
    unsigned next_pop_index() const { return m_tail; }

   private:
    void inc_head() {
      m_head = (m_head + 1) % m_size;
      m_num++;
    }
    void inc_tail() {
      assert(m_num > 0);
      m_tail = (m_tail + 1) % m_size;
      m_num--;
    }

    unsigned m_head;  // next entry goes here
    unsigned m_tail;  // oldest entry found here
    unsigned m_num;   // how many in fifo?
    unsigned m_size;  // maximum number of entries in fifo
    T *m_data;
  };

  tag_array m_tags;
  fifo<fragment_entry> m_fragment_fifo;
  fifo<mem_fetch *> m_request_fifo;
  fifo<rob_entry> m_rob;
  data_block *m_cache;
  fifo<mem_fetch *> m_result_fifo;  // next completed texture fetch

  mem_fetch_interface *m_memport;
  enum mem_fetch_status m_request_queue_status;
  enum mem_fetch_status m_rob_status;

  struct extra_mf_fields {
    extra_mf_fields() { m_valid = false; }
    extra_mf_fields(unsigned i, const cache_config &m_config) {
      m_valid = true;
      m_rob_index = i;
      pending_read = m_config.m_mshr_type == SECTOR_TEX_FIFO
                         ? m_config.m_line_sz / SECTOR_SIZE
                         : 0;
    }
    bool m_valid;
    unsigned m_rob_index;
    unsigned pending_read;
  };

  cache_stats m_stats;

  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;

  extra_mf_fields_lookup m_extra_mf_fields;
};

#endif
