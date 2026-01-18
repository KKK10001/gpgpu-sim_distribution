libcudart.so.12!tag_array::probe(const tag_array * const this, new_addr_type addr, unsigned int & idx, mem_access_sector_mask_t mask, bool is_write, unsigned long long time, bool probe_mode, mem_fetch * mf) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:291)
libcudart.so.12!tag_array::probe(const tag_array * const this, new_addr_type addr, unsigned int & idx, mem_fetch * mf, bool is_write, unsigned long long time, bool probe_mode) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:246)
libcudart.so.12!data_cache::access(data_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:2001)
libcudart.so.12!l1_cache::access(l1_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:2050)
libcudart.so.12!ldst_unit::L1_latency_queue_cycle(ldst_unit * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:2356)
libcudart.so.12!ldst_unit::cycle(ldst_unit * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:3230)
libcudart.so.12!shader_core_ctx::execute(shader_core_ctx * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:1976)
libcudart.so.12!shader_core_ctx::cycle(shader_core_ctx * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:4009)
libcudart.so.12!simt_core_cluster::core_cycle(simt_core_cluster * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:4820)
libcudart.so.12!gpgpu_sim::cycle(gpgpu_sim * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.cc:2080)
libcudart.so.12!gpgpu_sim_thread_concurrent(void * ctx_ptr) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpusim_entrypoint.cc:146)
libpthread.so.0!start_thread(void * arg) (\build\glibc-B3wQXB\glibc-2.31\nptl\pthread_create.c:477)
libc.so.6!clone() (\build\glibc-B3wQXB\glibc-2.31\sysdeps\unix\sysv\linux\x86_64\clone.S:95)

libcudart.so.12!tag_array::probe(const tag_array * const this, new_addr_type addr, unsigned int & idx, mem_access_sector_mask_t mask, bool is_write, unsigned long long time, bool probe_mode, mem_fetch * mf) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:291)
libcudart.so.12!tag_array::probe(const tag_array * const this, new_addr_type addr, unsigned int & idx, mem_fetch * mf, bool is_write, unsigned long long time, bool probe_mode) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:246)
libcudart.so.12!tag_array::access(tag_array * const this, new_addr_type addr, unsigned int time, unsigned int & idx, bool & wb, evicted_block_info & evicted, mem_fetch * mf) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:366)
libcudart.so.12!data_cache::wr_miss_wa_lazy_fetch_on_read(data_cache * const this, new_addr_type addr, unsigned int cache_index, mem_fetch * mf, unsigned int time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events, cache_request_status status) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:1769)
libcudart.so.12!data_cache::process_tag_probe(data_cache * const this, bool wr, cache_request_status probe_status, new_addr_type addr, unsigned int cache_index, mem_fetch * mf, unsigned int time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:1961)
libcudart.so.12!data_cache::access(data_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:2003)
libcudart.so.12!l1_cache::access(l1_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:2050)
libcudart.so.12!ldst_unit::L1_latency_queue_cycle(ldst_unit * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:2356)
libcudart.so.12!ldst_unit::cycle(ldst_unit * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:3230)
libcudart.so.12!shader_core_ctx::execute(shader_core_ctx * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:1976)
libcudart.so.12!shader_core_ctx::cycle(shader_core_ctx * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:4009)
libcudart.so.12!simt_core_cluster::core_cycle(simt_core_cluster * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:4820)
libcudart.so.12!gpgpu_sim::cycle(gpgpu_sim * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.cc:2080)
libcudart.so.12!gpgpu_sim_thread_concurrent(void * ctx_ptr) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpusim_entrypoint.cc:146)
libpthread.so.0!start_thread(void * arg) (\build\glibc-B3wQXB\glibc-2.31\nptl\pthread_create.c:477)
libc.so.6!clone() (\build\glibc-B3wQXB\glibc-2.31\sysdeps\unix\sysv\linux\x86_64\clone.S:95)

// 初始化的Seg Fault
libcudart.so!gpgpu_sim::active(gpgpu_sim * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.cc:1158)
<function called from gdb> (Unknown Source:0)
libpthread.so.0!__GI___pthread_mutex_lock(pthread_mutex_t * mutex) (\build\glibc-B3wQXB\glibc-2.31\nptl\pthread_mutex_lock.c:67)
libcudart.so!stream_manager::empty_protected(stream_manager * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\stream_manager.cc:466)
<function called from gdb> (Unknown Source:0)
libcudart.so!gpgpu_t::gpgpu_t(gpgpu_t * const this, const gpgpu_functional_sim_config & config, gpgpu_context * ctx) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\abstract_hardware_model.cc:187)
libcudart.so!gpgpu_sim::gpgpu_sim(gpgpu_sim * const this, const gpgpu_sim_config & config, gpgpu_context * ctx) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.cc:987)
trace_gpgpu_sim::trace_gpgpu_sim(trace_gpgpu_sim * const this, const gpgpu_sim_config & config, gpgpu_context * ctx) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\trace-driven\trace_driven.h:173)
accel_sim_framework::gpgpu_trace_sim_init_perf_model(accel_sim_framework * const this, int argc, const char ** argv, gpgpu_context * m_gpgpu_context, trace_config * m_config) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:231)
accel_sim_framework::accel_sim_framework(accel_sim_framework * const this, int argc, const char ** argv) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:30)
main(int argc, const char ** argv) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\main.cc:29)


libcudart.so!gpgpu_sim::active(gpgpu_sim * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.cc:1158)
<function called from gdb> (Unknown Source:0)
libpthread.so.0!__GI___pthread_mutex_lock(pthread_mutex_t * mutex) (\build\glibc-B3wQXB\glibc-2.31\nptl\pthread_mutex_lock.c:67)
libcudart.so!stream_manager::empty_protected(stream_manager * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\stream_manager.cc:466)
<function called from gdb> (Unknown Source:0)
libcudart.so!gpgpu_sim::gpgpu_sim(gpgpu_sim * const this, const gpgpu_sim_config & config, gpgpu_context * ctx) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.cc:988)
trace_gpgpu_sim::trace_gpgpu_sim(trace_gpgpu_sim * const this, const gpgpu_sim_config & config, gpgpu_context * ctx) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\trace-driven\trace_driven.h:173)
accel_sim_framework::gpgpu_trace_sim_init_perf_model(accel_sim_framework * const this, int argc, const char ** argv, gpgpu_context * m_gpgpu_context, trace_config * m_config) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:231)
accel_sim_framework::accel_sim_framework(accel_sim_framework * const this, int argc, const char ** argv) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:30)
main(int argc, const char ** argv) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\main.cc:29)

给输出stats赋值的链路
libcudart.so!cache_stats::inc_stats(cache_stats * const this, int access_type, int access_outcome, unsigned long long streamID) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:800)
libcudart.so!data_cache::access(data_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:2185)
libcudart.so!l1_cache::access(l1_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:2236)
libcudart.so!ldst_unit::L1_latency_queue_cycle(ldst_unit * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:2425)
libcudart.so!ldst_unit::cycle(ldst_unit * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:3377)
libcudart.so!shader_core_ctx::execute(shader_core_ctx * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:2016)
libcudart.so!shader_core_ctx::cycle(shader_core_ctx * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:4189)
libcudart.so!simt_core_cluster::core_cycle(simt_core_cluster * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:5000)
libcudart.so!gpgpu_sim::cycle(gpgpu_sim * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.cc:2141)
accel_sim_framework::simulate(accel_sim_framework * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:160)
accel_sim_framework::simulation_loop(accel_sim_framework * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:75)
main(int argc, const char ** argv) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\main.cc:30)

对L1D进行初始化配置的链路(包括m_miss_queue_size)
cache_config::init(cache_config * const this, char * config, FuncCache status) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.h:607)
l1d_cache_config::init(l1d_cache_config * const this, char * config, FuncCache status) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.h:975)
shader_core_config::init(shader_core_config * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.h:1582)
gpgpu_sim_config::init(gpgpu_sim_config * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.h:423)
accel_sim_framework::gpgpu_trace_sim_init_perf_model(accel_sim_framework * const this, int argc, const char ** argv, gpgpu_context * m_gpgpu_context, trace_config * m_config) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:226)
accel_sim_framework::accel_sim_framework(accel_sim_framework * const this, int argc, const char ** argv) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:29)
main(int argc, const char ** argv) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\main.cc:29)

/////////////////////////////////////// Refill L2 path ///////////////////////////////////////
libcudart.so!memory_sub_partition::cache_cycle(memory_sub_partition * const this, unsigned int cycle) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\l2cache.cc:497)
libcudart.so!gpgpu_sim::cycle(gpgpu_sim * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.cc:2120)
accel_sim_framework::simulate(accel_sim_framework * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:165)
accel_sim_framework::simulation_loop(accel_sim_framework * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:75)
main(int argc, const char ** argv) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\main.cc:30)

::cache_cycle能够顺利调用的前提是满足(m_L2cache->fill_port_free())
bool baseline_cache::bandwidth_management::fill_port_free() const {
  return (m_fill_port_occupied_cycles == 0);
}
*.config中设定<data_port_width>, 对应源代码中的m_config.m_data_port_width
m_config.get_atom_sz()返回的是完整的cacheline size
void baseline_cache::bandwidth_management::use_fill_port(mem_fetch *mf) {
  // assume filling the entire line with the returned request
  // 计算得到回填完整的一条line所需的cycles
  unsigned fill_cycles = m_config.get_atom_sz() / m_config.m_data_port_width;\
  // 对所需的cycles进行累加，得到最终完成refill所需的总cycles
  m_fill_port_occupied_cycles += fill_cycles;
}

libcudart.so!data_cache::update_m_readable(data_cache * const this, mem_fetch * mf, unsigned int cache_index) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:2244)
libcudart.so!data_cache::wr_hit_wb(data_cache * const this, new_addr_type addr, unsigned int cache_index, mem_fetch * mf, unsigned int time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events, cache_request_status status) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:2273)
libcudart.so!data_cache::process_tag_probe(data_cache * const this, bool wr, cache_request_status probe_status, new_addr_type addr, unsigned int cache_index, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:2927)
libcudart.so!data_cache::access(data_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:2971)
libcudart.so!l1_cache::access(l1_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:3023)
libcudart.so!ldst_unit::L1_latency_queue_cycle(ldst_unit * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:2452)
libcudart.so!ldst_unit::cycle(ldst_unit * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:3396)
libcudart.so!shader_core_ctx::execute(shader_core_ctx * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:2019)
libcudart.so!shader_core_ctx::cycle(shader_core_ctx * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:4200)
libcudart.so!simt_core_cluster::core_cycle(simt_core_cluster * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:5017)
libcudart.so!gpgpu_sim::cycle(gpgpu_sim * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.cc:2150)
accel_sim_framework::simulate(accel_sim_framework * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:165)
accel_sim_framework::simulation_loop(accel_sim_framework * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:75)
main(int argc, const char ** argv) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\main.cc:30)


// Path of replacement
libcudart.so!tag_array::probe(const tag_array * const this, new_addr_type addr, unsigned int & idx, mem_access_sector_mask_t mask, bool is_write, unsigned long long time, bool probe_mode, mem_fetch * mf) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc:503)
if (m_config.m_replacement_policy == LRU) {
  ...
}
libcudart.so!tag_array::probe(const tag_array * const this, new_addr_type addr, unsigned int & idx, mem_fetch * mf, bool is_write, unsigned long long time, bool probe_mode) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc:407)
libcudart.so!data_cache::access(data_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc:3541)
libcudart.so!l2_cache::access(l2_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc:3604)
libcudart.so!memory_sub_partition::cache_cycle(memory_sub_partition * const this, unsigned long long cycle, mem_fetch * mf_monitor) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc:728)
libcudart.so!gpgpu_sim::cycle(gpgpu_sim * const this) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc:2225)
accel_sim_framework::simulate(accel_sim_framework * const this) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/accel-sim.cc:165)
accel_sim_framework::simulation_loop(accel_sim_framework * const this) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/accel-sim.cc:75)
main(int argc, const char ** argv) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/main.cc:30)

// Path of set_recorded_in_mshr
libcudart.so!baseline_cache::send_read_request(baseline_cache * const this, new_addr_type block_addr, unsigned int cache_index, mem_fetch * mf, unsigned long long time, bool & do_miss, bool & wb, evicted_block_info & evicted, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events, bool read_only, bool wa) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc:2695)
// ...
mf->set_recorded_in_mshr();
// ...
libcudart.so!data_cache::rd_miss_base(data_cache * const this, new_addr_type addr, unsigned int cache_index, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events, cache_request_status status) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc:3396)
libcudart.so!data_cache::process_tag_probe(data_cache * const this, bool wr, cache_request_status probe_status, new_addr_type addr, unsigned int cache_index, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc:3509)
libcudart.so!data_cache::access(data_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc:3543)
libcudart.so!l2_cache::access(l2_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc:3604)
libcudart.so!memory_sub_partition::cache_cycle(memory_sub_partition * const this, unsigned long long cycle, mem_fetch * mf_monitor) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc:728)
libcudart.so!gpgpu_sim::cycle(gpgpu_sim * const this) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc:2225)
accel_sim_framework::simulate(accel_sim_framework * const this) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/accel-sim.cc:165)
accel_sim_framework::simulation_loop(accel_sim_framework * const this) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/accel-sim.cc:75)
main(int argc, const char ** argv) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/main.cc:30)

// force_l2_tag_update->fill->allocate
libcudart.so!sector_cache_block::allocate_line(sector_cache_block * const this, new_addr_type tag, new_addr_type block_addr, unsigned int time, mem_access_sector_mask_t sector_mask) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.h:430)
libcudart.so!sector_cache_block::allocate(sector_cache_block * const this, new_addr_type tag, new_addr_type block_addr, unsigned int time, mem_access_sector_mask_t sector_mask) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.h:404)
libcudart.so!tag_array::fill(tag_array * const this, new_addr_type addr, unsigned int time, mem_access_sector_mask_t mask, mem_access_byte_mask_t byte_mask, bool is_write) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc:689)
libcudart.so!baseline_cache::force_tag_access(baseline_cache * const this, new_addr_type addr, unsigned int time, mem_access_sector_mask_t mask) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.h:1784)
libcudart.so!memory_sub_partition::force_l2_tag_update(memory_sub_partition * const this, new_addr_type addr, unsigned int time, mem_access_sector_mask_t mask) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.h:209)
libcudart.so!memory_partition_unit::handle_memcpy_to_gpu(memory_partition_unit * const this, size_t addr, unsigned int global_subpart_id, mem_access_sector_mask_t mask) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc:105)
libcudart.so!gpgpu_sim::perf_memcpy_to_gpu(gpgpu_sim * const this, size_t dst_start_addr, size_t count) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc:2432)
accel_sim_framework::parse_commandlist(accel_sim_framework * const this) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/accel-sim.cc:113)
accel_sim_framework::simulation_loop(accel_sim_framework * const this) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/accel-sim.cc:48)
main(int argc, const char ** argv) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/main.cc:30)

libcudart.so!sector_cache_block::fill(sector_cache_block * const this, unsigned int time, mem_access_sector_mask_t sector_mask, mem_access_byte_mask_t byte_mask) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.h:478)
libcudart.so!tag_array::fill(tag_array * const this, new_addr_type addr, unsigned int time, mem_access_sector_mask_t mask, mem_access_byte_mask_t byte_mask, bool is_write) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc:698)
libcudart.so!baseline_cache::force_tag_access(baseline_cache * const this, new_addr_type addr, unsigned int time, mem_access_sector_mask_t mask) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.h:1784)
libcudart.so!memory_sub_partition::force_l2_tag_update(memory_sub_partition * const this, new_addr_type addr, unsigned int time, mem_access_sector_mask_t mask) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.h:209)
libcudart.so!memory_partition_unit::handle_memcpy_to_gpu(memory_partition_unit * const this, size_t addr, unsigned int global_subpart_id, mem_access_sector_mask_t mask) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc:105)
libcudart.so!gpgpu_sim::perf_memcpy_to_gpu(gpgpu_sim * const this, size_t dst_start_addr, size_t count) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc:2432)
accel_sim_framework::parse_commandlist(accel_sim_framework * const this) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/accel-sim.cc:113)
accel_sim_framework::simulation_loop(accel_sim_framework * const this) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/accel-sim.cc:48)
main(int argc, const char ** argv) (/home/hjs/dev/accel-sim/accel-sim-framework/gpu-simulator/main.cc:30)

调用tag_array::probe的各个位置:
1.
enum cache_request_status status = 
  probe(force_using_lru, "tag_array::fill", addr, idx, mask, is_write, time);

libcudart.so!tag_array::fill(tag_array * const this, new_addr_type addr, unsigned int time, mem_access_sector_mask_t mask, mem_access_byte_mask_t byte_mask, bool is_write) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:784)
libcudart.so!baseline_cache::force_tag_access(baseline_cache * const this, new_addr_type addr, unsigned int time, mem_access_sector_mask_t mask) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.h:1795)
libcudart.so!memory_sub_partition::force_l2_tag_update(memory_sub_partition * const this, new_addr_type addr, unsigned int time, mem_access_sector_mask_t mask) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\l2cache.h:209)
libcudart.so!memory_partition_unit::handle_memcpy_to_gpu(memory_partition_unit * const this, size_t addr, unsigned int global_subpart_id, mem_access_sector_mask_t mask) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\l2cache.cc:105)
libcudart.so!gpgpu_sim::perf_memcpy_to_gpu(gpgpu_sim * const this, size_t dst_start_addr, size_t count) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.cc:2447)
accel_sim_framework::parse_commandlist(accel_sim_framework * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:113)
accel_sim_framework::simulation_loop(accel_sim_framework * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:48)
main(int argc, const char ** argv) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\main.cc:30)

2. 
enum cache_request_status probe_status = m_tag_array->probe(
  force_using_lru, "data_cache::access", block_addr, cache_index, mf, mf->is_write(), time, true);    

libcudart.so!data_cache::access(data_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:3748)
libcudart.so!l1_cache::access(l1_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:3810)
libcudart.so!ldst_unit::L1_latency_queue_cycle(ldst_unit * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:2452)
libcudart.so!ldst_unit::cycle(ldst_unit * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:3402)
libcudart.so!shader_core_ctx::execute(shader_core_ctx * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:2019)
libcudart.so!shader_core_ctx::cycle(shader_core_ctx * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:4206)
libcudart.so!simt_core_cluster::core_cycle(simt_core_cluster * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:5023)
libcudart.so!gpgpu_sim::cycle(gpgpu_sim * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.cc:2266)
accel_sim_framework::simulate(accel_sim_framework * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:165)
accel_sim_framework::simulation_loop(accel_sim_framework * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:75)
main(int argc, const char ** argv) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\main.cc:30)

3.
enum cache_request_status status = 
    probe(force_using_lru, "tag_array::access", addr, idx, mf, mf->is_write(), time);

libcudart.so!tag_array::access(tag_array * const this, new_addr_type addr, unsigned int time, unsigned int & idx, bool & wb, evicted_block_info & evicted, mem_fetch * mf) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:712)
libcudart.so!baseline_cache::send_read_request(baseline_cache * const this, new_addr_type block_addr, unsigned int cache_index, mem_fetch * mf, unsigned long long time, bool & do_miss, bool & wb, evicted_block_info & evicted, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events, bool read_only, bool wa) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:2763)
libcudart.so!data_cache::rd_miss_base(data_cache * const this, new_addr_type addr, unsigned int cache_index, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events, cache_request_status status) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:3601)
libcudart.so!data_cache::process_tag_probe(data_cache * const this, bool wr, cache_request_status probe_status, new_addr_type addr, unsigned int cache_index, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:3715)
libcudart.so!data_cache::access(data_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:3758)
libcudart.so!l1_cache::access(l1_cache * const this, new_addr_type addr, mem_fetch * mf, unsigned long long time, std::__cxx11::list<cache_event, std::allocator<cache_event> > & events) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-cache.cc:3810)
libcudart.so!ldst_unit::L1_latency_queue_cycle(ldst_unit * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:2452)
libcudart.so!ldst_unit::cycle(ldst_unit * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:3402)
libcudart.so!shader_core_ctx::execute(shader_core_ctx * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:2019)
libcudart.so!shader_core_ctx::cycle(shader_core_ctx * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:4206)
libcudart.so!simt_core_cluster::core_cycle(simt_core_cluster * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\shader.cc:5023)
libcudart.so!gpgpu_sim::cycle(gpgpu_sim * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\gpgpu-sim\src\gpgpu-sim\gpu-sim.cc:2266)
accel_sim_framework::simulate(accel_sim_framework * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:165)
accel_sim_framework::simulation_loop(accel_sim_framework * const this) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\accel-sim.cc:75)
main(int argc, const char ** argv) (\home\kuanbba\dev\accel-sim\accel-sim-framework\gpu-simulator\main.cc:30)
