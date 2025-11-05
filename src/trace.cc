// Copyright (c) 2009-2013, Tor M. Aamodt, Timothy Rogers,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
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

#include "trace.h"
#include "string.h"
#include <stdio.h>
#include <ctype.h>

namespace Trace {

#define TS_TUP_BEGIN(X) const char* trace_streams_str[] = {
#define TS_TUP(X) #X
#define TS_TUP_END(X) \
  }                   \
  ;
#include "trace_streams.tup"
#undef TS_TUP_BEGIN
#undef TS_TUP
#undef TS_TUP_END

bool enabled = false;
int sampling_core = 0;
int sampling_memory_partition = -1;
bool trace_streams_enabled[NUM_TRACE_STREAMS] = {false};
const char* config_str;
FILE* out = stdout;
char* output_filename = NULL;
unsigned long long max_lines = 0ULL;
unsigned long long lines_emitted = 0ULL;
unsigned long long stop_cycle = 0ULL;

bool allow_emit(unsigned long long cycle) {
  if (!enabled) return false;
  if (stop_cycle > 0ULL && cycle > stop_cycle) return false;
  if (max_lines > 0ULL && lines_emitted >= max_lines) return false;
  return true;
}

static inline void trim(char* s) {
  if (!s) return;
  // left trim
  char* p = s;
  while (*p && isspace((unsigned char)*p)) ++p;
  if (p != s) memmove(s, p, strlen(p) + 1);
  // right trim
  size_t len = strlen(s);
  while (len > 0 && isspace((unsigned char)s[len - 1])) s[--len] = '\0';
}

void init() {
  // Reset all streams to disabled by default
  for (unsigned i = 0; i < NUM_TRACE_STREAMS; ++i) trace_streams_enabled[i] = false;

  if (!config_str || config_str[0] == '\0' || strcmp(config_str, "none") == 0) {
    // Nothing to enable
  } else {
    // Tokenize by comma, do exact token matches (avoid substring false-positives like the ALL in STALL)
    char buf[1024];
    buf[0] = '\0';
    // Guard buffer size
    strncpy(buf, config_str, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    bool enable_all = false;
    // First pass: check for any ALL-like tokens
    {
      char tmp[1024];
      strncpy(tmp, buf, sizeof(tmp) - 1);
      tmp[sizeof(tmp) - 1] = '\0';
      char* saveptr = NULL;
      for (char* tok = strtok_r(tmp, ",", &saveptr); tok; tok = strtok_r(NULL, ",", &saveptr)) {
        trim(tok);
        if (tok[0] == '\0') continue;
        if (strcmp(tok, "*") == 0 || strcasecmp(tok, "ALL") == 0 || strcmp(tok, "ExecAll") == 0) {
          enable_all = true;
          break;
        }
      }
    }

    if (enable_all) {
      for (unsigned i = 0; i < NUM_TRACE_STREAMS; ++i) trace_streams_enabled[i] = true;
    } else {
      // Second pass: enable exact-matched streams
      char* saveptr = NULL;
      for (char* tok = strtok_r(buf, ",", &saveptr); tok; tok = strtok_r(NULL, ",", &saveptr)) {
        trim(tok);
        if (tok[0] == '\0') continue;
        for (unsigned i = 0; i < NUM_TRACE_STREAMS; ++i) {
          if (strcmp(tok, trace_streams_str[i]) == 0) {
            trace_streams_enabled[i] = true;
            break;
          }
        }
      }
    }
  }
  // If an explicit output filename is provided, redirect trace prints there.
  if (output_filename && output_filename[0] != '\0') {
    FILE* f = fopen(output_filename, "w");
    if (f) {
      out = f;
      // Line-buffer the file to keep interleaved messages readable.
      setvbuf(out, NULL, _IOLBF, 0);
    }
  }
}
}  // namespace Trace
