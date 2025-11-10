// Copyright 2024 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "array.h"
#include "const_ops.h"
#include "defs.h"
#include "mp_helper.h"
#ifndef TV_METAL_RTC
#include "printf2.h"
#endif
#ifndef TV_PARALLEL_RTC
#include "cc17.h"
#endif
#ifndef TV_METAL_RTC
#include "const_string.h"
#endif
