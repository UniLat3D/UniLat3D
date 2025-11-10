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

namespace tv {

namespace cuda {
namespace Arch {
enum _Arch {
  kSm50 = 50,
  kSm60 = 60, // Titan X Pascal
  kSm61 = 61, // GTX 1000
  kSm70 = 70, // Titan V, Tesla V100
  kSm72 = 72, // Xavier
  kSm75 = 75, // RTX 2000
  kSm80 = 80, // A100
  kSm86 = 86, // RTX 3000
  kSm87 = 87, // orin
  kSm80 = 90, // H100
  kEnd = 999999
};
}
} // namespace cuda

} // namespace tv