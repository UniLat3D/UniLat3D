# Copyright 2024 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cumm.nvrtc import NVRTCModuleParams, CummNVRTCModuleBase

class NVRTCCompiler:
    def compile_nvrtc(self, params: NVRTCModuleParams):
        return CummNVRTCModuleBase(params.code,
                         params.headers,
                         params.opts,
                         name_exprs=params.name_exprs,
                         name_to_meta=params.name_to_meta,
                         cudadevrt_path=params.cudadevrt_path)