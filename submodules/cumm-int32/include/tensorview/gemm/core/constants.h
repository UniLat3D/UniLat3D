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

namespace gemm {

enum class ConvOpType { kForward = 0, kBackwardInput = 1, kBackwardWeight = 2 };

enum class ConvMode { kConvolution = 0, kCrossCorrelation = 1 };
enum class ConvIterAlgo { kAnalytic = 0, kOptimized = 1 };

enum class ConvLayoutType {
  kChannelFirst = 0,
  kChannelLast = 1,
  kSpatialFirst = 2

};

enum class ShuffleStrideType { kNoShuffle = 0, kShuffleAC = 1, kShuffleAB = 2 };

enum class SparseConvAlgo {
  kNative = 0,
  kMaskImplicitGemm = 1,
  kMaskSplitImplicitGemm = 2
};

enum class Activation {
  // we only support three activations here.
  kNone = 0,
  kReLU = 1,
  kSigmoid = 2,
  kLeakyReLU = 3,
  // kELU = 5,
  // kSeLU = 6,
  // kSoftsign = 7,
  // kSoftplus = 8,
  // kClip = 9,
  // kHardSigmoid = 10,
  // kScaledTanh = 11,
  // kThresholdedReLU = 12

};

} // namespace gemm
} // namespace tv