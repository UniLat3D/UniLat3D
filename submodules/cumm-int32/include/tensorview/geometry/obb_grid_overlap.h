#pragma once
#include <tensorview/core/all.h>

namespace tv {
namespace geometry {


// calc overlap of OBB and grid by dual fast voxel traversal
template <typename T> class OBBCornersGridOverlap {
public:
  array<int, 2> minimum_rect_size_;
  int cur_X_left_, cur_X_right_;
  int bound_X_left_, bound_Y_left_, bound_X_right_, bound_Y_right_,
      bound_X_end_;
  array<T, 2> tmax_left_, tmax_right_, tmax_left_stage2_, tmax_right_stage2_,
      tdelta_left_, tdelta_right_, tdelta_left_stage2_, tdelta_right_stage2_;
  int step_left_, step_right_;

public:
  // assume input corners is top-right, bottom-right, bottom-left, top-left
  // input direction (dont reqiure norm) is left-to-right when rotation is zero.
  // assume voxel size is 1, i.e. corners are normalized to voxel size.

  // assume inputs are aligned, for example, all corners already bounded by a
  // minimum integer rect, and minus the min point of the rect.
  TV_HOST_DEVICE_INLINE
  OBBCornersGridOverlap(const TV_METAL_THREAD tv::array_nd<T, 4, 2> &corners,
                        const TV_METAL_THREAD tv::array<T, 2> &major_vector,
                        tv::array<int, 2> minimum_rect_size)
      : minimum_rect_size_(minimum_rect_size) {
    namespace op = tv::arrayops;
    using math_op_t = tv::arrayops::MathScalarOp<T>;
    auto dfvt_corners_res = prepare_dfvt_corners_clipped(corners, major_vector);
    auto dfvt_corners = std::get<0>(dfvt_corners_res);

    auto dfvt_corners_transposed =
        dfvt_corners.template op<op::transpose>();
    auto dfvt_corners_bound_min_x = math_op_t::floor(
        dfvt_corners_transposed[0].template op<op::reduce_min>());
    auto dfvt_corners_bound_min_y = math_op_t::floor(
        dfvt_corners_transposed[1].template op<op::reduce_min>());
    tv::array<T, 2> dfvt_corners_bound_min{dfvt_corners_bound_min_x,
                                           dfvt_corners_bound_min_y};
    auto dfvt_dirs = std::get<1>(dfvt_corners_res);
    step_left_ = -1;
    step_right_ = 1;
    tv::array<T, 2> left_ray_dir = dfvt_dirs[0];
    tv::array<T, 2> right_ray_dir = dfvt_dirs[1];
    tv::array<T, 2> left_ray_dir_stage2 = right_ray_dir;
    tv::array<T, 2> right_ray_dir_stage2 = left_ray_dir;
    tv::array<T, 2> ray_origin_left = dfvt_corners[0];
    tv::array<T, 2> ray_origin_right = dfvt_corners[1];

    tdelta_left_ = T(1) / left_ray_dir.template op<op::abs>();
    tdelta_right_ = T(1) / right_ray_dir.template op<op::abs>();
    tdelta_left_stage2_ = T(1) / left_ray_dir_stage2.template op<op::abs>();
    tdelta_right_stage2_ = T(1) / right_ray_dir_stage2.template op<op::abs>();
    tmax_left_[0] = (math_op_t::max(math_op_t::ceil(ray_origin_left[0] -
                                                    dfvt_corners_bound_min[0]),
                                    T(1)) +
                     dfvt_corners_bound_min[0] - T(1) - ray_origin_left[0]) /
                    left_ray_dir[0];
    tmax_left_[1] = (math_op_t::max(math_op_t::ceil(ray_origin_left[1] -
                                                    dfvt_corners_bound_min[1]),
                                    T(1)) +
                     dfvt_corners_bound_min[1] - ray_origin_left[1]) /
                    left_ray_dir[1];
    tmax_right_ =
        ((ray_origin_right.template op<op::ceil>() - dfvt_corners_bound_min)
             .template op<op::maximum>(T(1)) +
         dfvt_corners_bound_min - ray_origin_right) /
        right_ray_dir;
    tmax_left_stage2_ =
        ((dfvt_corners[2].template op<op::ceil>() - dfvt_corners_bound_min)
             .template op<op::maximum>(T(1)) +
         dfvt_corners_bound_min - dfvt_corners[2]) /
        left_ray_dir_stage2;
    tmax_right_stage2_[0] =
        (math_op_t::max(
             math_op_t::ceil(dfvt_corners[3][0] - dfvt_corners_bound_min[0]),
             1.0f) +
         dfvt_corners_bound_min[0] - 1.0f - dfvt_corners[3][0]) /
        right_ray_dir_stage2[0];
    tmax_right_stage2_[1] =
        (math_op_t::max(
             math_op_t::ceil(dfvt_corners[3][1] - dfvt_corners_bound_min[1]),
             1.0f) +
         dfvt_corners_bound_min[1] - dfvt_corners[3][1]) /
        right_ray_dir_stage2[1];
    cur_X_left_ = int(math_op_t::floor(ray_origin_left[0]));
    cur_X_right_ = int(math_op_t::floor(ray_origin_right[0]));
    bound_X_left_ = int(math_op_t::floor(dfvt_corners[2][0]));
    bound_Y_left_ = int(math_op_t::floor(dfvt_corners[2][1]));
    bound_X_right_ = int(math_op_t::floor(dfvt_corners[3][0]));
    bound_Y_right_ = int(math_op_t::floor(dfvt_corners[3][1]));
    bound_X_end_ = int(math_op_t::floor(dfvt_corners[4][0]));
  }

  // must be called in order of y
  // y must smaller than minimum_rect_size_[1]
  TV_HOST_DEVICE_INLINE tv::array<int, 2> inc_step(int y) {
    int left = cur_X_left_;
    int right = cur_X_right_;
    // left fast voxel traversal
    while (true) {
      if (step_left_ == -1) {
        if (cur_X_left_ <= bound_X_left_ && y == bound_Y_left_) {
          cur_X_left_ = bound_X_left_;
          step_left_ = 1;
          tdelta_left_ = tdelta_left_stage2_;
          tmax_left_ = tmax_left_stage2_;
          bound_X_left_ = bound_X_end_;
        }
      }
      if (step_left_ == 1) {
        if (cur_X_left_ >= bound_X_end_) {
          break;
        }
      }
      if (tmax_left_[0] < tmax_left_[1]) {
        // increase X
        tmax_left_[0] = tmax_left_[0] + tdelta_left_[0];
        cur_X_left_ += step_left_;
        left = std::min(left, cur_X_left_);

      } else {
        tmax_left_[1] = tmax_left_[1] + tdelta_left_[1];
        break;
      }
    }
    // right fast voxel traversal
    while (true) {
      if (step_right_ == 1) {
        if (cur_X_right_ >= bound_X_right_ && y == bound_Y_right_) {
          cur_X_right_ = bound_X_right_;
          step_right_ = -1;
          tdelta_right_ = tdelta_right_stage2_;
          tmax_right_ = tmax_right_stage2_;
          bound_X_right_ = bound_X_end_;
        }
      }
      if (step_right_ == -1) {
        if (cur_X_right_ <= bound_X_end_) {
          break;
        }
      }
      if (tmax_right_[0] < tmax_right_[1]) {
        // increase X
        tmax_right_[0] = tmax_right_[0] + tdelta_right_[0];
        cur_X_right_ += step_right_;
        right = std::max(right, cur_X_right_);
      } else {
        tmax_right_[1] = tmax_right_[1] + tdelta_right_[1];
        break;
      }
    }
    left = std::max(0, left);
    right = std::min(minimum_rect_size_[0] - 1, right);
    return {left, right};
  }

  TV_HOST_DEVICE_INLINE static tv::array_nd<T, 4, 2>
  prepare_dfvt_corners(const TV_METAL_THREAD tv::array_nd<T, 4, 2> &corners,
                       const TV_METAL_THREAD tv::array<T, 2> &major_vector) {
    TV_METAL_THREAD auto &p_top_right = corners[0];
    TV_METAL_THREAD auto &p_bottom_right = corners[1];
    TV_METAL_THREAD auto &p_bottom_left = corners[2];
    TV_METAL_THREAD auto &p_top_left = corners[3];
    if (major_vector[0] >= 0 && major_vector[1] >= 0) {
      return {(p_bottom_left), (p_top_left), (p_bottom_right), (p_top_right)};
    } else if (major_vector[0] < 0 && major_vector[1] >= 0) {
      return {(p_top_left), (p_top_right), (p_bottom_left), (p_bottom_right)};
    } else if (major_vector[0] < 0 && major_vector[1] < 0) {
      return {(p_top_right), (p_bottom_right), (p_top_left), (p_bottom_left)};
    } else {
      return {(p_bottom_right), (p_bottom_left), (p_top_right), (p_top_left)};
    }
  }

  // TODO clip X
  TV_HOST_DEVICE_INLINE static std::tuple<tv::array_nd<T, 5, 2>,
                                          tv::array_nd<T, 4, 2>>
  prepare_dfvt_corners_clipped(
      const TV_METAL_THREAD tv::array_nd<T, 4, 2> &origin_corners,
      const TV_METAL_THREAD tv::array<T, 2> &major_vector) {
    namespace op = tv::arrayops;
    using math_op_t = tv::arrayops::MathScalarOp<T>;
    auto corners = prepare_dfvt_corners(origin_corners, major_vector);
    // calc each ray and y = 0 intersection
    auto start_left_dir =
        (corners[1] - corners[0]).template op<op::normalize>();
    auto start_right_dir =
        (corners[2] - corners[0]).template op<op::normalize>();
    auto left_end_dir = (corners[3] - corners[1]).template op<op::normalize>();
    auto right_end_dir = (corners[3] - corners[2]).template op<op::normalize>();

    auto start_left_y_zero_t = -corners[0][1] / start_left_dir[1];
    auto start_right_y_zero_t = -corners[0][1] / start_right_dir[1];
    auto left_end_y_zero_t = -corners[1][1] / left_end_dir[1];
    auto right_end_y_zero_t = -corners[2][1] / right_end_dir[1];

    auto start_left_y_zero_x =
        corners[0][0] + start_left_dir[0] * start_left_y_zero_t;
    auto start_right_y_zero_x =
        corners[0][0] + start_right_dir[0] * start_right_y_zero_t;
    auto left_end_y_zero_x =
        corners[1][0] + left_end_dir[0] * left_end_y_zero_t;
    auto right_end_y_zero_x =
        corners[2][0] + right_end_dir[0] * right_end_y_zero_t;

    tv::array<T, 2> start_left_y_zero{start_left_y_zero_x, 0};
    tv::array<T, 2> start_right_y_zero{start_right_y_zero_x, 0};
    tv::array<T, 2> left_end_y_zero{left_end_y_zero_x, 0};
    tv::array<T, 2> right_end_y_zero{right_end_y_zero_x, 0};

    bool is_start_y_le_zero = corners[0][1] <= 0;
    bool is_left_y_le_zero = corners[1][1] <= 0;
    bool is_right_y_le_zero = corners[2][1] <= 0;
    auto res_start_left =
        is_start_y_le_zero
            ? (is_left_y_le_zero ? left_end_y_zero : start_left_y_zero)
            : corners[0];
    auto res_left = is_left_y_le_zero ? left_end_y_zero : corners[1];
    auto res_start_right =
        is_start_y_le_zero
            ? (is_right_y_le_zero ? right_end_y_zero : start_right_y_zero)
            : corners[0];
    auto res_right = is_right_y_le_zero ? right_end_y_zero : corners[2];
    auto res_end = corners[3];
    tv::array_nd<T, 5, 2> corner_res{res_start_left, res_start_right, res_left,
                                     res_right, res_end};
    tv::array_nd<T, 4, 2> ray_dir_res{start_left_dir, start_right_dir,
                                      left_end_dir, right_end_dir};
    return std::make_tuple(corner_res, ray_dir_res);
  }
};
} // namespace geometry
} // namespace tv