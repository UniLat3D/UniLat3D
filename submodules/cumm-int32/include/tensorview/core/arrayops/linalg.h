#pragma once

#include "mathbase.h"
#include "simple.h"
#include <tensorview/core/array.h>
// from https://github.com/sgorsten/linalg/blob/main/linalg.h

namespace tv {
namespace arrayops {

template <typename T, size_t N, size_t Align> struct diagonal;

template <typename T> struct diagonal<array<T, 1, 0>, 1, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 1>
  operator()(const TV_METAL_THREAD array<array<T, 1>, 1> &self) {
    return {self[0][0]};
  }
};
template <typename T> struct diagonal<array<T, 2, 0>, 2, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 2>
  operator()(const TV_METAL_THREAD array<array<T, 2>, 2> &self) {
    return {self[0][0], self[1][1]};
  }
};

template <typename T> struct diagonal<array<T, 3, 0>, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const TV_METAL_THREAD array<array<T, 3>, 3> &self) {
    return {self[0][0], self[1][1], self[2][2]};
  }
};

template <typename T> struct diagonal<array<T, 4, 0>, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 4>
  operator()(const TV_METAL_THREAD array<array<T, 4>, 4> &self) {
    return {self[0][0], self[1][1], self[2][2], self[3][3]};
  }
};

template <typename T, size_t N, size_t Align> struct from_diagonal;

template <typename T> struct from_diagonal<T, 1, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 1>, 1>
  operator()(const TV_METAL_THREAD array<T, 1> &self) {
    return {{self[0]}};
  }
};

template <typename T> struct from_diagonal<T, 2, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 2>, 2>
  operator()(const TV_METAL_THREAD array<T, 2> &self) {
    return {array<T, 2>{self[0], T(0)}, array<T, 2>{T(0), self[1]}};
  }
};

template <typename T> struct from_diagonal<T, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 3>
  operator()(const TV_METAL_THREAD array<T, 3> &self) {
    return {array<T, 3>{self[0], T(0), T(0)}, array<T, 3>{T(0), self[1], T(0)},
            array<T, 3>{T(0), T(0), self[2]}};
  }
};

template <typename T> struct from_diagonal<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 4>, 4>
  operator()(const TV_METAL_THREAD array<T, 4> &self) {
    return {array<T, 4>{self[0], T(0), T(0), T(0)},
            array<T, 4>{T(0), self[1], T(0), T(0)},
            array<T, 4>{T(0), T(0), self[2], T(0)},
            array<T, 4>{T(0), T(0), T(0), self[3]}};
  }
};

template <typename T, size_t N, size_t Align> struct determinant;

template <typename T> struct determinant<array<T, 1, 0>, 1, 0> {
  TV_HOST_DEVICE_INLINE constexpr T
  operator()(const TV_METAL_THREAD array<array<T, 1>, 1> &self) {
    return self[0][0];
  }
};

template <typename T> struct determinant<array<T, 2, 0>, 2, 0> {
  TV_HOST_DEVICE_INLINE constexpr T
  operator()(const TV_METAL_THREAD array<array<T, 2>, 2> &self) {
    return self[0][0] * self[1][1] - self[0][1] * self[1][0];
  }
};

template <typename T> struct determinant<array<T, 3, 0>, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr T
  operator()(const TV_METAL_THREAD array<array<T, 3>, 3> &a) {
    return (a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2]) +
            a[0][1] * (a[1][2] * a[2][0] - a[2][2] * a[1][0]) +
            a[0][2] * (a[1][0] * a[2][1] - a[2][0] * a[1][1]));
  }
};

template <typename T> struct determinant<array<T, 4, 0>, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr T
  operator()(const TV_METAL_THREAD array<array<T, 4>, 4> &a) {
    return a[0][0] *
               (a[1][1] * a[2][2] * a[3][3] + a[3][1] * a[1][2] * a[2][3] +
                a[2][1] * a[3][2] * a[1][3] - a[1][1] * a[3][2] * a[2][3] -
                a[2][1] * a[1][2] * a[3][3] - a[3][1] * a[2][2] * a[1][3]) +
           a[0][1] *
               (a[1][2] * a[3][3] * a[2][0] + a[2][2] * a[1][3] * a[3][0] +
                a[3][2] * a[2][3] * a[1][0] - a[1][2] * a[2][3] * a[3][0] -
                a[3][2] * a[1][3] * a[2][0] - a[2][2] * a[3][3] * a[1][0]) +
           a[0][2] *
               (a[1][3] * a[2][0] * a[3][1] + a[3][3] * a[1][0] * a[2][1] +
                a[2][3] * a[3][0] * a[1][1] - a[1][3] * a[3][0] * a[2][1] -
                a[2][3] * a[1][0] * a[3][1] - a[3][3] * a[2][0] * a[1][1]) +
           a[0][3] *
               (a[1][0] * a[3][1] * a[2][2] + a[2][0] * a[1][1] * a[3][2] +
                a[3][0] * a[2][1] * a[1][2] - a[1][0] * a[2][1] * a[3][2] -
                a[3][0] * a[1][1] * a[2][2] - a[2][0] * a[3][1] * a[1][2]);
  }
};

template <typename T, size_t N, size_t Align> struct adjugate;
template <typename T> struct adjugate<array<T, 1, 0>, 1, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 1>, 1>
  operator()(const TV_METAL_THREAD array<array<T, 1>, 1> &self) {
    return {{T(1)}};
  }
};

template <typename T> struct adjugate<array<T, 2, 0>, 2, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 2>, 2>
  operator()(const TV_METAL_THREAD array<array<T, 2>, 2> &a) {
    return {array<T, 2>{a[1][1], -a[0][1]}, array<T, 2>{-a[1][0], a[0][0]}};
  }
};

template <typename T> struct adjugate<array<T, 3, 0>, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 3>
  operator()(const TV_METAL_THREAD array<array<T, 3>, 3> &a) {
    return {array<T, 3>{a[1][1] * a[2][2] - a[2][1] * a[1][2],
                        a[2][1] * a[0][2] - a[0][1] * a[2][2],
                        a[0][1] * a[1][2] - a[1][1] * a[0][2]},
            array<T, 3>{a[1][2] * a[2][0] - a[2][2] * a[1][0],
                        a[2][2] * a[0][0] - a[0][2] * a[2][0],
                        a[0][2] * a[1][0] - a[1][2] * a[0][0]},
            array<T, 3>{a[1][0] * a[2][1] - a[2][0] * a[1][1],
                        a[2][0] * a[0][1] - a[0][0] * a[2][1],
                        a[0][0] * a[1][1] - a[1][0] * a[0][1]}};
  }
};

template <typename T> struct adjugate<array<T, 4, 0>, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 4>, 4>
  operator()(const TV_METAL_THREAD array<array<T, 4>, 4> &a) {
    return {array<T, 4>{
                a[1][1] * a[2][2] * a[3][3] + a[3][1] * a[1][2] * a[2][3] +
                    a[2][1] * a[3][2] * a[1][3] - a[1][1] * a[3][2] * a[2][3] -
                    a[2][1] * a[1][2] * a[3][3] - a[3][1] * a[2][2] * a[1][3],
                a[0][1] * a[3][2] * a[2][3] + a[2][1] * a[0][2] * a[3][3] +
                    a[3][1] * a[2][2] * a[0][3] - a[3][1] * a[0][2] * a[2][3] -
                    a[2][1] * a[3][2] * a[0][3] - a[0][1] * a[2][2] * a[3][3],
                a[0][1] * a[1][2] * a[3][3] + a[3][1] * a[0][2] * a[1][3] +
                    a[1][1] * a[3][2] * a[0][3] - a[0][1] * a[3][2] * a[1][3] -
                    a[1][1] * a[0][2] * a[3][3] - a[3][1] * a[1][2] * a[0][3],
                a[0][1] * a[2][2] * a[1][3] + a[1][1] * a[0][2] * a[2][3] +
                    a[2][1] * a[1][2] * a[0][3] - a[0][1] * a[1][2] * a[2][3] -
                    a[2][1] * a[0][2] * a[1][3] - a[1][1] * a[2][2] * a[0][3]},
            array<T, 4>{
                a[1][2] * a[3][3] * a[2][0] + a[2][2] * a[1][3] * a[3][0] +
                    a[3][2] * a[2][3] * a[1][0] - a[1][2] * a[2][3] * a[3][0] -
                    a[3][2] * a[1][3] * a[2][0] - a[2][2] * a[3][3] * a[1][0],
                a[0][2] * a[2][3] * a[3][0] + a[3][2] * a[0][3] * a[2][0] +
                    a[2][2] * a[3][3] * a[0][0] - a[0][2] * a[3][3] * a[2][0] -
                    a[2][2] * a[0][3] * a[3][0] - a[3][2] * a[2][3] * a[0][0],
                a[0][2] * a[3][3] * a[1][0] + a[1][2] * a[0][3] * a[3][0] +
                    a[3][2] * a[1][3] * a[0][0] - a[0][2] * a[1][3] * a[3][0] -
                    a[3][2] * a[0][3] * a[1][0] - a[1][2] * a[3][3] * a[0][0],
                a[0][2] * a[1][3] * a[2][0] + a[2][2] * a[0][3] * a[1][0] +
                    a[1][2] * a[2][3] * a[0][0] - a[0][2] * a[2][3] * a[1][0] -
                    a[1][2] * a[0][3] * a[2][0] - a[2][2] * a[1][3] * a[0][0]},
            array<T, 4>{
                a[1][3] * a[2][0] * a[3][1] + a[3][3] * a[1][0] * a[2][1] +
                    a[2][3] * a[3][0] * a[1][1] - a[1][3] * a[3][0] * a[2][1] -
                    a[2][3] * a[1][0] * a[3][1] - a[3][3] * a[2][0] * a[1][1],
                a[0][3] * a[3][0] * a[2][1] + a[2][3] * a[0][0] * a[3][1] +
                    a[3][3] * a[2][0] * a[0][1] - a[0][3] * a[2][0] * a[3][1] -
                    a[3][3] * a[0][0] * a[2][1] - a[2][3] * a[3][0] * a[0][1],
                a[0][3] * a[1][0] * a[3][1] + a[3][3] * a[0][0] * a[1][1] +
                    a[1][3] * a[3][0] * a[0][1] - a[0][3] * a[3][0] * a[1][1] -
                    a[1][3] * a[0][0] * a[3][1] - a[3][3] * a[1][0] * a[0][1],
                a[0][3] * a[2][0] * a[1][1] + a[1][3] * a[0][0] * a[2][1] +
                    a[2][3] * a[1][0] * a[0][1] - a[0][3] * a[1][0] * a[2][1] -
                    a[2][3] * a[0][0] * a[1][1] - a[1][3] * a[2][0] * a[0][1]},
            array<T, 4>{
                a[1][0] * a[3][1] * a[2][2] + a[2][0] * a[1][1] * a[3][2] +
                    a[3][0] * a[2][1] * a[1][2] - a[1][0] * a[2][1] * a[3][2] -
                    a[3][0] * a[1][1] * a[2][2] - a[2][0] * a[3][1] * a[1][2],
                a[0][0] * a[2][1] * a[3][2] + a[3][0] * a[0][1] * a[2][2] +
                    a[2][0] * a[3][1] * a[0][2] - a[0][0] * a[3][1] * a[2][2] -
                    a[2][0] * a[0][1] * a[3][2] - a[3][0] * a[2][1] * a[0][2],
                a[0][0] * a[3][1] * a[1][2] + a[1][0] * a[0][1] * a[3][2] +
                    a[3][0] * a[1][1] * a[0][2] - a[0][0] * a[1][1] * a[3][2] -
                    a[3][0] * a[0][1] * a[1][2] - a[1][0] * a[3][1] * a[0][2],
                a[0][0] * a[1][1] * a[2][2] + a[2][0] * a[0][1] * a[1][2] +
                    a[1][0] * a[2][1] * a[0][2] - a[0][0] * a[2][1] * a[1][2] -
                    a[1][0] * a[0][1] * a[2][2] - a[2][0] * a[1][1] * a[0][2]}};
  }
};
template <typename T, size_t N, size_t Align> struct row;
template <typename T, size_t N, size_t Align> struct col;

template <typename T, size_t Row, size_t Col>
struct row<array<T, Col>, Row, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const TV_METAL_THREAD array<array<T, Col>, Row> &self,
             int idx) const {
    return self[idx];
  }
};

template <typename T, size_t Row, size_t Col>
struct col<array<T, Col>, Row, 0> {
private:
  template <int... Inds>
  TV_HOST_DEVICE_INLINE static constexpr array<T, Row>
  col_impl(const TV_METAL_THREAD array<array<T, Col>, Row> &a, int idx,
           mp_list_int<Inds...>) noexcept {
    return array<T, Row>{a[Inds][idx]...};
  }

public:
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const TV_METAL_THREAD array<array<T, Col>, Row> &self, int idx) {
    return col_impl(self, idx, mp_make_list_c_sequence<int, Row>{});
  }
};

template <typename T, size_t N, size_t Align> struct transpose;

template <typename T, size_t Row, size_t Col>
struct transpose<array<T, Col>, Row, 0> {
private:
  template <int... Inds>
  TV_HOST_DEVICE_INLINE static constexpr array<array<T, Row>, Col>
  transpose_impl(const TV_METAL_THREAD array<array<T, Col>, Row> &a,
                 mp_list_int<Inds...>) noexcept {
    return array<array<T, Row>, Col>{a.template op<col>(Inds)...};
  }

public:
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Row>, Col>
  operator()(const TV_METAL_THREAD array<array<T, Col>, Row> &self) {
    return transpose_impl(self, mp_make_list_c_sequence<int, Col>{});
  }
};

template <typename T, size_t N, size_t Align> struct inverse;

template <typename T, size_t N> struct inverse<array<T, N, 0>, N, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, N>, N>
  operator()(const TV_METAL_THREAD array<array<T, N>, N> &self) {
    return self.template op<adjugate>() / self.template op<determinant>();
  }
};

template <typename T, size_t N, size_t Align> struct mv_rowmajor;
template <typename T, size_t N, size_t Align> struct mv_colmajor;

#if __cplusplus >= 201703L
template <typename T, size_t Row, size_t Col>
struct mv_colmajor<array<T, Col, 0>, Row, 0> {
  template <size_t... Inds>
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  impl(const TV_METAL_THREAD array<array<T, Col>, Row> &self,
       const TV_METAL_THREAD array<T, Row> &vec, mp_list_int<Inds...>) {
    return ((self.template op<row>(Inds) * vec[Inds]) + ...);
  }

  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const TV_METAL_THREAD array<array<T, Col>, Row> &self,
             const TV_METAL_THREAD array<T, Row> &vec) {
    return impl(self, vec, mp_make_list_c_sequence<size_t, Row>{});
  }
};
#endif

template <typename T, size_t Col> struct mv_colmajor<array<T, Col, 0>, 1, 0> {

  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const TV_METAL_THREAD array<array<T, Col>, 1> &self,
             const TV_METAL_THREAD array<T, 1> &vec) {
    return self.template op<row>(0) * vec[0];
  }
};

template <typename T, size_t Col> struct mv_colmajor<array<T, Col, 0>, 2, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const TV_METAL_THREAD array<array<T, Col>, 2> &self,
             const TV_METAL_THREAD array<T, 2> &vec) {
    return self.template op<row>(0) * vec[0] +
           self.template op<row>(1) * vec[1];
  }
};

template <typename T, size_t Col> struct mv_colmajor<array<T, Col, 0>, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const TV_METAL_THREAD array<array<T, Col>, 3> &self,
             const TV_METAL_THREAD array<T, 3> &vec) {
    return self.template op<row>(0) * vec[0] +
           self.template op<row>(1) * vec[1] +
           self.template op<row>(2) * vec[2];
  }
};

template <typename T, size_t Col> struct mv_colmajor<array<T, Col, 0>, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const TV_METAL_THREAD array<array<T, Col>, 4> &self,
             const TV_METAL_THREAD array<T, 4> &vec) {
    return self.template op<row>(0) * vec[0] +
           self.template op<row>(1) * vec[1] +
           self.template op<row>(2) * vec[2] +
           self.template op<row>(3) * vec[3];
  }
};

template <typename T, size_t Col, size_t Align> struct mv_colmajor_grad_lfs {
  template <size_t Row>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Col>, Row>
  operator()(const TV_METAL_THREAD array<T, Col> &grad,
             const TV_METAL_THREAD array<array<T, Col>, Row> &inp_lfs,
             const TV_METAL_THREAD array<T, Row> &inp_rfs) {
    return reshape<Row, 1>(inp_rfs) * reshape<1, Col>(grad);
  }
  template <size_t Row>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Col>, Row>
  operator()(const TV_METAL_THREAD array<T, Col> &grad,
             const TV_METAL_THREAD array<T, Row> &inp_rfs) {
    return reshape<Row, 1>(inp_rfs) * reshape<1, Col>(grad);
  }
};

template <typename T, size_t Col, size_t Align> struct mv_colmajor_grad_rfs {
  template <size_t Row>
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const TV_METAL_THREAD array<T, Col> &grad,
             const TV_METAL_THREAD array<array<T, Col>, Row> &inp_lfs,
             const TV_METAL_THREAD array<T, Row> &inp_rfs) {
    return inp_lfs.template op<mv_rowmajor>(grad);
  }
  template <size_t Row>
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const TV_METAL_THREAD array<T, Col> &grad,
             const TV_METAL_THREAD array<array<T, Col>, Row> &inp_lfs) {
    return inp_lfs.template op<mv_rowmajor>(grad);
  }
};

#if __cplusplus >= 201703L
template <typename T, size_t Row, size_t Col>
struct mv_rowmajor<array<T, Col, 0>, Row, 0> {
  template <size_t... Inds>
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  impl(const TV_METAL_THREAD array<array<T, Col>, Row> &self,
       const TV_METAL_THREAD array<T, Col> &vec, mp_list_int<Inds...>) {
    return ((self.template op<col>(Inds) * vec[Inds]) + ...);
  }

  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const TV_METAL_THREAD array<array<T, Col>, Row> &self,
             const TV_METAL_THREAD array<T, Col> &vec) {
    return impl(self, vec, mp_make_list_c_sequence<size_t, Col>{});
  }
};
#endif

template <typename T, size_t Row> struct mv_rowmajor<array<T, 1, 0>, Row, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const TV_METAL_THREAD array<array<T, 1>, Row> &self,
             const TV_METAL_THREAD array<T, 1> &vec) {
    return self.template op<col>(0) * vec[0];
  }
};

template <typename T, size_t Row> struct mv_rowmajor<array<T, 2, 0>, Row, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const TV_METAL_THREAD array<array<T, 2>, Row> &self,
             const TV_METAL_THREAD array<T, 2> &vec) {
    return self.template op<col>(0) * vec[0] +
           self.template op<col>(1) * vec[1];
  }
};

template <typename T, size_t Row> struct mv_rowmajor<array<T, 3, 0>, Row, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const TV_METAL_THREAD array<array<T, 3>, Row> &self,
             const TV_METAL_THREAD array<T, 3> &vec) {
    return self.template op<col>(0) * vec[0] +
           self.template op<col>(1) * vec[1] +
           self.template op<col>(2) * vec[2];
  }
};

template <typename T, size_t Row> struct mv_rowmajor<array<T, 4, 0>, Row, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Row>
  operator()(const TV_METAL_THREAD array<array<T, 4>, Row> &self,
             const TV_METAL_THREAD array<T, 4> &vec) {
    return self.template op<col>(0) * vec[0] +
           self.template op<col>(1) * vec[1] +
           self.template op<col>(2) * vec[2] +
           self.template op<col>(3) * vec[3];
  }
};

template <typename T, size_t Row, size_t Align> struct mv_rowmajor_grad_lfs {
  template <size_t Col>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Col>, Row>
  operator()(const TV_METAL_THREAD array<T, Row> &grad,
            const TV_METAL_THREAD array<array<T, Col>, Row> &inp_lfs,
             const TV_METAL_THREAD array<T, Col> &inp_rfs) {
    return (reshape<Row, 1>(grad) * reshape<1, Col>(inp_rfs));
  }
  template <size_t Col>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Col>, Row>
  operator()(const TV_METAL_THREAD array<T, Row> &grad,
             const TV_METAL_THREAD array<T, Col> &inp_rfs) {
    return (reshape<Row, 1>(grad) * reshape<1, Col>(inp_rfs));
  }
};

template <typename T, size_t Row, size_t Align> struct mv_rowmajor_grad_rfs {
  template <size_t Col>
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const TV_METAL_THREAD array<T, Row> &grad,
             const TV_METAL_THREAD array<array<T, Col>, Row> &inp_lfs,
             const TV_METAL_THREAD array<T, Col> &inp_rfs) {
    return inp_lfs.template op<mv_colmajor>(grad);
  }
  template <size_t Col>
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const TV_METAL_THREAD array<T, Row> &grad,
             const TV_METAL_THREAD array<array<T, Col>, Row> &inp_lfs) {
    return inp_lfs.template op<mv_colmajor>(grad);
  }
};

template <typename T, size_t N, size_t Align> struct mm_nnn;
template <typename T, size_t N, size_t Align> struct mm_ttt;
template <typename T, size_t N, size_t Align> struct mm_ntn;
template <typename T, size_t N, size_t Align> struct mm_tnn;

template <typename T, size_t N, size_t Align> struct mm_nnn_grad_lfs;
template <typename T, size_t N, size_t Align> struct mm_nnn_grad_rfs;
template <typename T, size_t N, size_t Align> struct mm_ttt_grad_lfs;
template <typename T, size_t N, size_t Align> struct mm_ttt_grad_rfs;

template <typename T, size_t M, size_t K> struct mm_nnn<array<T, M>, K, 0> {
  // MK.T @ KN.T = MN.T
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, 1>
  operator()(const TV_METAL_THREAD array<array<T, M>, K> &self,
             const TV_METAL_THREAD array<array<T, K>, 1> &other) {
    return {self.template op<mv_colmajor>(other.template op<row>(0))};
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, 2>
  operator()(const TV_METAL_THREAD array<array<T, M>, K> &self,
             const TV_METAL_THREAD array<array<T, K>, 2> &other) {
    return {self.template op<mv_colmajor>(other.template op<row>(0)),
            self.template op<mv_colmajor>(other.template op<row>(1))};
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, 3>
  operator()(const TV_METAL_THREAD array<array<T, M>, K> &self,
             const TV_METAL_THREAD array<array<T, K>, 3> &other) {
    return {self.template op<mv_colmajor>(other.template op<row>(0)),
            self.template op<mv_colmajor>(other.template op<row>(1)),
            self.template op<mv_colmajor>(other.template op<row>(2))};
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, 4>
  operator()(const TV_METAL_THREAD array<array<T, M>, K> &self,
             const TV_METAL_THREAD array<array<T, K>, 4> &other) {
    return {self.template op<mv_colmajor>(other.template op<row>(0)),
            self.template op<mv_colmajor>(other.template op<row>(1)),
            self.template op<mv_colmajor>(other.template op<row>(2)),
            self.template op<mv_colmajor>(other.template op<row>(3))};
  }
};

template <typename T, size_t M, size_t N>
struct mm_nnn_grad_lfs<array<T, M>, N, 0> {
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, K>
  operator()(const TV_METAL_THREAD array<array<T, M>, N> &grad,
             const TV_METAL_THREAD array<array<T, M>, K> &self,
             const TV_METAL_THREAD array<array<T, K>, N> &other) {
    return grad.template op<mm_nnn>(other.template op<transpose>());
  }
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, K>
  operator()(const TV_METAL_THREAD array<array<T, M>, N> &grad,
             const TV_METAL_THREAD array<array<T, K>, N> &other) {
    return grad.template op<mm_nnn>(other.template op<transpose>());
  }

};

template <typename T, size_t M, size_t N>
struct mm_nnn_grad_rfs<array<T, M>, N, 0> {
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, K>, N>
  operator()(const TV_METAL_THREAD array<array<T, M>, N> &grad,
             const TV_METAL_THREAD array<array<T, M>, K> &self,
             const TV_METAL_THREAD array<array<T, K>, N> &other) {
    return self.template op<transpose>().template op<mm_nnn>(grad);
  }
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, K>, N>
  operator()(const TV_METAL_THREAD array<array<T, M>, N> &grad,
             const TV_METAL_THREAD array<array<T, M>, K> &self) {
    return self.template op<transpose>().template op<mm_nnn>(grad);
  }
};

template <typename T, size_t M, size_t N>
struct mm_ttt_grad_lfs<array<T, N>, M, 0> {
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, K>, M>
  operator()(const TV_METAL_THREAD array<array<T, N>, M> &grad,
             const TV_METAL_THREAD array<array<T, K>, M> &self,
             const TV_METAL_THREAD array<array<T, N>, K> &other) {
    return grad.template op<mm_ttt>(other.template op<transpose>());
  }
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, K>, M>
  operator()(const TV_METAL_THREAD array<array<T, N>, M> &grad,
             const TV_METAL_THREAD array<array<T, N>, K> &other) {
    return grad.template op<mm_ttt>(other.template op<transpose>());
  }
};

template <typename T, size_t M, size_t N>
struct mm_ttt_grad_rfs<array<T, N>, M, 0> {
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, N>, K>
  operator()(const TV_METAL_THREAD array<array<T, N>, M> &grad,
             const TV_METAL_THREAD array<array<T, K>, M> &self,
             const TV_METAL_THREAD array<array<T, N>, K> &other) {
    return self.template op<transpose>().template op<mm_ttt>(grad);
  }
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, N>, K>
  operator()(const TV_METAL_THREAD array<array<T, N>, M> &grad,
             const TV_METAL_THREAD array<array<T, K>, M> &self) {
    return self.template op<transpose>().template op<mm_ttt>(grad);
  }
};

template <typename T, size_t K, size_t M> struct mm_ttt<array<T, K>, M, 0> {
  // XC @ CR = XR
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 1>, M>
  operator()(const TV_METAL_THREAD array<array<T, K>, M> &self,
             const TV_METAL_THREAD array<array<T, 1>, K> &other) {
    return array<array<T, M>, 1>{
        self.template op<mv_rowmajor>(other.template op<col>(0))}
        .template op<transpose>();
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 2>, M>
  operator()(const TV_METAL_THREAD array<array<T, K>, M> &self,
             const TV_METAL_THREAD array<array<T, 2>, K> &other) {
    return array<array<T, M>, 2>{
        self.template op<mv_rowmajor>(other.template op<col>(0)),
        self.template op<mv_rowmajor>(other.template op<col>(1))}
        .template op<transpose>();
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, M>
  operator()(const TV_METAL_THREAD array<array<T, K>, M> &self,
             const TV_METAL_THREAD array<array<T, 3>, K> &other) {
    return array<array<T, M>, 3>{
        self.template op<mv_rowmajor>(other.template op<col>(0)),
        self.template op<mv_rowmajor>(other.template op<col>(1)),
        self.template op<mv_rowmajor>(other.template op<col>(2))}
        .template op<transpose>();
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 4>, M>
  operator()(const TV_METAL_THREAD array<array<T, K>, M> &self,
             const TV_METAL_THREAD array<array<T, 4>, K> &other) {
    return array<array<T, M>, 4>{
        self.template op<mv_rowmajor>(other.template op<col>(0)),
        self.template op<mv_rowmajor>(other.template op<col>(1)),
        self.template op<mv_rowmajor>(other.template op<col>(2)),
        self.template op<mv_rowmajor>(other.template op<col>(3))}
        .template op<transpose>();
  }
};

template <typename T, size_t M, size_t K> struct mm_tnn<array<T, K>, M, 0> {
  // tnt
  // MK @ NK = MN
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, 1>
  operator()(const TV_METAL_THREAD array<array<T, K>, M> &self,
             const TV_METAL_THREAD array<array<T, K>, 1> &other) {
    return array<array<T, M>, 1>{
        self.template op<mv_rowmajor>(other.template op<row>(0))};
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, 2>
  operator()(const TV_METAL_THREAD array<array<T, K>, M> &self,
             const TV_METAL_THREAD array<array<T, K>, 2> &other) {
    return array<array<T, M>, 2>{
        self.template op<mv_rowmajor>(other.template op<row>(0)),
        self.template op<mv_rowmajor>(other.template op<row>(1))};
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, 3>
  operator()(const TV_METAL_THREAD array<array<T, K>, M> &self,
             const TV_METAL_THREAD array<array<T, K>, 3> &other) {
    return array<array<T, M>, 3>{
        self.template op<mv_rowmajor>(other.template op<row>(0)),
        self.template op<mv_rowmajor>(other.template op<row>(1)),
        self.template op<mv_rowmajor>(other.template op<row>(2))};
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, 4>
  operator()(const TV_METAL_THREAD array<array<T, K>, M> &self,
             const TV_METAL_THREAD array<array<T, K>, 4> &other) {
    return array<array<T, M>, 4>{
        self.template op<mv_rowmajor>(other.template op<row>(0)),
        self.template op<mv_rowmajor>(other.template op<row>(1)),
        self.template op<mv_rowmajor>(other.template op<row>(2)),
        self.template op<mv_rowmajor>(other.template op<row>(3))};
  }
};

template <typename T, size_t M, size_t K> struct mm_ntn<array<T, M>, K, 0> {
  // ntn
  // KM @ KN = NM
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, 1>
  operator()(const TV_METAL_THREAD array<array<T, M>, K> &self,
             const TV_METAL_THREAD array<array<T, 1>, K> &other) {
    return array<array<T, M>, 1>{
        self.template op<mv_colmajor>(other.template op<col>(0))};
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, 2>
  operator()(const TV_METAL_THREAD array<array<T, M>, K> &self,
             const TV_METAL_THREAD array<array<T, 2>, K> &other) {
    return array<array<T, M>, 2>{
        self.template op<mv_colmajor>(other.template op<col>(0)),
        self.template op<mv_colmajor>(other.template op<col>(1))};
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, 3>
  operator()(const TV_METAL_THREAD array<array<T, M>, K> &self,
             const TV_METAL_THREAD array<array<T, 3>, K> &other) {
    return array<array<T, M>, 3>{
        self.template op<mv_colmajor>(other.template op<col>(0)),
        self.template op<mv_colmajor>(other.template op<col>(1)),
        self.template op<mv_colmajor>(other.template op<col>(2))};
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, 4>
  operator()(const TV_METAL_THREAD array<array<T, M>, K> &self,
             const TV_METAL_THREAD array<array<T, 4>, K> &other) {
    return array<array<T, M>, 4>{
        self.template op<mv_colmajor>(other.template op<col>(0)),
        self.template op<mv_colmajor>(other.template op<col>(1)),
        self.template op<mv_colmajor>(other.template op<col>(2)),
        self.template op<mv_colmajor>(other.template op<col>(3))};
  }
};

// V' = A @ V @ A.T
template <typename T, size_t N, size_t Align>
struct variance_transform_ttt; // A @ B @ A.T
template <typename T, size_t N, size_t Align>
struct variance_transform_nnn; // A @ B @ A.T
template <typename T, size_t N, size_t Align>
struct variance_transform_ttt_grad_lfs;
template <typename T, size_t N, size_t Align>
struct variance_transform_ttt_grad_rfs;
template <typename T, size_t N, size_t Align>
struct variance_transform_nnn_grad_lfs;
template <typename T, size_t N, size_t Align>
struct variance_transform_nnn_grad_rfs;
template <typename T, size_t N, size_t Align>
struct symmetric_variance_transform_nnn_grad_lfs;

template <typename T, size_t K, size_t M>
struct variance_transform_ttt<array<T, K>, M, 0> {
  // A: MK, B: KK
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, M>
  operator()(const TV_METAL_THREAD array<array<T, K>, M> &self,
             const TV_METAL_THREAD array<array<T, K>, K> &other) {
    return self.template op<mm_ttt>(other).template op<mm_ttt>(
        self.template op<transpose>());
  }
};

template <typename T, size_t K, size_t M>
struct variance_transform_nnn<array<T, M>, K, 0> {
  // A: KM, B: KK
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, M>
  operator()(const TV_METAL_THREAD array<array<T, M>, K> &self,
             const TV_METAL_THREAD array<array<T, K>, K> &other) {
    return self.template op<mm_nnn>(other).template op<mm_nnn>(
        self.template op<transpose>());
  }
};

template <typename T, size_t M>
struct variance_transform_ttt_grad_lfs<array<T, M>, M, 0> {
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, K>, M>
  operator()(const TV_METAL_THREAD array<array<T, M>, M> &G,
             const TV_METAL_THREAD array<array<T, K>, M> &A,
             const TV_METAL_THREAD array<array<T, K>, K> &B) {
    // Q = ABA^T
    // dL/dA = dL/dQ dQ/dA
    // assume dL/dQ = G
    // then dL/dA = G^T A B + G A B^T
    // see matrix cookbook
    // https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf if B = I, then
    // dQ/dA = (G + G^T) A
    auto tmp =
        G.template op<transpose>().template op<mm_ttt>(A).template op<mm_ttt>(
            B);
    auto tmp2 = G.template op<mm_ttt>(
        A.template op<mm_ttt>(B.template op<transpose>()));
    return tmp + tmp2;
  }
};

template <typename T, size_t M>
struct variance_transform_ttt_grad_rfs<array<T, M>, M, 0> {
  // A: KM, B: KK
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, K>, K>
  operator()(const TV_METAL_THREAD array<array<T, M>, M> &grad,
             const TV_METAL_THREAD array<array<T, K>, M> &self,
             const TV_METAL_THREAD array<array<T, K>, K> &other) {
    return grad.template op<mm_ttt>(self)
        .template op<transpose>()
        .template op<mm_ttt>(self)
        .template op<transpose>();
  }
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, K>, K>
  operator()(const TV_METAL_THREAD array<array<T, M>, M> &grad,
             const TV_METAL_THREAD array<array<T, K>, M> &self) {
    return grad.template op<mm_ttt>(self)
        .template op<transpose>()
        .template op<mm_ttt>(self)
        .template op<transpose>();
  }
};

template <typename T, size_t M>
struct variance_transform_nnn_grad_lfs<array<T, M>, M, 0> {
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, K>
  operator()(const TV_METAL_THREAD array<array<T, M>, M> &G,
             const TV_METAL_THREAD array<array<T, M>, K> &A,
             const TV_METAL_THREAD array<array<T, K>, K> &B) {
    // Q = ABA^T
    // dL/dA = dL/dQ dQ/dA
    // assume dL/dQ = G
    // then dL/dA = G^T A B + G A B^T
    // see matrix cookbook
    // https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf if B = I, then
    // dQ/dA = (G + G^T) A
    // if B is symmetric, then dQ/dA = (G + G^T) A B
    auto tmp =
        G.template op<transpose>().template op<mm_nnn>(A).template op<mm_nnn>(
            B);
    auto tmp2 = G.template op<mm_nnn>(
        A.template op<mm_nnn>(B.template op<transpose>()));
    return tmp + tmp2;
  }
};

template <typename T, size_t M>
struct symmetric_variance_transform_nnn_grad_lfs<array<T, M>, M, 0> {
  // dL/dA = G^T A B + G A B^T
  // B is symmetric, so dQ/dA = (G + G^T) A B
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, K>
  operator()(const TV_METAL_THREAD array<array<T, M>, M> &G,
             const TV_METAL_THREAD array<array<T, M>, K> &A,
             const TV_METAL_THREAD array<array<T, K>, K> &B) {
    return (G + G.template op<transpose>())
        .template op<mm_nnn>(A)
        .template op<mm_nnn>(B);
  }
};

template <typename T, size_t M>
struct variance_transform_nnn_grad_rfs<array<T, M>, M, 0> {
  // A: KM, B: KK
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, K>, K>
  operator()(const TV_METAL_THREAD array<array<T, M>, M> &grad,
             const TV_METAL_THREAD array<array<T, M>, K> &self,
             const TV_METAL_THREAD array<array<T, K>, K> &other) {
    return grad.template op<mm_nnn>(self)
        .template op<transpose>()
        .template op<mm_nnn>(self)
        .template op<transpose>();
  }
  template <size_t K>
  TV_HOST_DEVICE_INLINE constexpr array<array<T, K>, K>
  operator()(const TV_METAL_THREAD array<array<T, M>, M> &grad,
             const TV_METAL_THREAD array<array<T, M>, K> &self) {
    return grad.template op<mm_nnn>(self)
        .template op<transpose>()
        .template op<mm_nnn>(self)
        .template op<transpose>();
  }
};

// A' = A @ A.T
template <typename T, size_t N, size_t Align>
struct identity_variance_transform_nnn;
template <typename T, size_t N, size_t Align>
struct identity_variance_transform_ttt;
template <typename T, size_t N, size_t Align>
struct identity_variance_transform_nnn_grad;
template <typename T, size_t N, size_t Align>
struct identity_variance_transform_ttt_grad;

template <typename T, size_t K, size_t M>
struct identity_variance_transform_ttt<array<T, K>, M, 0> {
  // A: MK, B: KK
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, M>
  operator()(const TV_METAL_THREAD array<array<T, K>, M> &self) {
    return self.template op<mm_ttt>(self.template op<transpose>());
  }
};

template <typename T, size_t K, size_t M>
struct identity_variance_transform_nnn<array<T, M>, K, 0> {
  // A: KM, B: KK
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, M>
  operator()(const TV_METAL_THREAD array<array<T, M>, K> &self) {
    return self.template op<mm_nnn>(self.template op<transpose>());
  }
};

template <typename T, size_t K, size_t M>
struct identity_variance_transform_ttt_grad<array<T, K>, M, 0> {
  // A: MK, B: KK
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, M>
  operator()(const TV_METAL_THREAD array<array<T, M>, M> &G,
             const TV_METAL_THREAD array<array<T, K>, M> &self) {
    return (G + G.template op<transpose>()).template op<mm_ttt>(self);
  }
};

template <typename T, size_t K, size_t M>
struct identity_variance_transform_nnn_grad<array<T, M>, K, 0> {
  // A: KM, B: KK
  TV_HOST_DEVICE_INLINE constexpr array<array<T, M>, M>
  operator()(const TV_METAL_THREAD array<array<T, M>, M> &G,
             const TV_METAL_THREAD array<array<T, M>, K> &self) {
    return (G + G.template op<transpose>()).template op<mm_nnn>(self);
  }
};

namespace arrayops_detail {
template <typename T, size_t Row, size_t Col> struct transform_3d_impl {
  template <int... Inds>
  TV_HOST_DEVICE_INLINE static constexpr array<array<T, Col>, Row>
  run(const TV_METAL_THREAD array<array<T, Col>, Row> &a,
      const TV_METAL_THREAD array<array<T, 3>, 3> &b,
      mp_list_int<Inds...>) noexcept {
    return array<array<T, Col>, Row>{
        (concat(b.template op<mv_rowmajor>(slice<0, 3>(a[Inds])),
                slice<3, Col>(a[Inds])))...};
  }
};

template <typename T, size_t Row> struct transform_3d_impl<T, Row, 3> {
  template <int... Inds>
  TV_HOST_DEVICE_INLINE static constexpr array<array<T, 3>, Row>
  run(const TV_METAL_THREAD array<array<T, 3>, Row> &a,
      const TV_METAL_THREAD array<array<T, 3>, 3> &b,
      mp_list_int<Inds...>) noexcept {
    return array<array<T, 3>, Row>{(b.template op<mv_rowmajor>(a[Inds]))...};
  }
};

template <typename T, size_t Row, size_t Col> struct add_offset_impl {
  template <int... Inds>
  TV_HOST_DEVICE_INLINE static constexpr array<array<T, Col>, Row>
  run(const TV_METAL_THREAD array<array<T, Col>, Row> &a,
      const TV_METAL_THREAD array<T, 3> &b, mp_list_int<Inds...>) noexcept {
    return array<array<T, Col>, Row>{
        concat((slice<0, 3>(a[Inds]) + b), slice<3, Col>(a[Inds]))...};
  }
};

template <typename T, size_t Row> struct add_offset_impl<T, Row, 3> {
  template <int... Inds>
  TV_HOST_DEVICE_INLINE static constexpr array<array<T, 3>, Row>
  run(const TV_METAL_THREAD array<array<T, 3>, Row> &a,
      const TV_METAL_THREAD array<T, 3> &b, mp_list_int<Inds...>) noexcept {
    return array<array<T, 3>, Row>{(a[Inds] + b)...};
  }
};

} // namespace arrayops_detail

template <typename T, size_t N, size_t Align> struct add_offset_3d;

template <typename T, size_t Row, size_t Col>
struct add_offset_3d<array<T, Col>, Row, 0> {
public:
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Col>, Row>
  operator()(const TV_METAL_THREAD array<array<T, Col>, Row> &self,
             const TV_METAL_THREAD array<T, 3> &other) {
    return arrayops_detail::add_offset_impl<T, Row, Col>::run(
        self, other, mp_make_list_c_sequence<int, Row>{});
  }
};

template <typename T, size_t Col> struct add_offset_3d<T, Col, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const TV_METAL_THREAD array<T, Col> &self,
             const TV_METAL_THREAD array<T, 3> &other) {
    return concat((slice<0, 3>(self) + other), slice<3, Col>(self));
  }
};

template <typename T> struct add_offset_3d<T, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const TV_METAL_THREAD array<T, 3> &self,
             const TV_METAL_THREAD array<T, 3> &other) {
    return self + other;
  }
};

template <typename T, size_t N, size_t Align> struct transform_3d;

template <typename T, size_t Row, size_t Col>
struct transform_3d<array<T, Col>, Row, 0> {
public:
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Col>, Row>
  operator()(const TV_METAL_THREAD array<array<T, Col>, Row> &self,
             const TV_METAL_THREAD array<array<T, 3>, 3> &other) {
    return arrayops_detail::transform_3d_impl<T, Row, Col>::run(
        self, other, mp_make_list_c_sequence<int, Row>{});
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, Col>, Row>
  operator()(const TV_METAL_THREAD array<array<T, Col>, Row> &self,
             const TV_METAL_THREAD array<array<T, 4>, 4> &other) {
    return arrayops_detail::transform_3d_impl<T, Row, Col>::run(
               self, slice_2d<0, 0, 3, 3>(other),
               mp_make_list_c_sequence<int, Row>{})
        .template op<add_offset_3d>(slice<0, 3>(other.template op<col>(3)));
  }
};

template <typename T, size_t Col> struct transform_3d<T, Col, 0> {
public:
  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const TV_METAL_THREAD array<T, Col> &self,
             const TV_METAL_THREAD array<array<T, 3>, 3> &other) {
    return concat(other.template op<mv_rowmajor>(slice<0, 3>(self)),
                  slice<3, Col>(self));
  }

  TV_HOST_DEVICE_INLINE constexpr array<T, Col>
  operator()(const TV_METAL_THREAD array<T, Col> &self,
             const TV_METAL_THREAD array<array<T, 4>, 4> &other) {
    return operator()(self, {slice<0, 3>(other[0]), slice<0, 3>(other[1]),
                             slice<0, 3>(other[2])})
        .template op<add_offset_3d>(slice<0, 3>(other.template op<col>(3)));
  }
};

template <typename T> struct transform_3d<T, 3, 0> {
public:
  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const TV_METAL_THREAD array<T, 3> &self,
             const TV_METAL_THREAD array<array<T, 3>, 3> &other) {
    return other.template op<mv_rowmajor>(self);
  }

  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const TV_METAL_THREAD array<T, 3> &self,
             const TV_METAL_THREAD array<array<T, 4>, 4> &other) {
    return operator()(self, {slice<0, 3>(other[0]), slice<0, 3>(other[1]),
                             slice<0, 3>(other[2])})
        .template op<add_offset_3d>(slice<0, 3>(other.template op<col>(3)));
  }
};

template <typename T, size_t N, size_t Align> struct transform_matrix;

template <typename T> struct transform_matrix<array<T, 3>, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 4>, 4>
  operator()(const TV_METAL_THREAD array<array<T, 3>, 3> &self,
             const TV_METAL_THREAD array<T, 3> &other) {
    return {array<T, 4>{self[0][0], self[0][1], self[0][2], other[0]},
            array<T, 4>{self[1][0], self[1][1], self[1][2], other[1]},
            array<T, 4>{self[2][0], self[2][1], self[2][2], other[2]},
            array<T, 4>{T(0), T(0), T(0), T(1)}};
  }
};

template <typename T, size_t N, size_t Align> struct transform_matrix_colmajor_inverse;
template <typename T, size_t N, size_t Align> struct transform_matrix_colmajor_inverse_grad;

template <typename T, size_t N, size_t Align> struct transform_matrix_mm_nnn;
template <typename T, size_t N, size_t Align> struct transform_matrix_mm_nnn_grad_lfs;
template <typename T, size_t N, size_t Align> struct transform_matrix_mm_nnn_grad_rfs;

template <typename T> struct transform_matrix_colmajor_inverse<array<T, 3>, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 4>
  operator()(const TV_METAL_THREAD array<array<T, 3>, 4> &self) {
    return {array<T, 3>{self[0][0], self[1][0], self[2][0]},
            array<T, 3>{self[0][1], self[1][1], self[2][1]},
            array<T, 3>{self[0][2], self[1][2], self[2][2]},
            -slice<0, 3>(self).template op<mv_rowmajor>(self[3])};
  }
};

template <typename T> struct transform_matrix_colmajor_inverse_grad<array<T, 3>, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 4>
  operator()(const TV_METAL_THREAD array<array<T, 3>, 4> &grad, const TV_METAL_THREAD array<array<T, 3>, 4> &self) {
    auto grad_R_T = slice<0, 3>(grad).template op<transpose>() - grad[3].template op<mv_rowmajor_grad_lfs>(self[3]);
    return concat(grad_R_T, reshape<1, 3>(-slice<0, 3>(self).template op<mv_colmajor>(grad[3])));
  }
};


template <typename T> struct transform_matrix_mm_nnn<array<T, 3>, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 4>
  // tr: self @ other
  // y = self.R @ (other.R @ x + other.t) + self.t
  //   = self.R @ other.R @ x + self.R @ other.t + self.t
  operator()(const TV_METAL_THREAD array<array<T, 3>, 4> &self, const TV_METAL_THREAD array<array<T, 3>, 4> &other) {
    return concat(slice<0, 3>(self).template op<mm_nnn>(slice<0, 3>(other)),
                  reshape<1, 3>(slice<0, 3>(self).template op<mv_colmajor>(other[3]) + self[3]));
  }
};

template <typename T> struct transform_matrix_mm_nnn_grad_lfs<array<T, 3>, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 4>
  // tr: self @ other
  // y = self.R @ (other.R @ x + other.t) + self.t
  //   = self.R @ other.R @ x + self.R @ other.t + self.t
  operator()(const TV_METAL_THREAD array<array<T, 3>, 4> &grad, const TV_METAL_THREAD array<array<T, 3>, 4> &self, const TV_METAL_THREAD array<array<T, 3>, 4> &other) {
    auto grad_R = slice<0, 3>(grad);
    auto other_R = slice<0, 3>(other);
    auto self_R_grad = (grad_R.template op<mm_nnn_grad_lfs>(other_R)
       + grad[3].template op<mv_colmajor_grad_lfs>(other[3]));
    auto self_T_grad = grad[3];
    return concat(self_R_grad, reshape<1, 3>(self_T_grad));
  }
  
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 4>
  operator()(const TV_METAL_THREAD array<array<T, 3>, 4> &grad, const TV_METAL_THREAD array<array<T, 3>, 4> &other) {
    auto grad_R = slice<0, 3>(grad);
    auto other_R = slice<0, 3>(other);
    auto self_R_grad = (grad_R.template op<mm_nnn_grad_lfs>(other_R)
       + grad[3].template op<mv_colmajor_grad_lfs>(other[3]));
    auto self_T_grad = grad[3];
    return concat(self_R_grad, reshape<1, 3>(self_T_grad));
  }

};

template <typename T> struct transform_matrix_mm_nnn_grad_rfs<array<T, 3>, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 4>
  // tr: self @ other
  // y = self.R @ (other.R @ x + other.t) + self.t
  //   = self.R @ other.R @ x + self.R @ other.t + self.t
  operator()(const TV_METAL_THREAD array<array<T, 3>, 4> &grad, const TV_METAL_THREAD array<array<T, 3>, 4> &self, const TV_METAL_THREAD array<array<T, 3>, 4> &other) {
    auto grad_R = slice<0, 3>(grad);
    auto self_R = slice<0, 3>(self);
    auto other_R_grad = grad_R.template op<mm_nnn_grad_rfs>(self_R);
    auto other_T_grad = grad[3].template op<mv_colmajor_grad_rfs>(self_R);
    return concat(other_R_grad, reshape<1, 3>(other_T_grad));
  }
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 4>
  operator()(const TV_METAL_THREAD array<array<T, 3>, 4> &grad, const TV_METAL_THREAD array<array<T, 3>, 4> &self) {
    auto grad_R = slice<0, 3>(grad);
    auto self_R = slice<0, 3>(self);
    auto other_R_grad = grad_R.template op<mm_nnn_grad_rfs>(self_R);
    auto other_T_grad = grad[3].template op<mv_colmajor_grad_rfs>(self_R);
    return concat(other_R_grad, reshape<1, 3>(other_T_grad));
  }

};

template <typename T, size_t N, size_t Align> struct qxdir;
template <typename T, size_t N, size_t Align> struct qydir;
template <typename T, size_t N, size_t Align> struct qzdir;

template <typename T, size_t N, size_t Align> struct uqxdir;
template <typename T, size_t N, size_t Align> struct uqydir;
template <typename T, size_t N, size_t Align> struct uqzdir;

template <typename T, size_t N, size_t Align> struct qangle;
template <typename T, size_t N, size_t Align> struct qaxis;

template <typename T, size_t N, size_t Align> struct uqxdir_grad;
template <typename T, size_t N, size_t Align> struct uqydir_grad;
template <typename T, size_t N, size_t Align> struct uqzdir_grad;

template <typename T> struct qxdir<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const TV_METAL_THREAD array<T, 4> &q) {
    return {q[3] * q[3] + q[0] * q[0] - q[1] * q[1] - q[2] * q[2],
            (q[0] * q[1] + q[2] * q[3]) * T(2),
            (q[2] * q[0] - q[1] * q[3]) * T(2)};
  }
};

template <typename T> struct uqxdir<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const TV_METAL_THREAD array<T, 4> &q) {
    return {T(1) - 2 * q[1] * q[1] - 2 * q[2] * q[2],
            (q[0] * q[1] + q[2] * q[3]) * T(2),
            (q[2] * q[0] - q[1] * q[3]) * T(2)};
  }
};

template <typename T> struct uqxdir_grad<T, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 4>
  operator()(const TV_METAL_THREAD array<T, 3> &dqdir,
             const TV_METAL_THREAD array<T, 4> &q) {
    return {T(2) * q[1] * dqdir[1] + T(2) * q[2] * dqdir[2], // dL/dq0
            -T(4) * q[1] * dqdir[0] + T(2) * q[0] * dqdir[1] -
                T(2) * q[3] * dqdir[2], // dL/dq1
            -T(4) * q[2] * dqdir[0] + T(2) * q[3] * dqdir[1] +
                T(2) * q[0] * dqdir[2], // dL/dq2
            T(2) * q[2] * dqdir[1] - T(2) * q[1] * dqdir[2]};
  }
};

template <typename T> struct qydir<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const TV_METAL_THREAD array<T, 4> &q) {
    return {(q[0] * q[1] - q[2] * q[3]) * T(2),
            q[3] * q[3] - q[0] * q[0] + q[1] * q[1] - q[2] * q[2],
            (q[1] * q[2] + q[0] * q[3]) * T(2)};
  }
};
template <typename T> struct uqydir<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const TV_METAL_THREAD array<T, 4> &q) {
    return {(q[0] * q[1] - q[2] * q[3]) * T(2),
            T(1) - T(2) * q[0] * q[0] - T(2) * q[2] * q[2],
            (q[1] * q[2] + q[0] * q[3]) * T(2)};
  }
};

template <typename T> struct uqydir_grad<T, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 4>
  operator()(const TV_METAL_THREAD array<T, 3> &dqdir,
             const TV_METAL_THREAD array<T, 4> &q) {
    return {dqdir[0] * T(2) * q[1] - dqdir[1] * T(4) * q[0] +
                dqdir[2] * T(2) * q[3],                      // dL/dq0
            T(2) * q[0] * dqdir[0] + T(2) * q[2] * dqdir[2], // dL/dq1
            dqdir[0] * T(-2) * q[3] - dqdir[1] * T(4) * q[2] +
                T(2) * q[1] * dqdir[2], // dL/dq2
            T(-2) * q[2] * dqdir[0] + T(2) * q[0] * dqdir[2]};
  }
};

template <typename T> struct qzdir<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const TV_METAL_THREAD array<T, 4> &q) {
    return {(q[2] * q[0] + q[1] * q[3]) * T(2),
            (q[1] * q[2] - q[0] * q[3]) * T(2),
            q[3] * q[3] - q[0] * q[0] - q[1] * q[1] + q[2] * q[2]};
  }
};

template <typename T> struct uqzdir<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const TV_METAL_THREAD array<T, 4> &q) {
    return {(q[2] * q[0] + q[1] * q[3]) * T(2),
            (q[1] * q[2] - q[0] * q[3]) * T(2),
            T(1) - 2 * q[0] * q[0] - 2 * q[1] * q[1]};
  }
};

template <typename T> struct uqzdir_grad<T, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 4>
  operator()(const TV_METAL_THREAD array<T, 3> &dqdir,
             const TV_METAL_THREAD array<T, 4> &q) {
    return {
        T(2) * q[2] * dqdir[0] - T(2) * q[3] * dqdir[1] -
            T(4) * q[0] * dqdir[2], // dL/dq0
        T(2) * q[3] * dqdir[0] + T(2) * q[2] * dqdir[1] -
            T(4) * q[1] * dqdir[2],                      // dL/dq1
        T(2) * q[0] * dqdir[0] + T(2) * q[1] * dqdir[1], // dL/dq1
        T(2) * q[1] * dqdir[0] - T(2) * q[0] * dqdir[1]  // dL/dq1
    };
  }
};

template <typename T> struct qaxis<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3>
  operator()(const TV_METAL_THREAD array<T, 4> &q) {
    return slice<0, 3>(q).template op<normalize>();
  }
};

template <typename T> struct qangle<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr T
  operator()(const TV_METAL_THREAD array<T, 4> &q) {

    return MathScalarOp<T>::atan2(slice<0, 3>(q).template op<l2norm>(), q[3]) *
           T(2);
  }
};

template <typename T, size_t N, size_t Align> struct rotation_quat;

template <typename T> struct rotation_quat<T, 3, 0> {
  TV_HOST_DEVICE_INLINE array<T, 4>
  operator()(const TV_METAL_THREAD array<T, 3> &self, T angle) {
    return concat(self * MathScalarOp<T>::sin(angle / 2),
                  create_array(MathScalarOp<T>::cos(angle / 2)));
  }
};

// template <typename T, size_t N, size_t Align> struct
// rotation_quat_matrix_simple;

// template <typename T> struct rotation_quat_matrix_simple<array<T, 3>, 3, 0> {
//   TV_HOST_DEVICE_INLINE array<T, 4>
//   operator()(const TV_METAL_THREAD array<array<T, 3>, 3> &m) {
//     array<T, 4> q{m[0][0]-m[1][1]-m[2][2], m[1][1]-m[0][0]-m[2][2],
//     m[2][2]-m[0][0]-m[1][1], m[0][0]+m[1][1]+m[2][2]}; array<array<T, 4>, 4>
//     s{
//         array<T, 4>{T(1), m[0][1] + m[1][0], m[2][0] + m[0][2], m[2][1] -
//         m[1][2]}, array<T, 4>{m[0][1] + m[1][0], T(1), m[1][2] + m[2][1],
//         m[0][2] - m[2][0]}, array<T, 4>{m[0][2] + m[2][0], m[1][2] + m[2][1],
//         T(1), m[1][0] - m[0][1]}, array<T, 4>{m[2][1] - m[1][2], m[0][2] -
//         m[2][0], m[1][0] - m[0][1], T(1)}
//     };

//     int j = 0;
//     for (int i = 0; i < 4; ++i){
//       if(q[i] > q[j])
//         j = i;
//     }
//     return apply(MathScalarOp<T>::copysign, apply(MathScalarOp<T>::sqrt, (q +
//     T(1)).template op<arrayops::max>(T(0))).template op<normalize>(), s[j]);
//   }
// };

template <typename T> struct rotation_quat<array<T, 3>, 3, 0> {
  TV_HOST_DEVICE_INLINE array<T, 4>
  operator()(const TV_METAL_THREAD array<array<T, 3>, 3> &m) {
    T t = m[0][0] + m[1][1] + m[2][2];
    array<T, 4> q;
    if (t > T(0)) {
      t = MathScalarOp<T>::sqrt(t + T(1.0));
      q[3] = T(0.5) * t;
      t = T(0.5) / t;
      q[0] = (m[2][1] - m[1][2]) * t;
      q[1] = (m[0][2] - m[2][0]) * t;
      q[2] = (m[1][0] - m[0][1]) * t;
    } else {
      int i = 0;
      if (m[1][1] > m[0][0])
        i = 1;
      if (m[2][2] > m[i][i])
        i = 2;
      int j = (i + 1) % 3;
      int k = (j + 1) % 3;

      t = MathScalarOp<T>::sqrt(m[i][i] - m[j][j] - m[k][k] + T(1.0));
      q[i] = T(0.5) * t;
      q[i] = T(0.5) * t;
      t = T(0.5) / t;
      q[3] = (m[k][j] - m[j][k]) * t;
      q[j] = (m[j][i] + m[i][j]) * t;
      q[k] = (m[k][i] + m[i][k]) * t;
    }

    return q;
  }
};

template <typename T, size_t N, size_t Align> struct qmat;

template <typename T> struct qmat<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 3>
  operator()(const TV_METAL_THREAD array<T, 4> &self) {
    return (array<array<T, 3>, 3>{self.template op<qxdir>(),
                                  self.template op<qydir>(),
                                  self.template op<qzdir>()})
        .template op<transpose>();
  }
};

template <typename T, size_t N, size_t Align> struct qmat_colmajor;

template <typename T> struct qmat_colmajor<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 3>
  operator()(const TV_METAL_THREAD array<T, 4> &self) {
    return array<array<T, 3>, 3>{self.template op<qxdir>(),
                                 self.template op<qydir>(),
                                 self.template op<qzdir>()};
  }
};

template <typename T, size_t N, size_t Align> struct uqmat_colmajor;

template <typename T> struct uqmat_colmajor<T, 4, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<array<T, 3>, 3>
  operator()(const TV_METAL_THREAD array<T, 4> &self) {
    return array<array<T, 3>, 3>{self.template op<uqxdir>(),
                                 self.template op<uqydir>(),
                                 self.template op<uqzdir>()};
  }
};

template <typename T, size_t N, size_t Align> struct uqmat_colmajor_grad;

template <typename T> struct uqmat_colmajor_grad<array<T, 3>, 3, 0> {
  TV_HOST_DEVICE_INLINE constexpr array<T, 4>
  operator()(const TV_METAL_THREAD array<array<T, 3>, 3> &grad,
             const TV_METAL_THREAD array<T, 4> &q) {
    return grad[0].template op<uqxdir_grad>(q) +
           grad[1].template op<uqydir_grad>(q) +
           grad[2].template op<uqzdir_grad>(q);
  }
};

template <typename T, size_t N, size_t Align> struct angleaxis_mat;

template <typename T> struct angleaxis_mat<T, 3, 0> {
  TV_HOST_DEVICE_INLINE array<array<T, 3>, 3>
  operator()(const TV_METAL_THREAD array<T, 3> &m_axis, T m_angle) {
    array<array<T, 3>, 3> res;
    array<T, 3> sin_axis = MathScalarOp<T>::sin(m_angle) * m_axis;
    T c = MathScalarOp<T>::cos(m_angle);
    array<T, 3> cos1_axis = (T(1) - c) * m_axis;

    T tmp;
    tmp = cos1_axis[0] * m_axis[1];
    res[0][1] = tmp - sin_axis[2];
    res[1][0] = tmp + sin_axis[2];

    tmp = cos1_axis[0] * m_axis[2];
    res[0][2] = tmp + sin_axis[1];
    res[2][0] = tmp - sin_axis[1];

    tmp = cos1_axis[1] * m_axis[2];
    res[1][2] = tmp - sin_axis[0];
    res[2][1] = tmp + sin_axis[0];
    auto tmp2 = cos1_axis * m_axis;
    res[0][0] = tmp2[0] + c;
    res[1][1] = tmp2[1] + c;
    res[2][2] = tmp2[2] + c;
    return res;
  }

  TV_HOST_DEVICE_INLINE array<array<T, 3>, 3>
  operator()(const TV_METAL_THREAD array<T, 3> &m_axis) {
    T angle = m_axis.template op<length>();
    if (angle == 0) {
      auto res = array<array<T, 3>, 3>{};
      res[0][0] = res[1][1] = res[2][2] = T(1);
      return res;
    }
    return operator()(m_axis / angle, angle);
  }
};

template <typename T, size_t N, size_t Align> struct uangle {
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self,
             const TV_METAL_THREAD array<T, N, Align> &other) {
    T d = self.template op<dot>(other);
    return d > 1 ? 0 : MathScalarOp<T>::acos(d < -1 ? -1 : d);
  }
};

template <typename T, size_t N, size_t Align> struct angle {
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self,
             const TV_METAL_THREAD array<T, N, Align> &other) {
    return self.template op<normalize>().template op<uangle>(
        other.template op<normalize>());
  }
};

namespace detail {
template <class A, class B, class C>
TV_HOST_DEVICE_INLINE constexpr auto
lerp(A a, B b, C c) -> decltype(a * (1 - c) + b * c) {
  return a * (1 - c) + b * c;
}
} // namespace detail

template <typename T, size_t N, size_t Align> struct nlerp {
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self,
             const TV_METAL_THREAD array<T, N, Align> &other, T t) {
    return apply(detail::lerp<T, T, T>, self, other, t)
        .template op<normalize>();
  }
};

template <typename T, size_t N, size_t Align> struct slerp {
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &a,
             const TV_METAL_THREAD array<T, N, Align> &b, T t) {
    T th = a.template op<uangle>(b);
    return th == 0 ? a
                   : a * (MathScalarOp<T>::sin(th * (1 - t)) /
                          MathScalarOp<T>::sin(th)) +
                         b * (MathScalarOp<T>::sin(th * t) /
                              MathScalarOp<T>::sin(th));
  }
};

template <typename T, size_t N, size_t Align> struct qnlerp {
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, 4, Align> &a,
             const TV_METAL_THREAD array<T, 4, Align> &b, T t) {
    return a.template op<nlerp>(a.template op<dot>(b) < 0 ? -b : b, t);
  }
};

template <typename T, size_t N, size_t Align> struct qslerp {
  TV_HOST_DEVICE_INLINE auto
  operator()(const TV_METAL_THREAD array<T, 4, Align> &a,
             const TV_METAL_THREAD array<T, 4, Align> &b, T t) {
    return a.template op<slerp>(a.template op<dot>(b) < 0 ? -b : b, t);
  }
};

template <typename T, size_t N, size_t Align> struct qconj {
  TV_HOST_DEVICE_INLINE constexpr array<T, 4, Align>
  operator()(const TV_METAL_THREAD array<T, 4, Align> &q) {
    return {-q[0], -q[1], -q[2], q[3]};
  }
};

template <typename T, size_t N, size_t Align> struct qinv {
  TV_HOST_DEVICE_INLINE constexpr array<T, 4, Align>
  operator()(const TV_METAL_THREAD array<T, 4, Align> &q) {
    return q.template op<qconj>() / q.template op<length2>();
  }
};

template <typename T, size_t N, size_t Align> struct qexp {
  TV_HOST_DEVICE_INLINE constexpr array<T, 4, Align>
  operator()(const TV_METAL_THREAD array<T, 4, Align> &q) {
    const auto v = slice<0, 3>(q);
    const auto vv = v.template op<length>();
    return MathScalarOp<T>::exp(q.w) *
           array<T, 4, Align>{v * (vv > 0 ? MathScalarOp<T>::sin(vv) / vv : 0),
                              MathScalarOp<T>::cos(vv)};
  }
};

template <typename T, size_t N, size_t Align> struct normlineproj {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3, Align>
  operator()(const TV_METAL_THREAD array<T, 3, Align> &self,
             const TV_METAL_THREAD array<T, 3, Align> &origin,
             const TV_METAL_THREAD array<T, 3, Align> &norm_dir) {
    return origin + (self - origin).template op<dot>(norm_dir) * norm_dir;
  }
};

template <typename T, size_t N, size_t Align> struct lineproj {
  TV_HOST_DEVICE_INLINE constexpr array<T, 3, Align>
  operator()(const TV_METAL_THREAD array<T, 3, Align> &self,
             const TV_METAL_THREAD array<T, 3, Align> &origin,
             const TV_METAL_THREAD array<T, 3, Align> &dir) {
    return self.template op<normlineproj>(origin, dir.template op<normalize>());
  }
};

// constexpr tv::array_nd<float, 3, 3> a{tv::array<float, 3>{1, 2, 3},
//                                               tv::array<float, 3>{4, 5, 6},
//                                               tv::array<float, 3>{7, 8, 3}};
// constexpr tv::array_nd<float, 4, 3> a2{tv::array<float, 3>{1, 2, 3},
//                                               tv::array<float, 3>{4, 5, 6},
//                                               tv::array<float, 3>{7, 8, 3},
//                                               tv::array<float, 3>{3, 5, 2}};
// constexpr tv::array_nd<float, 8, 3> a3{};
// constexpr auto a_slice = slice_2d<0, 0, 2, 2>(a);
// constexpr tv::array_nd<float, 3, 3> b{tv::array<float, 3>{9, 7, 8},
//                                               tv::array<float, 3>{6, 5, 4},
//                                               tv::array<float, 3>{3, 2, 1}};
// constexpr tv::array_nd<float, 4, 4> b_4x4{tv::array<float, 4>{9, 7, 8, 0},
//                                               tv::array<float, 4>{6, 5, 4,
//                                               0}, tv::array<float, 4>{3, 2,
//                                               1, 0}, tv::array<float, 4>{0,
//                                               0, 0, 1}};

// constexpr auto c = a.op<mm_tt>(b);
// constexpr int rank = detail::get_tv_array_rank<decltype(a)>::value;
// // constexpr auto inv_a = a.op<row>(0);
// constexpr auto x = a.op<determinant>();
// constexpr auto y = a.op<inverse>();

// constexpr tv::array<tv::array<float, 3UL>, 3UL> inv_a2 = a.op<adjugate>();
// constexpr auto rtx =
//     tv::arrayops::apply(tv::detail::array_div<float>, inv_a2, x);

// constexpr auto rtx2 = inv_a2 / x;

// constexpr auto cc = a.op<transform_3d>(b_4x4);
// constexpr auto cccc = a3.op<transform_3d>(b_4x4);

// constexpr auto ccc = a[0].op<transform_3d>(b);

} // namespace arrayops
} // namespace tv