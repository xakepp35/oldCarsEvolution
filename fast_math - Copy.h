#pragma once

#include <cstdint>
#include <xmmintrin.h>
#include <array>

typedef float scalar;

const scalar pi = (scalar) 3.1415926535897932384626433832795;
const scalar piDiv2 = (scalar) 1.5707963267948966192313216916398;

scalar sqr(const scalar& x);

scalar sgn_sigmf(const scalar& x);
scalar linv_sigmf(const scalar& x);
scalar sqinv_sigmf(const scalar& x);

// exponential sigmoid, it sucks abit :D
scalar posexp_sigmf(const scalar& x);

// tanh rocks!
scalar fastest_tanh(const scalar& x);
float fastest_atan2(float y, float x);

// various cases of dot products
float hsum_ps_sse3(__m128 v);
scalar vdot(const __m128* a, const __m128* b, scalar bias, size_t n); // optimized for 4*n weights
scalar vdot4(const __m128* a, const __m128* b); // optimized for 4 weights
scalar vdot8(const __m128* a, const __m128* b);  // optimized for 8 weights

// LinearAlgebra

_MM_ALIGN16
class point:
	public std::array< scalar, 2 > {
public:

	point(const scalar& x = 0, const scalar& y = 0);

};

point operator- (const point& a, const point& b);
point operator+ (const point& a, const point& b);
point operator* (const point& a, const scalar& b);
point operator/ (const point& a, const scalar& b);

typedef point hvec2f;

// dot product
scalar dot(const hvec2f& a, const hvec2f& b);

// determinant, or cross product (signed area)
scalar det(const hvec2f& a, const hvec2f& b);

// construct from angle, uses LUT for reading both sin and cos values in one memory read
point unit(const scalar& angle);
hvec2f hunit(const scalar& angle);

// qp is a query point, s0 is a first point of a segment, s0s1 is a vector from s0 to s1;
scalar point_to_segment_squared_distance_impl(const point& qp, const point& s0, const point& s1);

// s0qp is a vector from s0(first point of a segment) to a qp(query point); s0s1 is a vector from s0 to s1; d is a unit direction vector, (originated at a querypoint)
scalar get_distance_inverse_impl(const hvec2f& s0qp, const hvec2f& s0s1, const hvec2f& d);

// call this once at program start (to use tanh and sincos LUT)
void init_fast_math();


