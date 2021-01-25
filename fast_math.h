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

scalar fastest_sin(scalar x);
scalar fastest_cos(scalar x);

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
class point
{
public:

	point(const __m128& vS);
	point(const scalar& x = 0, const scalar& y = 0);

	const scalar& operator[](unsigned i) const;
	scalar& operator[](unsigned i);

	_MM_ALIGN16 __m128 _vS;

};

point operator- (const point& a, const point& b);
point operator+ (const point& a, const point& b);
point operator* (const point& a, const scalar& b);
point operator/ (const point& a, const scalar& b);

// construct from angle
point unit(const scalar& angle);

__m128 _mm_sin_ps(const __m128& a); // 4 sines approximations
void _mm_sincos_ps(__m128&s, __m128&c, const __m128& x); // 4 sines + 4 cosines

// HORISONTAL operations

// dot product
__m128 _mm_dot_ps(const __m128& a0, const __m128& a1, const __m128& b0, const __m128& b1);

// determinant, or cross product (signed area)
__m128 _mm_det_ps(const __m128& a0, const __m128& a1, const __m128& b0, const __m128& b1);



// qp is a query point, s0 is a first point of a segment, s0s1 is a vector from s0 to s1;
//scalar point_to_segment_squared_distance_impl(const point& qp, const point& s0, const point& s1);

// returns mask
__m128 _mm_circle_segment_collides(const __m128& s0qp0, const __m128& s0qp1, const __m128& s0s10, const __m128& s0s11, const __m128& rSqr);

// s0qp is a vector from s0(first point of a segment) to a qp(query point); s0s1 is a vector from s0 to s1; d is a unit direction vector, (originated at a querypoint)
__m128 _mm_ray_segment_distance_inverse(const __m128& s0qp0, const __m128& s0qp1, const __m128& s0s10, const __m128& s0s11, const __m128& d0, const __m128& d1);

// call this once at program start (to use tanh and sincos LUT)
void init_fast_math();


