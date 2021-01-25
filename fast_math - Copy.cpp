#include "fast_math.h"

#include <cmath>
#include <smmintrin.h>


using namespace std;

scalar sqr(const scalar& x) {
	return x*x;
}

scalar sgn_sigmf(const scalar& x) {
	if (x != 0)
		return (x > 0) ? scalar(1) : scalar(-1);
	else
		return 0;
	//return x / abs(x);
}

scalar linv_sigmf(const scalar& x) {
	return sgn_sigmf(x) * (scalar(1) - scalar(1) / (scalar(1) + abs(x)));
}

scalar sqinv_sigmf(const scalar& x) {
	return sgn_sigmf(x) * (scalar(1) - scalar(1) / (scalar(1) + x*x));
}

// exponential sigmoid
scalar posexp_sigmf(const scalar& x) {
	return ((scalar)1) / (((scalar)1) + exp(-x));
}

//  fast speed, high accuracy, also its derivatives at -3 and at 3 are equal zero
scalar fastest_tanh(const scalar& x) {
	if(x >= 3)
		return 1;
	if (x <= -3)
		return -1;
	return x * (x * x + 27) / (x * x * 9 + 27);
}

float fastest_atan2(float y, float x) {
	const float n1 = 0.97239411f;
	const float n2 = -0.19194795f;
	float result = 0.0f;
	if (x != 0) {
		const union { float flVal; uint32_t nVal; } tYSign = { y };
		const union { float flVal; uint32_t nVal; } tXSign = { x };
		if (fabsf(x) >= fabsf(y)) {
			union { float flVal; uint32_t nVal; } tOffset = { pi };
			// Add or subtract PI based on y's sign.
			tOffset.nVal |= tYSign.nVal & 0x80000000u;
			// No offset if x is positive, so multiply by 0 or based on x's sign.
			tOffset.nVal *= tXSign.nVal >> 31;
			result = tOffset.flVal;
			const float z = y / x;
			result += (n1 + n2 * z * z) * z;
		}
		else {// Use atan(y/x) = pi/2 - atan(x/y) if |y/x| > 1.
			union { float flVal; uint32_t nVal; } tOffset = { piDiv2 };
			// Add or subtract PI/2 based on y's sign.
			tOffset.nVal |= tYSign.nVal & 0x80000000u;
			result = tOffset.flVal;
			const float z = x / y;
			result -= (n1 + n2 * z * z) * z;
		}
	}
	else if (y > 0) {
		result = pi / 2;
	}
	else if (y < 0) {
		result = -pi / 2;
	}
	return result;
}



float hsum_ps_sse3(__m128 v) {
	__m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
	__m128 sums = _mm_add_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
	sums = _mm_add_ss(sums, shuf);
	return        _mm_cvtss_f32(sums);
}



scalar vdot(const __m128* a, const __m128* b, scalar bias, size_t n) {
	auto outResult = _mm_set_ps1(bias);
	for (size_t i = 0; i < n; i++)
		outResult = _mm_add_ps(outResult, _mm_mul_ps(a[i], b[i]));
	return hsum_ps_sse3(outResult);
	//outResult = _mm_hadd_ps(outResult, outResult);
	//outResult = _mm_hadd_ps(outResult, outResult);
	//return outResult.m128_f32[3];
}

scalar vdot4(const __m128* a, const __m128* b) {
	auto outResult = _mm_mul_ps(a[0], b[0]);
	return hsum_ps_sse3(outResult);
}

scalar vdot8(const __m128* a, const __m128* b) {
	auto outResult = _mm_add_ps(_mm_mul_ps(a[0], b[0]), _mm_mul_ps(a[1], b[1]));
	return hsum_ps_sse3(outResult);
}


// Linear Algebra

point::point(const scalar& x, const scalar& y):
	std::array< scalar, 2 >({ x,y })
{}

point operator+ (const point& a, const point& b) {
	return point(a[0] + b[0], a[1] + b[1]);
}

point operator- (const point& a, const point& b) {
	return point(a[0] - b[0], a[1] - b[1]);
}

point operator* (const point& a, const scalar& b) {
	return point(a[0] * b, a[1] * b);
}

point operator/ (const point& a, const scalar& b) {
	return point(a[0] / b, a[1] / b);
}

// dot product
scalar dot(const hvec2f& a, const hvec2f& b) {
	return a[0] * b[0] + a[1] * b[1];
}

// determinant, or cross product (signed area)
scalar det(const hvec2f& a, const hvec2f& b) {
	return a[0] * b[1] - a[1] * b[0];
}

// construct from angle, uses LUT for reading both sin and cos values in one memory read
point unit(const scalar& angle) {
	return point(cos(angle), sin(angle));
}

hvec2f hunit(const scalar& angle) {
	return unit(angle);
}

//__declspec(noinline) // good way to see disassembly, to estimate compiler work
scalar point_to_segment_squared_distance_impl(const point& qp, const point& s0, const point& s1) {
	auto s0s1 = s1 - s0;
	auto hs0s1 = hvec2f(s0s1);
	auto hs0qp = hvec2f(qp - s0);
	auto len2 = dot(hs0s1, hs0s1);
	auto t = max(((scalar)0), min(len2, dot(hs0s1, hs0qp))) / len2; // t is a number in [0,1] describing the closest point on the line segment s, as a blend of endpoints
	auto cp = s0 + s0s1 * t; // cp is the position (actual coordinates) of the closest point on the segment s
	auto dv = cp - qp;
	return dot(dv, dv);
}

//__declspec(noinline) 
scalar get_distance_inverse_impl(const hvec2f& s0qp, const hvec2f& s0s1, const hvec2f& d) {
	auto dd = det(d, s0s1); //auto dd = d[0] * s0s1[1] - d[1] * s0s1[0];
	if (dd != 0) { // lines are not parallel
		auto s = det(d, s0qp) / dd; //auto s = (s0qp[1] * d[0] - s0qp[0] * d[1]) / dd;
		if (s >= 0 && s < 1) { // segment intersects ray (s >= 0 && s <= 1)
			auto r = det(s0s1, s0qp) / dd; // auto r = (s0qp[1] * s0s1[0] - s0qp[0] * s0s1[1]) / dd;
			if (r >= 0) // ray is not going in direction, opposing to unit vector
				return scalar(1) / (r + 1);
		}
	}
	return 0; // infinitely far, parallel
}

void init_fast_math() {
	//build_fastest_tanh_lut();
	//build_fastest_sincos_lut();
}