#include "full_sse_fast_math.h"

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

/*
static float faster_tanh(float x) {
return x * (2.45550750702956f + 2.45550750702956f * abs(x) + (0.893229853513558f + 0.821226666969744f * abs(x)) * x*x) / (2.44506634652299f + (2.44506634652299f + x*x) * abs(x + 0.814642734961073f * x * abs(x)));
}
*/



// this is much better than previous, and does have high accuracy from -3 to 3, also its derivatives at -3 and at 3 are equal zero
scalar faster_tanh(const scalar& x) {
	if ((x < 3) && (x > -3))
		return x * (x * x + 27) / (x * x * 9 + 27);
	else
		return sgn_sigmf(x);
}


// but best way to compute tanh.. is not to compute it at all? NOPE, you could test :)
static const size_t		tanhLutSize = 1 << 20; // 4 bytes * 1M items = 4 megabyte RAM will be used for tanh[] lookup table
static std::array< scalar, tanhLutSize > fastTanhs;
static const scalar tanhRenormCoeff = tanhLutSize / (2*8); // from -8.0 to 8.0 we use table, otherwise we use constants. This covers almost all floating point precision (20 bit precision guranteed)


static scalar dextin(size_t i) {
	return static_cast<scalar>(((intptr_t)(i)) - ((intptr_t)(tanhLutSize / 2))) / tanhRenormCoeff;
}

void build_fastest_tanh_lut() {
	for (size_t i = 0; i < tanhLutSize; i++)
		fastTanhs[i] = tanh(dextin(i));
}

// convert angle in radians to index in LUT table
static intptr_t tindex(const scalar& x) {
	return static_cast<intptr_t>(x * tanhRenormCoeff) + tanhLutSize / 2;
}


scalar fastest_tanh(const scalar& x) {
	return faster_tanh(x); 
	/*auto tIndex = tindex(x);
	if (tIndex < 0)
		return -1;
	if (tIndex >= tanhLutSize)
		return 1;
	return fastTanhs[tIndex];*/
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


point::point(const __m128& vS) :
	_vS(vS)
{}

point::point(const scalar & x, const scalar & y) :
	_vS(_mm_set_ps(0, 0, y, x))
{}

const scalar& point::operator[](const size_t i) const {
	return _vS.m128_f32[i];
}

scalar & point::operator[](const size_t i) {
	return _vS.m128_f32[i];
}


point operator+ (const point& a, const point& b) {
	return point(_mm_add_ps(a._vS, b._vS));
}

point operator- (const point& a, const point& b) {
	return point(_mm_sub_ps(a._vS, b._vS));
}

point operator* (const point& a, const scalar& b) {
	auto mulCoeff(_mm_set_ps1(b));
	return point(_mm_mul_ps(a._vS, mulCoeff));
}

point operator/ (const point& a, const scalar& b) {
	auto divCoeff(_mm_set_ps1(b));
	return point(_mm_div_ps(a._vS, divCoeff));
}


hvec2f::hvec2f(const point& p):
	hvec2f(p._vS, _mm_set_ps1(p[1]))
{
	//_mm_shuffle_ps(p._vS)
	//p._vS
}

hvec2f::hvec2f(const __m128 & x, const __m128 & y):
	_vS({ x, y })
{}

hvec2f::hvec2f(const scalar & x, const scalar & y) :
	//_vS({ _mm_set_ps1(x),_mm_set_ps1(y) })
	_vS({ _mm_set_ps(0,0,0,x),_mm_set_ps(0,0,0,y) })
{}

scalar hvec2f::operator[](const size_t i) const {
	return _mm_cvtss_f32(_vS[i]);
}

scalar & hvec2f::operator[](const size_t i) {
	return _vS[i].m128_f32[0];
}


scalar dot(const hvec2f& a, const hvec2f& b) {
	auto outResult = _mm_add_ps(_mm_mul_ps(a._vS[0], b._vS[0]), _mm_mul_ps(a._vS[1], b._vS[1]));
	//return outResult.m128_f32[3];
	return _mm_cvtss_f32(outResult);
}

scalar det(const hvec2f& a, const hvec2f& b) {
	auto outResult = _mm_sub_ps(_mm_mul_ps(a._vS[0], b._vS[1]), _mm_mul_ps(a._vS[1], b._vS[0]));
	//return outResult.m128_f32[3];
	return _mm_cvtss_f32(outResult);
}

/*
scalar dot(const point& a, const point& b) {
	//return a[0] * b[0] + a[1] * b[1];
	auto mulRes(_mm_mul_ps(a._vS, b._vS));
	return _mm_hadd_ps(mulRes, mulRes).m128_f32[3];
	//auto k = _mm_dp_ps(a._vS, b._vS, 0xff);
	//return k.m128_f32[4];
}

scalar det(const point& a, const point& b) {
	//return a[0] * b[1] - b[0] * a[1];
	auto bShuf = _mm_shuffle_ps(b._vS, b._vS, _MM_SHUFFLE(2, 3, 2, 3));
	auto mulRes = _mm_mul_ps(a._vS, bShuf);
	return _mm_hsub_ps(mulRes, mulRes).m128_f32[3];
}
*/

/*
inline __m128 CrossProduct(__m128 a, __m128 b)
{
return _mm_sub_ps(
_mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2))),
_mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1)))
);
}
*/

/*

point operator+ (const point& a, const point& b) {
return point({ a[0] + b[0], a[1] + b[1] });
}

point operator- (const point& a, const point& b) {
return point({ a[0] - b[0], a[1] - b[1] });
}

point operator* (const point& a, const scalar& b) {
return point({ a[0] * b, a[1] * b });
}

point operator/ (const point& a, const scalar& b) {
return point({ a[0] / b, a[1] / b });
}

scalar dot(const point& a, const point& b) {
return a[0] * b[0] + a[1] * b[1];
}

scalar det(const point& a, const point& b) {
return a[0] * b[1] - b[0] * a[1];
}
*/

/*
__m128 faster_abs(__m128 m) {
	__m128 sign = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
	return _mm_andnot_ps(sign, m);
}


__m128 faster_sin(__m128 m_x) {
	static const float B = 4.f / pi;
	static const float C = -4.f / (pi * pi);
	static const float P = 0.225f;
	//float y = B * x + C * x * abs(x);
	//y = P * (y * abs(y) - y) + y;
	static const __m128 m_pi = _mm_set1_ps(pi);
	static const __m128 m_mpi = _mm_set1_ps(-pi);
	static const __m128 m_2pi = _mm_set1_ps(pi * 2);
	static const __m128 m_B = _mm_set1_ps(B);
	static const __m128 m_C = _mm_set1_ps(C);
	static const __m128 m_P = _mm_set1_ps(P);
	__m128 m1 = _mm_cmpnlt_ps(m_x, m_pi);
	m1 = _mm_and_ps(m1, m_2pi);
	m_x = _mm_sub_ps(m_x, m1);
	m1 = _mm_cmpngt_ps(m_x, m_mpi);
	m1 = _mm_and_ps(m1, m_2pi);
	m_x = _mm_add_ps(m_x, m1);
	__m128 m_abs = faster_abs(m_x);
	m1 = _mm_mul_ps(m_abs, m_C);
	m1 = _mm_add_ps(m1, m_B);
	__m128 m_y = _mm_mul_ps(m1, m_x);
	m_abs = faster_abs(m_y);
	m1 = _mm_mul_ps(m_abs, m_y);
	m1 = _mm_sub_ps(m1, m_y);
	m1 = _mm_mul_ps(m1, m_P);
	m_y = _mm_add_ps(m1, m_y);
	return m_y;
}


scalar faster_fsin(float x) {
	return _mm_cvtss_f32(faster_sin(_mm_set_ps1(x)));
}

scalar faster_fcos(float x) {
	return _mm_cvtss_f32( faster_sin(_mm_set_ps1(x + pi / 2)) );
}


// fastest shit ever
__m128 faster_sincos(const scalar& x) {
	return faster_sin(_mm_set_ps(x + pi/2, x, 0, 0));
}
*/

// sincos lut
static const size_t		sincosLutSize = 1 << 20; // 16 bytes * 1M items = 16 megabyte RAM will be used for sincos lookup table
static std::array< point, sincosLutSize > fastUnits;
static const scalar sincosRenormCoeff = sincosLutSize / (2 * pi); // 20 bit mantissa precision


static scalar dexsin(size_t i) {
	return static_cast<scalar>(i) / sincosRenormCoeff;
}

void build_fastest_sincos_lut() {
	for (size_t i = 0; i < sincosLutSize; i++)
		fastUnits[i] = point(cos(dexsin(i)), sin(dexsin(i)));
}

// convert angle in radians to index in LUT table
static size_t sindex(const scalar& x) {
	//return static_cast< size_t > (fastLutSize * 128 + x * fastLutSize / (2*pi)) & 0xffff;
	return static_cast<intptr_t>(x * sincosRenormCoeff) & 0xfffff; // i MUST exploit this perf hack! // int % fastLutSize + fastLutSize % fastLutSize;
}

scalar fast_sin(const scalar& x) {
	return fastUnits[sindex(x)][1];
}

scalar fast_cos(const scalar& x) {
	return fastUnits[sindex(x)][0];
}

//extern void sincos_ps(__m128 x, __m128 *s, __m128 *c);

point unit(const scalar& angle) {
	return point( cos(angle), sin(angle) );
	//return point( faster_sincos(angle) );
	//return point(faster_fcos(angle), faster_fsin(angle));
	//return fastUnits[sindex(angle)];
	//point p;
	//sincos_ps(_mm_set_ps1(angle), &p._vS[0], &p._vS[0]);
	//return p;
}

hvec2f hunit(const scalar& angle) {
	return hvec2f(cos(angle), sin(angle));
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
	build_fastest_tanh_lut();
	build_fastest_sincos_lut();
}