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

scalar fastest_sin(scalar x) {
	return sin(x);/*
	if (x >= pi)
		x -= 2 * pi;
	if (x < -pi)
		x += 2* pi;
	//compute sine
	auto x1 = 1.27323954f * x;
	//auto x2 = x1 * x / 3.14159265f;
	auto x2 = 0.405284735f * x * x;
	/*int sign = int(x<0);
	return x1 + x2 * sign;*/

	/*
	if (x < 0)
		return x1 + x2;
	else
		return x1 - x2;*/
}

scalar fastest_cos(scalar x) {
	return cos(x); // fastest_sin(x + pi / 2);
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
point::point(const __m128& vS) :
	_vS(vS)
{}

point::point(const scalar& x, const scalar& y) :
	_vS(_mm_set_ps(0, 0, y, x))
{}

const scalar& point::operator[](unsigned i) const {
	return _vS.m128_f32[i];
}

scalar& point::operator[](unsigned i) {
	return _vS.m128_f32[i];
}


point operator+ (const point& a, const point& b) {
	return point(_mm_add_ps( a._vS, b._vS ) );
}

point operator- (const point& a, const point& b) {
	return point(_mm_sub_ps(a._vS, b._vS));
}

point operator* (const point& a, const scalar& b) {
	return point(_mm_mul_ps(a._vS, _mm_set_ps1(b)));
}

point operator/ (const point& a, const scalar& b) {
	return point(_mm_div_ps(a._vS, _mm_set_ps1(b)));
}

// construct from angle, uses LUT for reading both sin and cos values in one memory read
point unit(const scalar& angle) {
	//return point(cos(angle), sin(angle));
	return point(fastest_cos(angle), fastest_sin(angle));
}

_MM_ALIGN16 static const __m128 _mm_0_ps(_mm_set_ps1(0));
_MM_ALIGN16 static const __m128 _mm_1_ps(_mm_set_ps1(1));

__m128 _mm_sqr_ps(const __m128& x) {
	return _mm_mul_ps(x, x);
}

__m128 _mm_abs_ps(const __m128& m) {
	static const auto signBit = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
	return _mm_andnot_ps(signBit, m);
}
/*
inline __m128 _mm_floor_ps(const __m128& a, const __m128& r) {
}
*/
//auto ss = _mm_set_ps1 { 0x80000000 ,0x80000000 ,0x80000000 ,0x80000000 };

inline __m128 _mm_fmod_ps(const __m128& a, const __m128& r) {
	auto c = _mm_div_ps(a, r);
	auto t = _mm_floor_ps(c); //auto t = _mm_cvtepi32_ps(_mm_cvttps_epi32(c)); // floor
	auto b = _mm_mul_ps(t, r);
	return _mm_sub_ps(a, b);
}
/*
__m128 _mm_fmodsgn_ps(const __m128& a, const __m128& r) {
	auto c = _mm_add_ps(_mm_div_ps(a, r), _mm_set_ps1(0.5f));
	auto t = _mm_floor_ps(c); //_mm_cvtepi32_ps(_mm_cvttps_epi32(c)); // round(x) = floor(x+0.5)
	auto b = _mm_mul_ps(t, r);
	auto y = _mm_sub_ps(a, b);
	auto mask = _mm_cmplt_ps(c, _mm_0_ps);
	return _mm_or_ps(
		_mm_and_ps(mask, _mm_add_ps(y, r)),
		_mm_andnot_ps(mask, y)
	);
}
*/




__m128 _mm_sin_chebushev_ps(const __m128& x) {
	//chebushev (6E-7 error): sin(x) = 0.99999660*x - 0.16664824*x ^ 3 + 0.00830629*x ^ 5 - 0.00018363*x ^ 7
	_MM_ALIGN16 static const __m128 mmChebSin1(_mm_set_ps1(0.99999660f));
	_MM_ALIGN16 static const __m128 mmChebSin3(_mm_set_ps1(-0.16664824f));
	_MM_ALIGN16 static const __m128 mmChebSin5(_mm_set_ps1(0.00830629f));
	_MM_ALIGN16 static const __m128 mmChebSin7(_mm_set_ps1(-0.00018363f));
	auto x2 = _mm_mul_ps(x, x); // x^2
	auto xp = x; // x to required power
	auto s = _mm_mul_ps(mmChebSin1, xp);
	s = _mm_add_ps(s, _mm_mul_ps(mmChebSin3, xp = _mm_mul_ps(xp, x2)));
	s = _mm_add_ps(s, _mm_mul_ps(mmChebSin5, xp = _mm_mul_ps(xp, x2)));
	s = _mm_add_ps(s, _mm_mul_ps(mmChebSin7, xp = _mm_mul_ps(xp, x2)));
	return s;
}

__m128 _mm_sin_ps(const __m128& x) {
	_MM_ALIGN16 static const __m128 psSinC1(_mm_set_ps1(scalar(4) / pi )); // 1.27323954f
	_MM_ALIGN16 static const __m128 psSinC2(_mm_set_ps1(scalar(4) / (pi*pi) )); //0.405284735 =-4/(pi^2)
	auto x1 = _mm_mul_ps(x, psSinC1);
	auto x2 = _mm_mul_ps(_mm_sqr_ps(x), psSinC2);
	auto mask = _mm_cmplt_ps(x, _mm_0_ps); // if (x < 0)
	return _mm_or_ps( 
		_mm_and_ps(mask, _mm_add_ps(x1, x2)), // return x1 + x2;
		_mm_andnot_ps(mask, _mm_sub_ps(x1, x2)) // else return x1 - x2;
	);
}



//x * 1.27323954 - x*x*0.405284735;


void _mm_sincos_ps(__m128&s, __m128&c, const __m128& x) {
	s = _mm_sin_ps(x);
	c = _mm_sqrt_ps(_mm_sub_ps(_mm_1_ps, _mm_sqr_ps(s)));
}



// 2mul, 1add
__m128 _mm_dot_ps(const __m128& a0, const __m128& a1, const __m128& b0, const __m128& b1) {
	return _mm_add_ps(_mm_mul_ps(a0, b0), _mm_mul_ps(a1, b1));
}

// 2mul, 1sub
__m128 _mm_det_ps(const __m128& a0, const __m128& a1, const __m128& b0, const __m128& b1) {
	return _mm_sub_ps(_mm_mul_ps(a0, b1), _mm_mul_ps(a1, b0));
}

// collision detector:  3dot, 2mul, 1div, 2sub, 4set, 3comiss
__m128 _mm_circle_segment_collides(const __m128& s0qp0, const __m128& s0qp1, const __m128& s0s10, const __m128& s0s11, const __m128& rSqr) {
	auto a = _mm_dot_ps(s0s10, s0s11, s0s10, s0s11); //dot(s0s1, s0s1);
	//auto amask = _mm_cmpneq_ps( a, _mm_0_ps ); //if( a != 0 ) // if you haven't zero-length segments omit this
	auto b = _mm_dot_ps(s0s10, s0s11, s0qp0, s0qp1);// dot(s0s1, s0qp);
	auto t = _mm_div_ps(b, a); //b / a; // length of projection of s0qp onto s0s1
	auto tmask = _mm_and_ps(_mm_cmpge_ps(t, _mm_0_ps), _mm_cmple_ps(t, _mm_1_ps)); // ((t >= 0) && (t <= 1)) 
	auto c = _mm_dot_ps(s0qp0, s0qp1, s0qp0, s0qp1); //dot(s0qp, s0qp);
	auto r2 = _mm_sub_ps(c, _mm_mul_ps(a, _mm_sqr_ps(t))); //r^2 = c - a * t^2;
	auto d2 = _mm_sub_ps(r2, rSqr); // d^2 = r^2 - rSqr
	auto dmask = _mm_cmple_ps(d2, _mm_0_ps); // dist2 <= 0;
	return _mm_and_ps(tmask, dmask);	
}

// neural network proximity sensor input: 3det, 3div, 1add, 7set, 4cmp
__m128 _mm_ray_segment_distance_inverse(const __m128& s0qp0, const __m128& s0qp1, const __m128& s0s10, const __m128& s0s11, const __m128& d0, const __m128& d1) {
	auto dd = _mm_det_ps(d0, d1, s0s10, s0s11); //auto dd = d[0] * s0s1[1] - d[1] * s0s1[0];
	auto dmask = _mm_cmpneq_ps(dd, _mm_0_ps); // dd != 0 // lines are not parallel
	auto s = _mm_div_ps(_mm_det_ps(d0, d1, s0qp0, s0qp1), dd); //auto s = (s0qp[1] * d[0] - s0qp[0] * d[1]) / dd;
	auto smask = _mm_and_ps(_mm_cmpge_ps(s, _mm_0_ps), _mm_cmple_ps(s, _mm_1_ps)); // segment intersects ray (s >= 0 && s <= 1)
	auto r = _mm_div_ps(_mm_det_ps(s0s10, s0s11, s0qp0, s0qp1), dd); // auto r = (s0qp[1] * s0s1[0] - s0qp[0] * s0s1[1]) / dd;
	auto rmask = _mm_cmpge_ps(r, _mm_0_ps); // r >= 0
	auto mask = _mm_and_ps(_mm_and_ps(dmask, smask), rmask);
	auto rinv = _mm_div_ps(_mm_1_ps, _mm_add_ps(r, _mm_1_ps)); // 1 / (r+1)
	return _mm_or_ps(	
		_mm_and_ps(mask, rinv), // 0 >= rinv >=1 
		_mm_andnot_ps(mask, _mm_0_ps)); // 0: parallel, not intersecting, infinitely far
}



void init_fast_math() {
	//build_fastest_tanh_lut();
	//build_fastest_sincos_lut();
}

/*

__m128 _mm_circle_segment_collision_determinant(const __m128& s0qp0, const __m128& s0qp1, const __m128& s0s10, const __m128& s0s11, const __m128& rSqr) {
auto a = _mm_dot_ss(s0s10, s0s11, s0s10, s0s11); // auto a = dot( s0s1, s0s1 );
auto b = _mm_mul_ss( _mm_set_ss(2), _mm_dot_ss(s0qp0, s0qp1, s0s10, s0s11)); // auto b = 2 * dot( s0qp, s0s1 );
auto c = _mm_sub_ss(_mm_dot_ss(s0qp0, s0qp1, s0qp0, s0qp1), rSqr); // auto c = dot(s0qp, s0qp) - rSqr
return _mm_sub_ss( _mm_mul_ss( b, b ), _mm_mul_ss(_mm_set_ss(4), _mm_mul_ss( a, c ) ) ); // auto d = b*b - 4*a*c
}

bool _mm_circle_segment_collision(const __m128& s0qp0, const __m128& s0qp1, const __m128& s0s10, const __m128& s0s11, const __m128& rSqr) {
static const __m128 _mm_2_ss(_mm_set_ss(2));
static const __m128 _mm_2n_ss(_mm_set_ss(-2));
auto a2 = _mm_mul_ss(_mm_2_ss, _mm_dot_ss(s0s10, s0s11, s0s10, s0s11)); // auto 2*a = dot( s0s1, s0s1 );
auto bn = _mm_mul_ss(_mm_2n_ss, _mm_dot_ss(s0qp0, s0qp1, s0s10, s0s11)); // auto -b = -2 * dot( s0qp, s0s1 );
auto c = _mm_sub_ss(_mm_dot_ss(s0qp0, s0qp1, s0qp0, s0qp1), rSqr); // auto c = dot(s0qp, s0qp) - rSqr
auto d = _mm_sub_ss(_mm_mul_ss(bn, bn), _mm_mul_ss(_mm_2_ss, _mm_mul_ss(a2, c))); // auto d = b*b - 4*a*c
if (_mm_comige_ss(d, _mm_set_ss(0))) {
auto ds = _mm_sqrt_ss(d); // sqrt(d)
auto t1 =_mm_cvtss_f32( _mm_div_ss(_mm_sub_ss(bn, ds), a2) );
auto t2 = _mm_cvtss_f32( _mm_div_ss(_mm_add_ss(bn, ds), a2) );
if( (t1 >= 0 && t1 <= 1) || (t2 >= 0 && t2 <= 1) )
return true;
}
return false;
}





__m128 _mm_sgn_distance_inverse(const __m128& r) {
auto sign = size_t(_mm_cvtss_si32(r) >> 31);
if(sign)
return _mm_div_ss(_mm_1_ss, _mm_sub_ss(r, _mm_1_ss));
else
return _mm_div_ss(_mm_1_ss, _mm_add_ss(r, _mm_1_ss));
}


__m128 _mm_distance_inverse_sqr(const __m128& r) {
auto rr = _mm_mul_ss(r, r);
return _mm_div_ss(_mm_1_ss, _mm_add_ss(rr, _mm_1_ss));
}


//Normalizes any number to an arbitrary range
//by assuming the range wraps around when going below min or above max
double normalise(const double value, const double start, const double end)
{
const double width = end - start;   //
const double offsetValue = value - start;   // value relative to 0

return (offsetValue - (floor(offsetValue / width) * width)) + start;
// + start to reset back to start of original range
}



*/