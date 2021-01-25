_MM_ALIGN16 static const __m128 _mm_0_ss(_mm_set_ss(0));
_MM_ALIGN16 static const __m128 _mm_1_ss(_mm_set_ss(1));

__m128 _mm_sqr_ss(const __m128& x) {
	return _mm_mul_ss(x, x);
}

// 2mul, 1add
__m128 _mm_dot_ss(const __m128& a0, const __m128& a1, const __m128& b0, const __m128& b1) {
	return _mm_add_ss(_mm_mul_ss(a0, b0), _mm_mul_ss(a1, b1));
}

// 2mul, 1sub
__m128 _mm_det_ss(const __m128& a0, const __m128& a1, const __m128& b0, const __m128& b1) {
	return _mm_sub_ss(_mm_mul_ss(a0, b1), _mm_mul_ss(a1, b0));
}

// collision detector:  3dot, 2mul, 1div, 2sub, 4set, 3comiss
bool _mm_circle_segment_collides(const __m128& s0qp0, const __m128& s0qp1, const __m128& s0s10, const __m128& s0s11, const __m128& rSqr) {
	auto a = _mm_dot_ss(s0s10, s0s11, s0s10, s0s11); //dot(s0s1, s0s1);
	//if(_mm_comineq_ss( a, _mm_set_ss(0) ) ) //if( a != 0 ) // if you haven't zero-length segments omit this
	{
		auto b = _mm_dot_ss(s0s10, s0s11, s0qp0, s0qp1);// dot(s0s1, s0qp);
		auto t = _mm_div_ss(b, a); //b / a; // length of projection of s0qp onto s0s1
		auto tge0 = _mm_comige_ss(t, _mm_0_ss);
		auto tle1 = _mm_comile_ss(t, _mm_1_ss);
		if(tge0 && tle1)// ((t >= 0) && (t <= 1)) 
		{
			auto c = _mm_dot_ss(s0qp0, s0qp1, s0qp0, s0qp1); //dot(s0qp, s0qp);
			auto r2 = _mm_sub_ss(c, _mm_mul_ss(a, _mm_sqr_ss(t))); //r^2 = c - a * t^2;
			auto dist2 = _mm_sub_ss(r2, rSqr); // dist^2 = r2 - rSqr
			return _mm_comile_ss(dist2, _mm_0_ss); // dist2 <= 0;
		}
	}
	return false;
}

__m128 _mm_pos_distance_inverse(const __m128& r) {
	if (_mm_comige_ss(r, _mm_0_ss))
		return _mm_div_ss(_mm_1_ss, _mm_add_ss(r, _mm_1_ss));
	else
		return _mm_0_ss;
}

// neural network proximity sensor input: 3det, 3div, 1add, 6set, 4comiss
__m128 _mm_ray_segment_distance_inverse(const __m128& s0qp0, const __m128& s0qp1, const __m128& s0s10, const __m128& s0s11, const __m128& d0, const __m128& d1) {

	auto dd = _mm_det_ss(d0, d1, s0s10, s0s11); //auto dd = d[0] * s0s1[1] - d[1] * s0s1[0];
	//if (_mm_comineq_ss(dd, _mm_0_ss)) // lines are not parallel, would try to omit
	{
		auto s = _mm_div_ss(_mm_det_ss(d0, d1, s0qp0, s0qp1), dd ); //auto s = (s0qp[1] * d[0] - s0qp[0] * d[1]) / dd;
		auto sge0 = _mm_comige_ss(s, _mm_0_ss);
		auto sle1 = _mm_comile_ss(s, _mm_1_ss);
		if (sge0 && sle1) { // segment intersects ray (s >= 0 && s <= 1)
			auto r = _mm_div_ss( _mm_det_ss(s0s10, s0s11, s0qp0, s0qp1), dd ); // auto r = (s0qp[1] * s0s1[0] - s0qp[0] * s0s1[1]) / dd;
			return _mm_pos_distance_inverse(r);
		}
	}
	return _mm_0_ss; // infinitely far, parallel, not intersecting
}