#include "physics_engine.h"

#include <iostream>

#include <smmintrin.h>

using namespace std;

namespace physics {

	segment segment_from_points(const point& s0, const point& s1) {
		auto s0s1(s1 - s0);
		// latter 2 points are not a points but a performance hints for dot/cross products
		return segment({ s0, s1, point( _mm_set_ss(s0s1[0]) ), point( _mm_set_ss(s0s1[1]) ) });
	}

	void vehicle::step(const engine& phyEngine) {
		if (!_isActive)
			return;
		auto dT = phyEngine.get_dt();
		auto dT2 = phyEngine.get_dt_squared();

		

		// angular rotation with acceleration. yep, i am total nerd :D
		scalar rotationFriction = ((scalar)(1.95)); // 1.0 = no friction; 0.0 = infinite friction
		//auto ang1 = ang + rotationFriction * (ang - ang0) + _vehicleControls[SteeringControl] * dT2;
		//ang0 = ang;
		//ang = ang1;
		auto stepVelocity = p - p0;
		//auto velFactor = dot(stepVelocity, stepVelocity);
		ang = ang + _vehicleControls[SteeringControl] * dT2 * 64; // *dot(stepVelocity, stepVelocity) * 1024;		
		while (ang >= pi)
			ang -= 2 * pi;
		while (ang < -pi)
			ang += 2 * pi;

		// motion with acceleration;
		auto mc = _vehicleControls[MotionControl] * dT2; // Acceleration vector
		auto acc1 = unit(ang);
		auto acc = acc1 * mc; // Acceleration vector
		//auto acc = mul(unit(ang), _vehicleControls[MotionControl] * dT2); // Acceleration vector
		scalar motionFriction = ((scalar)(0.85)); // Energy loss: 1.0 = no friction; 0.0 = infinite friction
		//scalar motionFriction = 1;
		
		auto p1 = 
		//( dot(stepVelocity, acc1) >= 0 ) ?
		p + stepVelocity * motionFriction + acc; // newPosition = currentPosition + (currentPosition - prevPosition)*motionFriction + a * dT^2
		//: p;


		// check for collision, if collides with wall then blow the vehicle
		if (phyEngine.check_collisions(p1, rSqr)) {
			_isActive = false; // respawn()
			return;
		}

		p0 = p; // save old position
		p = p1; // actuate new position
	}

/*	scalar vehicle::get_speed() const {
		if (_isActive) {
			auto dv = p - p0;
			return sqrt(dot(dv, dv));
		}
		else
			return 0;
	}*/

	engine::engine(const scalar & fixedDT)	{
		dT = fixedDT;
		dT2 = dT*dT;
	}

	scalar engine::get_dt() const {
		return dT;
	}

	scalar engine::get_dt_squared() const {
		return dT2;
	}

	// convenient mapping to optimized collision detector
	bool circle_segment_collides(const point & qp, const segment& s, const __m128& rSqr) {
		auto s0qp = qp - s[0];
		return _mm_circle_segment_collides(_mm_set_ss(s0qp[0]), _mm_set_ss(s0qp[1]), s[2]._vS, s[3]._vS, rSqr).m128_u32[0] == 0xffffffff;
	}


	bool engine::check_collisions(const point& qp, const scalar& rSqr) const {
		auto rSqr_ss = _mm_set_ss(rSqr);
		for (auto& i : _vWalls)
			if (circle_segment_collides(qp, i, rSqr_ss)) // <= rSqr
				return true;
		return false;
	}

	// convenient mapping to optimized proximity
	scalar ray_segment_distance_inverse(const point& qp, const segment& s, const __m128& cosAngle, const __m128& sinAngle) {
		auto s0qp = qp - s[0];
		return _mm_cvtss_f32( _mm_ray_segment_distance_inverse(_mm_set_ss(s0qp[0]), _mm_set_ss(s0qp[1]), s[2]._vS, s[3]._vS, cosAngle, sinAngle) );
	}

	scalar engine::ray_trace(const point& qp, const scalar& curAngle) const {
		auto cosAngle = _mm_set_ss(fastest_cos(curAngle));
		auto sinAngle = _mm_set_ss(fastest_sin(curAngle));
		scalar maxDistanceInverse = 0;
		/*for (auto& i : _vWalls) {
			auto md = ray_segment_distance_inverse(qp, i, cosAngle, sinAngle);
			if (abs(md) > abs(maxDistanceInverse))
				maxDistanceInverse = md;
		}*/
		for (auto& i : _vWalls)
			maxDistanceInverse = max(maxDistanceInverse, ray_segment_distance_inverse(qp, i, cosAngle, sinAngle));
		return maxDistanceInverse;
	}

	vector< scalar > engine::build_sensor_angles(size_t sensorWidth) {
		scalar startAngle = piDiv2 * (((scalar)1) - ((scalar)sensorWidth)) / sensorWidth;
		vector< scalar > sensorAngles;
		sensorAngles.reserve(sensorWidth);
		for (size_t i = 0; i < sensorWidth; i++)
			sensorAngles.emplace_back( startAngle + (pi * i) / sensorWidth );
		return sensorAngles;
	}

	void engine::calculate_sensor_proximity_field(const vehicle& v, scalar * sensorOutput, const vector< scalar >& sensorAngles) const {
		for (size_t i = 0; i < sensorAngles.size(); i++)
			sensorOutput[i] = ray_trace(v.p, v.ang + sensorAngles[i]);
	}

}
