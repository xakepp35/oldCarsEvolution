#pragma once

#include <cstdint>	// all standart integer types
#include <cmath>
#include <array>	// std::array, fixed length container
#include <vector>	// std::vector, storage container and memory management

#include "fast_math.h"


namespace physics {

	// line segment, defined by 2 points
	typedef std::array< point, 4 > segment;

	segment segment_from_points(const point& p0, const point& p1);

	class engine;

	// physics shape: circle; controlled by _vehicleControls: MotionControl, and SteeringControl;


	class vehicle 
	{
	public:
		point p; // current pos
		point p0; // old pos;
		scalar ang; // current turn angle (would be quaternion in 3D space)
		scalar ang0; // previous turn angle, would like to use accelerated rotation as well :D

		scalar rSqr; // bounding sphere radius for collision detector

		std::array< scalar, 2 > _vehicleControls;

		enum control_type: size_t {
			SteeringControl, // positive means CCW, negative means CW?
			MotionControl,  // positive means accelerate, negative means brakes
			ControlsCount
		};

		bool _isActive; // if false vehicle will not move at all

		// verlet rotation and motion integrator
		void step(const engine& phyEngine);

		//scalar get_speed() const;

	};

	class engine {
	public:

		// timedelta for physics engine better to be fixed..
		engine(const scalar& fixedDT);

		scalar get_dt() const;
		scalar get_dt_squared() const;
		bool check_collisions(const point& p, const scalar& rSqr) const;

		// returns (distance+1) inverse, computationally heavy
		scalar ray_trace(const point& qp, const scalar& curAngle) const;

		// renders proximity field, proportional to inverse squared distance for any automatically choosen angular resulution, of pi/sensorWidth between each ray
		std::vector< scalar > build_sensor_angles(size_t sensorWidth);
		void calculate_sensor_proximity_field(const vehicle& v, scalar* sensorOutput, const std::vector< scalar >& sensorAngles) const;
		std::vector< segment > _vWalls; // race track walls are represented as segments

	protected:

		scalar dT; // timedelta
		scalar dT2; // dt^2

	};

	


}