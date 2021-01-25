#include "cars_simulation.h"
#include "sheduler_utils.h"

#include "../glfw-3.2.1/include/GLFW/glfw3.h"

#pragma comment(lib, "OpenGL32.lib")

#include <random>
#include <iostream>
#include <sstream>


static std::array< bool, GLFW_KEY_LAST > btnReleased = { false };
static GLFWwindow* window = nullptr;

bool im_key(int glfwKey) {
	if ((glfwGetKey(window, glfwKey) == GLFW_PRESS)) {
		if (btnReleased[glfwKey]) {
			btnReleased[glfwKey] = false;
			return true;
		}
	}
	else
		btnReleased[glfwKey] = true;
	return false;
}

int main() {
	std::cout <<
	"Natural Selection of Neural Network(Perceptrone) weights demo\n"
	"\n"
	"Date : Feb 2018\n"
	"Author : xakepp35@gmail.com\n"
	"License : FreeBSD(ISC) for non - commercial(personal, educational) use.\n"
	"\n"
	;

	//GLFWwindow* window;
	if (!glfwInit())
		return -1;

	init_fast_math();

	//glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); //asking for opengl version 3
	//glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); //and then .3, so 3.3
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //so we're using ONLY the core profile functions
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
	static const char winTitle[] = "Hello Neuro!";
	window = glfwCreateWindow(800, 800, winTitle, NULL, NULL);
	//window = glfwCreateWindow(1920, 1080, winTitle, glfwGetPrimaryMonitor(), NULL);

	
	if (!window) {
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

// setup simulation 
	simulation::engine e(true, 10, 2048, 1.0f / 15);

	xoroshiro128plus rGen(123);
	e.generate_track(rGen);
	//e.generate_rectangle_track();

	e.initial_agent_spawn();
	//e._startingPoint = 
	//int i = 0;
	////char c[2];
	//c[1] = 0;
	//c[0] = 60;

	std::cout <<
		"controls:\n"
		"R: enable/disable rendering (for fastest possible generation)\n"
		"V: enable/disable vSync (limits to 60 fps)\n"
		"F: renders view fields of agents\n"
		"+-: modifies max step count for each generation\n"
		"*/: modifies weights adjustment during mutation, less=finer slower evolution, greater=increased breakthrough possibility\n"
		"\n"
		;

	bool enableRender = false;// true;
	bool enableFOVrender = false;


	int enableVSync = 0;
	glfwSwapInterval(enableVSync);
	
	//SetPriorityClass(GetCurrentProcess())
	
	static const double lowpassAlphaRV = 1.0 / 64;
	static const double lowpass1AlphaRV = 1.0 - lowpassAlphaRV;
	static const double lowpassAlphaR = 1.0 / 4096;
	static const double lowpass1AlphaR = 1.0 - lowpassAlphaR;
	static const double lowpassAlphaF = 1.0 / 65536;
	static const double lowpass1AlphaF = 1.0 - lowpassAlphaF;
	double lowpassFPS = 0;

	////////////////////////////////////////////////////////////////
	// Starting GameLoop
	////////////////////////////////////////////////////////////////
	std::cout << "Starting GameLoop..\n";
	auto hpc0 = sheduler::now()-sheduler::freq();
	while (!glfwWindowShouldClose(window))
	{
		////////////////////////////////////////////////////////////////
		// Window title update
		////////////////////////////////////////////////////////////////
		{
			auto hpc1 = sheduler::now();
			auto frameTime = sheduler::map_time< double >(hpc1 - hpc0);
			hpc0 = hpc1;
			auto measuredFPS = 1.0 / frameTime;
			auto lowpass1Alpha = lowpass1AlphaF;
			auto lowpassAlpha = lowpassAlphaF;
			
			if (enableRender) { // renderrer has lower fps
				if (enableVSync != 0) {
					lowpass1Alpha = lowpass1AlphaRV;
					lowpassAlpha = lowpassAlphaRV;
				}
				else {
					lowpass1Alpha = lowpass1AlphaR;
					lowpassAlpha = lowpassAlphaR;
				}
			}
			lowpassFPS *= lowpass1Alpha;
			lowpassFPS += measuredFPS*lowpassAlpha;
	
			std::ostringstream s;
			if (enableRender && enableVSync != 0) {
				s << "Hello Neuro! Gen#" << e._nGeneration << "; Step#" << e._generationStep << "; FPS=" << lowpassFPS;
				glfwSetWindowTitle(window, s.str().c_str());
			}
			else
				if (e._generationStep == 0) {// update window title rarely
					s << "Hello Neuro! Gen#" << e._nGeneration << "; FPS=" << lowpassFPS;
					glfwSetWindowTitle(window, s.str().c_str());
				}
		}

		////////////////////////////////////////////////////////////////
		// Renderer code
		////////////////////////////////////////////////////////////////
		if (enableRender) {
			
			glClearColor(0, 0, 0, 0);
			glClear(GL_COLOR_BUFFER_BIT); // | GL_DEPTH_BUFFER_BIT);

			////////////////////////////////////////////////////////////////
			// render walls
			////////////////////////////////////////////////////////////////
			glLineWidth(1.0);
			glColor3f(1.0, 1.0, 0.0);
			glBegin(GL_LINES);
			for (auto&i : e._mPhysics._vWalls) {
				glVertex2f(i[0][0], i[0][1]);
				glVertex2f(i[1][0], i[1][1]);
			}
			glEnd();

			////////////////////////////////////////////////////////////////
			// render cars
			////////////////////////////////////////////////////////////////
			glBegin(GL_TRIANGLES);
			for (auto&i : e._vAgent) {
				auto& car = *static_cast<simulation::car*>(i.get());
				scalar carR = sqrt(car._mPhysics.rSqr);
				//glTranslatef(-car._mPhysics.p[0], -car._mPhysics.p[1], 0);
				auto pos0 = car._mPhysics.p + unit(car._mPhysics.ang) * carR;
				auto pos1 = car._mPhysics.p + unit(car._mPhysics.ang + pi * 2 / 3) * carR;
				auto pos2 = car._mPhysics.p + unit(car._mPhysics.ang - pi * 2 / 3) * carR;
				//auto pos0 = physics::mul(physics::unit(car._mPhysics.ang), carR);
				//auto pos1 = physics::mul(physics::unit(car._mPhysics.ang + physics::pi * 2 / 3), carR);
				//auto pos2 = physics::mul(physics::unit(car._mPhysics.ang - physics::pi * 2 / 3), carR);
				if (car._mPhysics._isActive)
					glColor3f(0.0, 1.0, 0.0);
				else
					glColor3f(1.0, 0.0, 0.0);
				glVertex2f(pos0[0], pos0[1]);
				glColor3f(0.0, 0.5, 1.0);
				glVertex2f(pos1[0], pos1[1]);
				glVertex2f(pos2[0], pos2[1]);
				//glLoadIdentity();
			}
			glEnd();

			////////////////////////////////////////////////////////////////
			// render view fields
			////////////////////////////////////////////////////////////////
			if (enableFOVrender) {
				glBegin(GL_LINES);
				glColor3f(1.0, 1.0, 1.0);
				for (auto&i : e._vAgent) {
					auto& car = *static_cast<simulation::car*>(i.get());
					auto& mem = car._mNeural._vMemory[1 - car._mNeural._tapeIndex];
					auto pos0 = car._mPhysics.p;
					size_t sensorWidth = car._mNeural.input_size();
					float startAngle = car._mPhysics.ang + piDiv2 * (((scalar)1) - ((scalar)sensorWidth)) / sensorWidth;
					for (size_t j = 0; j < sensorWidth; j++) {
						auto dist = ((scalar*)mem.data())[car._mNeural._vcell.size() + j];
						if (dist > 0) {
							//auto t = sqrt(1 / dist - 1);
							auto t = 1 / dist - 1;
							auto currentAngle = startAngle + (pi * j) / sensorWidth;
							auto unitDir = unit(currentAngle);
							auto pos1 = car._mPhysics.p + unitDir * t;
							glVertex2f(pos0[0], pos0[1]);
							glVertex2f(pos1[0], pos1[1]);
						}
					}
				}
				glEnd();
			}

			// Swap front and back buffers
			glfwSwapBuffers(window);
		}

		////////////////////////////////////////////////////////////////
		// "Immediate mode" keyboard & window events handling
		////////////////////////////////////////////////////////////////
		glfwPollEvents();

		if (im_key(GLFW_KEY_R)) {
			enableRender = !enableRender;
			std::cout << "\t* enableRender = " << (int)(enableRender) << "\n";
		}
		if (im_key(GLFW_KEY_V)) {
			enableVSync = 1 - enableVSync;
			glfwSwapInterval(enableVSync);
			std::cout << "\t* vSync = " << enableVSync << "\n";
		}
		if (im_key(GLFW_KEY_F)) {
			enableFOVrender = !enableFOVrender;
			std::cout << "\t* fovRender = " << (int)(enableFOVrender) << "\n";
		}
		if (im_key( GLFW_KEY_KP_ADD)) {
			e._generationStepMax *= 2;
			std::cout << "\t* _generationSteps = " << e._generationStepMax << "\n";
		}
		if( im_key(GLFW_KEY_KP_SUBTRACT ) ) {
			e._generationStepMax /= 2;
			std::cout << "\t* _generationSteps = " << e._generationStepMax << "\n";
		}
		if (im_key(GLFW_KEY_KP_MULTIPLY)) {
			e._weightDelta *= 2;
			std::cout << "\t* _weightDelta = " << e._weightDelta << "\n";
		}
		if (im_key(GLFW_KEY_KP_DIVIDE)) {
			e._weightDelta /= 2;
			std::cout << "\t* _weightDelta = " << e._weightDelta << "\n";
		}

		////////////////////////////////////////////////////////////////
		// Phycics calculation
		////////////////////////////////////////////////////////////////


		e.step();
	}

	////////////////////////////////////////////////////////////////
	// GameLoop is finished
	////////////////////////////////////////////////////////////////
	std::cout << "\t* GameLoop is finished\n";

	glfwTerminate();
	return 0;
}
