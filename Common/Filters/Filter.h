#pragma once

#include "Items.h"


using namespace TRT::Types;


namespace TRT
{
	/**
	 * @brief This class is an abstract class which all pipeline filters inherit from
	 * 
	 */
	class Filter
	{
	public :
		Filter();
		~Filter();

		void setConfig(nlohmann::json config);
		void start();
		void stop();
		void waitUntilFinished();

	protected:
		virtual void run() = 0;

	protected:
		std::thread _Thread;
		std::atomic_bool _IsRunning;
		nlohmann::json _Config;
	};
}
