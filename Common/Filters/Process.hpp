#pragma once

#include "Filter.h"
#include "Core.h"

using namespace TRT::Types;

namespace TRT
{
	/**
	 * @brief A type of Filter class that carries out a certain process
	 * on an input of a certain data type and returns an output of the same type.
	 * 
	 * @tparam T A class or object type that is processed in filter
	 */
	template <typename Ti, typename To>
	class Process : public Filter 
	{
	public:
		Process() {}
		~Process() {}

		void setInputQueue(sharedQ<Ti> &q)
		{
			_InputQueue = q;
		}

		sharedQ<Ti> getInputQueue()
		{
            return _InputQueue;
        }

		void setOutputQueue(sharedQ<To> &q)
		{
            _OutputQueue = q;
        }
		
		sharedQ<To> getOutputQueue()
		{
            return _OutputQueue
        }

		void setEngine(shared<Engine> engine)
		{
            _Engine = engine;
        }

	protected:
		sharedQ<Ti> _InputQueue;
		sharedQ<To> _OutputQueue;

		int _BatchSize;
		To _InputBatch;
		shared<Engine> _Engine;
	};
}
