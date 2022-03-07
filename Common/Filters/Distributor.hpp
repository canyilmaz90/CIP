#pragma once

#include "Filter.h"


namespace TRT
{

	template <typename Ti=void, typename To1=void, typename To2=void>
	class Distributor : public Filter
	{
	public:
		Distributor();
		~Distributor();

		
		void setInputQueue(sharedQ<Ti> & inputQueue)
		{
			_InputQueue = inputQueue;
		}
		sharedQ<Ti> getInputQueue()
		{
			return _InputQueue;
		}
		void setOutputQueues(sharedV<sharedQ<To1>> & outputQueues)
		{
			_OutputQueues = outputQueues;
		}
		void setInfoOutputQueue(sharedQ<To2> & infoQueue)
		{
			_InfoOutputQueue = infoQueue;
		}

	public:
		sharedQ<Ti> _InputQueue;
		sharedV<sharedQ<To1>> _OutputQueues;
		sharedQ<To2> _InfoOutputQueue;

	
	protected:
		template <typename T>
		virtual void send(const sharedQ<T> &queue, T info)
		{
			queue->push(info);
		}
	};
}
