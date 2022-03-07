#pragma once

#include "Types.h"

namespace TRT
{
	/**
	 * @brief A thread-safe queue class
	 * 
	 * @tparam T whatever you want to carry with the queue
	 * @param bs buffer size
	 */
	template <typename T>
	class ConcurrentQueue 
	{
	public:
		T pop() 
		{
			std::unique_lock<std::mutex> mlock(mutex_);
			while (queue_.empty()) 
			{
				cond_.wait(mlock);
			}
			auto val = queue_.front();
			queue_.pop();
			mlock.unlock();
			cond_.notify_one();
			return val;
		}

		void pop(T& item)
		{
			std::unique_lock<std::mutex> mlock(mutex_);
			while (queue_.empty()) 
			{
				cond_.wait(mlock);
			}
			item = queue_.front();
			queue_.pop();
			mlock.unlock();
			cond_.notify_one();
		}

		void push(const T& item) 
		{
			
			std::unique_lock<std::mutex> mlock(mutex_);
			while (queue_.size() >= BUFFER_SIZE) //config["vQueueBufferSize"])
			{
				cond_.wait(mlock);
			}
			queue_.push(item);
			mlock.unlock();
			cond_.notify_one();
		}

		int size()
		{
			return queue_.size();
		}

		ConcurrentQueue()
		{
			BUFFER_SIZE = max_unsigned_int;
		}
		ConcurrentQueue(unsigned int bs)
		{
			BUFFER_SIZE = bs;
		}
		ConcurrentQueue(const ConcurrentQueue&) = delete;            // disable copying
		ConcurrentQueue& operator=(const ConcurrentQueue&) = delete; // disable assignment


	private:
		std::queue<T> queue_;
		std::mutex mutex_;
		std::condition_variable cond_;
		unsigned int BUFFER_SIZE;
	};
}

/**
 * @brief simplified shared pointer declaration for queues
 * 
 * @tparam T 
 */
template <class T>
using sharedQ = std::shared_ptr<TRT::ConcurrentQueue<T>>;

/**
 * @brief template function that simplifies creating queues
 * 
 * @tparam T 
 * @tparam Args 
 * @param args 
 * @return std::shared_ptr<TRT::ConcurrentQueue<T>> 
 */
template <typename T, typename... Args>
inline sharedQ<T> make_Q(Args && ... args)
{
    return std::make_shared<TRT::ConcurrentQueue<T>>((args)...);
}