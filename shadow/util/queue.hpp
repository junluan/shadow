#ifndef SHADOW_UTIL_QUEUE_HPP
#define SHADOW_UTIL_QUEUE_HPP

#include <atomic>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <queue>

namespace Shadow {

template <typename T>
class Queue {
 public:
  explicit Queue(unsigned int max_size = 0) : interrupt_(false) {
    if (max_size > 0) {
      max_size_ = max_size;
    }
  }

  void push(T item) {
    std::unique_lock<std::mutex> lock{lock_};
    cond_full_.wait(lock,
                    [&]() { return queue_.size() < max_size_ || interrupt_; });
    queue_.push(std::move(item));
    lock.unlock();
    cond_.notify_one();
  }

  T pop() {
    static auto int_return = T{};
    std::unique_lock<std::mutex> lock{lock_};
    cond_.wait(lock, [&]() { return !queue_.empty() || interrupt_; });
    if (interrupt_) {
      return std::move(int_return);
    }
    auto item = std::move(queue_.front());
    queue_.pop();
    cond_full_.notify_one();
    return item;
  }

  const T& peek() {
    static auto int_return = T{};
    std::unique_lock<std::mutex> lock{lock_};
    cond_.wait(lock, [&]() { return !queue_.empty() || interrupt_; });
    if (interrupt_) {
      return std::move(int_return);
    }
    return queue_.front();
  }

  void cancel_pops() {
    std::unique_lock<std::mutex> lock{lock_};
    interrupt_ = true;
    cond_.notify_all();
    cond_full_.notify_all();
  }

  void clear() {
    std::unique_lock<std::mutex> lock{lock_};
    std::queue<T> empty;
    std::swap(queue_, empty);
  }

  size_t size() const { return queue_.size(); }

  bool empty() const { return queue_.empty(); }

 private:
  std::queue<T> queue_;
  std::mutex lock_;
  std::condition_variable cond_, cond_full_;
  std::atomic<bool> interrupt_;
  unsigned int max_size_ = std::numeric_limits<unsigned int>::max();
};

}  // namespace Shadow

#endif  // SHADOW_UTIL_QUEUE_HPP
