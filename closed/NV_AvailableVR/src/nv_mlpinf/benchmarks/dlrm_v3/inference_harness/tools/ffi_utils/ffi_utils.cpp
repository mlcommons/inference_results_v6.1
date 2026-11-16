/*
 * Multi-threaded QuerySamplesComplete pool for MLPerf loadgen
 * 
 * This module provides:
 * 1. QuerySamplesCompletePool - thread pool for calling QuerySamplesComplete
 * 
 * NOTE: This module must be imported AFTER mlperf_loadgen to avoid
 * "type already registered" errors for QuerySampleResponse.
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "query_sample_library.h"
#include "loadgen.h"

// NVTX for profiling (optional, enabled via HAVE_NVTX)
#ifdef HAVE_NVTX
#include <nvtx3/nvToolsExt.h>

// RAII wrapper for NVTX range
class NvtxRange {
public:
    NvtxRange(const char* name, uint32_t color = 0xFF00FF00) {
        nvtxEventAttributes_t attr = {0};
        attr.version = NVTX_VERSION;
        attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attr.colorType = NVTX_COLOR_ARGB;
        attr.color = color;
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = name;
        nvtxRangePushEx(&attr);
    }
    ~NvtxRange() { nvtxRangePop(); }
};

#define NVTX_RANGE(name) NvtxRange _nvtx_range_##__LINE__(name)
#define NVTX_RANGE_COLOR(name, color) NvtxRange _nvtx_range_##__LINE__(name, color)
#else
#define NVTX_RANGE(name)
#define NVTX_RANGE_COLOR(name, color)
#endif


// ============================================================================
// QuerySamplesCompletePool - Thread pool for calling QuerySamplesComplete
// ============================================================================

struct WorkItem
{
    std::vector<mlperf::QuerySampleResponse> responses;
};

class QuerySamplesCompletePool
{
public:
    QuerySamplesCompletePool(size_t numThreads, bool testMode = false)
        : mStopWork(false), mTestMode(testMode)
    {
        for (size_t i = 0; i < numThreads; ++i)
        {
            mThreads.emplace_back(&QuerySamplesCompletePool::HandleResult, this, i);
        }
    }

    ~QuerySamplesCompletePool()
    {
        {
            std::unique_lock<std::mutex> lock(mMtx);
            mStopWork = true;
            mCondVar.notify_all();
        }

        for (auto& t : mThreads)
        {
            t.join();
        }
    }

    void EnqueueBatch(
        const std::vector<mlperf::ResponseId>& queryIds,
        uintptr_t basePtr,
        size_t bytesPerElement)
    {
        NVTX_RANGE_COLOR("EnqueueBatch", 0xFF00FF00);  // Green
        
        WorkItem item;
        item.responses.reserve(queryIds.size());
        
        for (size_t i = 0; i < queryIds.size(); ++i)
        {
            item.responses.emplace_back(mlperf::QuerySampleResponse{
                queryIds[i],
                basePtr + i * bytesPerElement,
                bytesPerElement
            });
        }

        {
            std::unique_lock<std::mutex> lock(mMtx);
            mWorkQueue.push_back(std::move(item));
            mCondVar.notify_one();
        }
    }

    void EnqueueBatchAuto(
        const std::vector<mlperf::ResponseId>& queryIds,
        uintptr_t basePtr,
        size_t totalBytes)
    {
        if (queryIds.empty()) return;
        size_t bytesPerElement = totalBytes / queryIds.size();
        EnqueueBatch(queryIds, basePtr, bytesPerElement);
    }

    void Enqueue(uintptr_t ptr, size_t nResp)
    {
        mlperf::QuerySampleResponse* responses = reinterpret_cast<mlperf::QuerySampleResponse*>(ptr);
        
        WorkItem item;
        item.responses.assign(responses, responses + nResp);

        {
            std::unique_lock<std::mutex> lock(mMtx);
            mWorkQueue.push_back(std::move(item));
            mCondVar.notify_one();
        }
    }

    size_t GetQueueSize() const
    {
        std::unique_lock<std::mutex> lock(mMtx);
        return mWorkQueue.size();
    }

private:
    void HandleResult(int threadIdx)
    {
        while (true)
        {
            WorkItem item;
            {
                std::unique_lock<std::mutex> lock(mMtx);
                mCondVar.wait(lock, [&]() { return !mWorkQueue.empty() || mStopWork; });

                if (mStopWork && mWorkQueue.empty())
                {
                    break;
                }

                if (mWorkQueue.empty())
                {
                    continue;
                }

                item = std::move(mWorkQueue.front());
                mWorkQueue.pop_front();
            }

            if (mTestMode)
            {
                std::cout << "[Thread " << threadIdx << "] TEST MODE - processing " 
                          << item.responses.size() << " responses" << std::endl;
            }
            else
            {
                NVTX_RANGE_COLOR("Pool::QuerySamplesComplete", 0xFF0000FF);  // Blue
                mlperf::QuerySamplesComplete(item.responses.data(), item.responses.size());
            }
        }
    }

    std::vector<std::thread> mThreads;
    std::deque<WorkItem> mWorkQueue;
    mutable std::mutex mMtx;
    std::condition_variable mCondVar;
    bool mStopWork;
    bool mTestMode;
};


// ============================================================================
// Python bindings
// ============================================================================

PYBIND11_MODULE(TestPybind, m) {
    m.doc() = "Multi-threaded QuerySamplesComplete pool for MLPerf loadgen";

    // QuerySamplesCompletePool
    py::class_<QuerySamplesCompletePool>(m, "QuerySamplesCompletePool")
        .def(py::init<size_t, bool>(), 
             py::arg("num_threads"), 
             py::arg("test_mode") = false)
        .def("enqueue_batch", &QuerySamplesCompletePool::EnqueueBatch,
             py::arg("query_ids"), py::arg("base_ptr"), py::arg("bytes_per_element"),
             py::call_guard<py::gil_scoped_release>())
        .def("enqueue_batch_auto", &QuerySamplesCompletePool::EnqueueBatchAuto,
             py::arg("query_ids"), py::arg("base_ptr"), py::arg("total_bytes"),
             py::call_guard<py::gil_scoped_release>())
        .def("enqueue", &QuerySamplesCompletePool::Enqueue, 
             py::arg("ptr"), py::arg("n_resp"),
             py::call_guard<py::gil_scoped_release>())
        .def("queue_size", &QuerySamplesCompletePool::GetQueueSize);
}
