/**
 * @file teleop_latency_logger.hpp
 * @brief Lightweight CSV logger for teleop ZMQ playback latency.
 *
 * Records, once per merged ZMQ pose message, how far the playback cursor is
 * behind the newest received mocap frame (the dominant, variable component of
 * teleop latency).  Rows are buffered in memory under a dedicated mutex and
 * flushed to disk periodically so file I/O never sits on the 50 Hz / 100 Hz
 * control/input critical paths.
 *
 * Output: <dir>/teleop_latency_<YYYYMMDD_HHMMSS>.csv
 * Columns:
 *   wall_time_ms          system_clock timestamp (ms) for human correlation
 *   monotonic_ms          steady_clock timestamp (ms)
 *   msg_interarrival_ms   ms since the previous logged merge (~ message period)
 *   newest_frame_index    global index of the newest buffered frame
 *   playback_frame        global index currently at the playback cursor
 *   latency_frames        newest_frame_index - playback_frame (current-rate frames)
 *   latency_ms            latency_frames * control_period_ms
 *   did_catchup           1 if this merge triggered a catch-up reset, else 0
 *   frame_step            detected stride between consecutive frame indices
 */

#ifndef TELEOP_LATENCY_LOGGER_HPP
#define TELEOP_LATENCY_LOGGER_HPP

#include <sys/stat.h>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>

class TeleopLatencyLogger {
public:
    struct Row {
        double wall_time_ms = 0.0;
        double monotonic_ms = 0.0;
        double msg_interarrival_ms = 0.0;
        long newest_frame_index = 0;
        long playback_frame = 0;
        long latency_frames = 0;
        double latency_ms = 0.0;
        int did_catchup = 0;
        int frame_step = 1;
    };

    /// @param dir          Output directory (created if missing).
    /// @param flush_every  Buffered rows before a disk flush (~1 s at 50 msg/s).
    /// @param summary_every Rows between 1-line stdout summaries (0 = disabled).
    explicit TeleopLatencyLogger(const std::string& dir = "logs",
                                 int flush_every = 50,
                                 int summary_every = 50)
        : flush_every_(flush_every > 0 ? flush_every : 1),
          summary_every_(summary_every) {
        ::mkdir(dir.c_str(), 0755);  // ignore errors (e.g. already exists)
        path_ = dir + "/teleop_latency_" + Timestamp() + ".csv";
        file_.open(path_, std::ios::out);
        if (file_.is_open()) {
            file_ << "wall_time_ms,monotonic_ms,msg_interarrival_ms,newest_frame_index,"
                     "playback_frame,latency_frames,latency_ms,did_catchup,frame_step\n";
            file_.flush();
            std::cout << "[TeleopLatencyLogger] Logging teleop latency to " << path_ << std::endl;
        } else {
            std::cerr << "[TeleopLatencyLogger] Failed to open " << path_
                      << " (latency CSV disabled)" << std::endl;
        }
    }

    ~TeleopLatencyLogger() {
        Flush();
        if (file_.is_open()) file_.close();
    }

    /// Append one latency sample.  Cheap: in-memory append + periodic flush.
    void Record(const Row& r) {
        std::lock_guard<std::mutex> lock(mtx_);
        buffer_.push_back(r);

        // Running summary stats.
        sum_latency_ms_ += r.latency_ms;
        if (r.latency_ms > max_latency_ms_) max_latency_ms_ = r.latency_ms;
        if (r.latency_ms < min_latency_ms_) min_latency_ms_ = r.latency_ms;
        if (r.did_catchup) catchup_count_++;
        n_since_summary_++;

        if (static_cast<int>(buffer_.size()) >= flush_every_) FlushLocked();

        if (summary_every_ > 0 && n_since_summary_ >= summary_every_) {
            std::cout << "[teleop_latency] avg="
                      << std::fixed << std::setprecision(1)
                      << (sum_latency_ms_ / n_since_summary_) << "ms"
                      << " min=" << min_latency_ms_ << "ms"
                      << " max=" << max_latency_ms_ << "ms"
                      << " catchups=" << catchup_count_
                      << " (last " << n_since_summary_ << " msgs)" << std::endl;
            n_since_summary_ = 0;
            sum_latency_ms_ = 0.0;
            max_latency_ms_ = -1e30;
            min_latency_ms_ = 1e30;
            catchup_count_ = 0;
        }
    }

    void Flush() {
        std::lock_guard<std::mutex> lock(mtx_);
        FlushLocked();
    }

    const std::string& path() const { return path_; }

private:
    void FlushLocked() {
        if (!file_.is_open()) {
            buffer_.clear();
            return;
        }
        for (const auto& r : buffer_) {
            file_ << std::fixed << std::setprecision(3)
                  << r.wall_time_ms << ',' << r.monotonic_ms << ',' << r.msg_interarrival_ms << ','
                  << r.newest_frame_index << ',' << r.playback_frame << ',' << r.latency_frames << ','
                  << r.latency_ms << ',' << r.did_catchup << ',' << r.frame_step << '\n';
        }
        file_.flush();
        buffer_.clear();
    }

    static std::string Timestamp() {
        std::time_t t = std::time(nullptr);
        std::tm tm_buf{};
        localtime_r(&t, &tm_buf);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
        return std::string(buf);
    }

    std::ofstream file_;
    std::string path_;
    std::mutex mtx_;
    std::vector<Row> buffer_;
    int flush_every_;
    int summary_every_;

    // Summary accumulators.
    int n_since_summary_ = 0;
    double sum_latency_ms_ = 0.0;
    double max_latency_ms_ = -1e30;
    double min_latency_ms_ = 1e30;
    int catchup_count_ = 0;
};

#endif  // TELEOP_LATENCY_LOGGER_HPP
