#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct WhiteCornerCandidate {
    // 保存一个疑似白色 L 形角点的轮廓、外接框和方向信息。
    std::vector<cv::Point> contour;
    cv::Rect bounding_box;
    int orientation = -1;
    cv::Point2f representative_point;
};

struct MarkerCandidate {
    // 保存由四个角点组成的标记边界及其几何评分。
    std::array<cv::Point2f, 4> corners;
    float score = 0.0F;
};

struct VideoMetadata {
    // 记录输入视频参数，输出视频需要复用这些参数。
    int width = 0;
    int height = 0;
    double fps = 0.0;
    int declared_frame_count = 0;
};

struct ProcessedFrame {
    // 保存单帧处理后的图像以及用于统计的检测数量。
    cv::Mat annotated;
    int white_corner_count = 0;
    int marker_count = 0;
};

bool parseArguments(int argc, char* argv[], std::string& input_path,
                    fs::path& output_path) {
    // 程序要求输入视频和输出视频两个命令行参数。
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input-video> <output-video>\n";
        return false;
    }

    // 读取路径，后续分别交给 VideoCapture 和 VideoWriter。
    input_path = argv[1];
    output_path = argv[2];
    return true;
}

bool readVideoMetadata(const cv::VideoCapture& capture,
                       VideoMetadata& metadata) {
    // 获取分辨率、帧率和声明帧数，保证输出视频能正确创建。
    metadata.width =
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    metadata.height =
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    metadata.fps = capture.get(cv::CAP_PROP_FPS);
    metadata.declared_frame_count =
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));

    return metadata.width > 0 && metadata.height > 0 && metadata.fps > 0.0;
}

bool createVideoWriter(const fs::path& output_path,
                       const VideoMetadata& metadata,
                       cv::VideoWriter& writer) {
    // 输出目录可能不存在，因此先创建目录再打开编码器。
    if (output_path.has_parent_path()) {
        fs::create_directories(output_path.parent_path());
    }

    // 使用 mp4v 编码，并保持输入视频的帧率、尺寸和彩色格式。
    writer.open(output_path.string(),
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                metadata.fps,
                cv::Size(metadata.width, metadata.height),
                true);
    return writer.isOpened();
}

bool isLShape(const std::vector<cv::Point>& contour,
              const cv::Rect& bounding_box, int& best_orientation,
              int& best_error) {
    // 将轮廓转换为外接框局部坐标，便于独立判断其形状。
    cv::Mat mask = cv::Mat::zeros(bounding_box.size(), CV_8UC1);
    std::vector<cv::Point> local_contour;
    local_contour.reserve(contour.size());
    for (const cv::Point& point : contour) {
        local_contour.emplace_back(point.x - bounding_box.x,
                                   point.y - bounding_box.y);
    }
    std::vector<std::vector<cv::Point>> contours{local_contour};
    cv::drawContours(mask, contours, 0, cv::Scalar(255), cv::FILLED);

    // 缩放到固定 8x8 网格，降低尺寸差异对形状比较的影响。
    cv::Mat normalized;
    cv::resize(mask, normalized, cv::Size(8, 8), 0.0, 0.0,
               cv::INTER_AREA);
    cv::threshold(normalized, normalized, 127.0, 1.0, cv::THRESH_BINARY);

    // 依次与四种旋转方向的 L 形模板比较，选择像素误差最小者。
    best_orientation = -1;
    best_error = std::numeric_limits<int>::max();
    for (int orientation = 0; orientation < 4; ++orientation) {
        int error = 0;
        for (int row = 0; row < 8; ++row) {
            for (int column = 0; column < 8; ++column) {
                // 模板由两条相邻的粗边组成，分别覆盖网格的三行或三列。
                const bool top = row < 3;
                const bool bottom = row >= 5;
                const bool left = column < 3;
                const bool right = column >= 5;
                bool expected = false;
                if (orientation == 0) {
                    expected = top || left;
                } else if (orientation == 1) {
                    expected = top || right;
                } else if (orientation == 2) {
                    expected = bottom || right;
                } else {
                    expected = bottom || left;
                }

                const bool actual = normalized.at<uchar>(row, column) != 0;
                if (actual != expected) {
                    ++error;
                }
            }
        }
        if (error < best_error) {
            best_error = error;
            best_orientation = orientation;
        }
    }

    // 误差不超过阈值才认为该轮廓确实接近 L 形角点。
    return best_error <= 18;
}

bool findWhiteCorner(const std::vector<cv::Point>& contour,
                     WhiteCornerCandidate& candidate) {
    // 先用面积、尺寸和填充率排除噪声及明显不符合的轮廓。
    const double area = cv::contourArea(contour);
    if (area < 40.0) {
        return false;
    }

    const cv::Rect bounding_box = cv::boundingRect(contour);
    if (bounding_box.width < 8 || bounding_box.height < 8) {
        return false;
    }

    const double fill_ratio =
        area / static_cast<double>(bounding_box.area());
    if (fill_ratio < 0.15 || fill_ratio > 0.85) {
        return false;
    }

    // 再通过模板匹配确认 L 形，并记录其朝向。
    int best_orientation = -1;
    int best_error = 0;
    if (!isLShape(contour, bounding_box, best_orientation, best_error)) {
        return false;
    }

    // 保存候选信息，并将 L 形缺口对应的外接框顶点作为代表点。
    candidate.contour = contour;
    candidate.bounding_box = bounding_box;
    candidate.orientation = best_orientation;
    if (best_orientation == 0) {
        candidate.representative_point =
            cv::Point2f(bounding_box.x, bounding_box.y);
    } else if (best_orientation == 1) {
        candidate.representative_point = cv::Point2f(
            bounding_box.x + bounding_box.width, bounding_box.y);
    } else if (best_orientation == 2) {
        candidate.representative_point = cv::Point2f(
            bounding_box.x + bounding_box.width,
            bounding_box.y + bounding_box.height);
    } else {
        candidate.representative_point = cv::Point2f(
            bounding_box.x, bounding_box.y + bounding_box.height);
    }
    return true;
}

std::vector<MarkerCandidate> findMarkers(
    const std::vector<WhiteCornerCandidate>& corners) {
    // 标记必须由四个不同方向的 L 形角点组成。
    std::vector<MarkerCandidate> candidates;
    if (corners.size() < 4) {
        return candidates;
    }

    // 遍历所有四点组合，选出几何形状最接近矩形的一组。
    float best_geometry_score = std::numeric_limits<float>::max();
    float best_average_area = 0.0F;
    std::array<int, 4> best_indices{-1, -1, -1, -1};
    for (std::size_t i = 0; i + 3 < corners.size(); ++i) {
        for (std::size_t j = i + 1; j + 2 < corners.size(); ++j) {
            for (std::size_t k = j + 1; k + 1 < corners.size(); ++k) {
                for (std::size_t l = k + 1; l < corners.size(); ++l) {
                    const std::array<std::size_t, 4> indices{i, j, k, l};
                    // 收集四个候选的代表点，并用轮廓面积作为平局时的依据。
                    std::array<cv::Point2f, 4> representative_points;
                    float area_sum = 0.0F;
                    for (std::size_t n = 0; n < indices.size(); ++n) {
                        const auto& corner = corners[indices[n]];
                        representative_points[n] =
                            corner.representative_point;
                        const float area =
                            static_cast<float>(cv::contourArea(
                                corner.contour));
                        area_sum += area;
                    }

                    // 每个方向只能出现一次，否则不能组成完整标记。
                    std::array<int, 4> orientation_indices{-1, -1, -1, -1};
                    for (std::size_t n = 0; n < indices.size(); ++n) {
                        const int orientation =
                            corners[indices[n]].orientation;
                        if (orientation < 0 || orientation >= 4 ||
                            orientation_indices[orientation] >= 0) {
                            orientation_indices = {-1, -1, -1, -1};
                            break;
                        }
                        orientation_indices[orientation] =
                            static_cast<int>(n);
                    }
                    if (std::any_of(
                            orientation_indices.begin(),
                            orientation_indices.end(),
                            [](int value) { return value < 0; })) {
                        continue;
                    }

                    // 按方向固定顺序：左上、右上、右下、左下。
                    std::array<cv::Point2f, 4> ordered_points;
                    for (std::size_t orientation = 0; orientation < 4;
                         ++orientation) {
                        ordered_points[orientation] =
                            representative_points[static_cast<std::size_t>(
                                orientation_indices[orientation])];
                    }

                    // 用四点平均值估计中心，检查四个点是否位于对应象限。
                    cv::Point2f center(0.0F, 0.0F);
                    for (const auto& point : ordered_points) {
                        center += point;
                    }
                    center *= 0.25F;

                    if (!(ordered_points[0].x < center.x &&
                          ordered_points[0].y < center.y &&
                          ordered_points[1].x > center.x &&
                          ordered_points[1].y < center.y &&
                          ordered_points[2].x > center.x &&
                          ordered_points[2].y > center.y &&
                          ordered_points[3].x < center.x &&
                          ordered_points[3].y > center.y)) {
                        continue;
                    }

                    // 检查边长比例，过滤过于细长或组合错误的四边形。
                    const auto edge_length = [&ordered_points](
                                                 std::size_t from,
                                                 std::size_t to) {
                        return cv::norm(ordered_points[to] -
                                        ordered_points[from]);
                    };
                    const float top = edge_length(0, 1);
                    const float right = edge_length(1, 2);
                    const float bottom = edge_length(2, 3);
                    const float left = edge_length(3, 0);
                    const float shortest_edge =
                        std::min({top, right, bottom, left});
                    const float longest_edge =
                        std::max({top, right, bottom, left});
                    if (shortest_edge <= 0.0F ||
                        longest_edge / shortest_edge > 3.0F) {
                        continue;
                    }

                    // 通过连续边叉积符号判断四边形是否凸且顶点顺序正确。
                    float cross_sign = 0.0F;
                    bool convex = true;
                    for (std::size_t n = 0; n < ordered_points.size(); ++n) {
                        const auto& a = ordered_points[n];
                        const auto& b = ordered_points[(n + 1) % 4];
                        const auto& c = ordered_points[(n + 2) % 4];
                        const cv::Point2f ab = b - a;
                        const cv::Point2f bc = c - b;
                        const float cross = ab.x * bc.y - ab.y * bc.x;
                        if (std::abs(cross) < 1.0F) {
                            convex = false;
                            break;
                        }
                        if (n == 0) {
                            cross_sign = cross;
                        } else if (cross * cross_sign < 0.0F) {
                            convex = false;
                            break;
                        }
                    }
                    if (!convex) {
                        continue;
                    }

                    // 对边越接近等长，几何误差越小，越可能是真实矩形标记。
                    const float opposite_edge_error =
                        std::abs(top - bottom) / std::max(top, bottom) +
                        std::abs(left - right) / std::max(left, right);
                    const float geometry_score = opposite_edge_error;
                    const float average_area = area_sum * 0.25F;
                    if (geometry_score < best_geometry_score ||
                        (std::abs(geometry_score - best_geometry_score) <
                             1e-3F &&
                         average_area > best_average_area)) {
                        best_geometry_score = geometry_score;
                        best_average_area = average_area;
                        for (std::size_t n = 0; n < indices.size(); ++n) {
                            best_indices[n] =
                                static_cast<int>(indices[n]);
                        }
                    }
                }
            }
        }
    }

    if (best_indices[0] < 0) {
        return candidates;
    }

    // 合并最佳四个角点的轮廓，用最小外接旋转矩形得到标记边界。
    std::vector<cv::Point> all_points;
    for (const int index : best_indices) {
        const auto& corner = corners[static_cast<std::size_t>(index)];
        all_points.insert(all_points.end(), corner.contour.begin(),
                          corner.contour.end());
    }

    const cv::RotatedRect marker_box = cv::minAreaRect(all_points);
    MarkerCandidate marker;
    marker_box.points(marker.corners.data());
    // 将旋转矩形的四个顶点整理成顺时针的左上、右上、右下、左下顺序。
    std::array<cv::Point2f, 4> ordered = marker.corners;
    std::sort(ordered.begin(), ordered.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) {
                  return a.y == b.y ? a.x < b.x : a.y < b.y;
              });
    if (ordered[0].x > ordered[1].x) {
        std::swap(ordered[0], ordered[1]);
    }
    if (ordered[2].x > ordered[3].x) {
        std::swap(ordered[2], ordered[3]);
    }
    marker.corners = {ordered[0], ordered[1], ordered[3], ordered[2]};
    marker.score = 0.0F;
    candidates.push_back(marker);
    return candidates;
}

void drawMarker(cv::Mat& image, const MarkerCandidate& marker) {
    // 在原始帧上绘制绿色边界和红色角点，形成可视化检测结果。
    for (std::size_t i = 0; i < marker.corners.size(); ++i) {
        const cv::Point start(cvRound(marker.corners[i].x),
                              cvRound(marker.corners[i].y));
        const auto& next = marker.corners[(i + 1) % marker.corners.size()];
        const cv::Point end(cvRound(next.x), cvRound(next.y));
        cv::line(image, start, end, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::circle(image, start, 4, cv::Scalar(0, 0, 255), cv::FILLED,
                   cv::LINE_AA);
    }
}

ProcessedFrame processFrame(const cv::Mat& frame) {
    // 单帧处理流程：灰度化、二值化、形态学去噪、轮廓检测和标记验证。
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 提取亮度较高的区域，白色角点会因此成为前景。
    cv::Mat binary;
    cv::threshold(gray, binary, 180.0, 255.0, cv::THRESH_BINARY);

    // 闭运算连接小间隙，使角点轮廓更完整。
    const cv::Mat close_kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat morphology;
    cv::morphologyEx(binary, morphology, cv::MORPH_CLOSE, close_kernel);

    // 开运算去除小噪声，减少无关轮廓进入后续检测。
    const cv::Mat open_kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(morphology, morphology, cv::MORPH_OPEN, open_kernel);

    // findContours 会修改输入图像，因此使用副本保护形态学结果。
    cv::Mat contour_input = morphology.clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(contour_input, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    // 对每个外部轮廓执行面积、形状和方向筛选。
    std::vector<WhiteCornerCandidate> corners;
    for (const auto& contour : contours) {
        WhiteCornerCandidate candidate;
        if (findWhiteCorner(contour, candidate)) {
            corners.push_back(candidate);
        }
    }

    // 从原始彩色帧开始绘制，保留原始画面供输出视频观看。
    ProcessedFrame result;
    result.annotated = frame.clone();
    for (const auto& candidate : corners) {
        cv::rectangle(result.annotated, candidate.bounding_box,
                      cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        ++result.white_corner_count;
    }

    // 只有四个角点通过整体几何验证后，才绘制完整标记。
    const std::vector<MarkerCandidate> markers = findMarkers(corners);
    if (!markers.empty()) {
        drawMarker(result.annotated, markers.front());
        result.marker_count = 1;
    }
    return result;
}

bool processVideo(cv::VideoCapture& capture, cv::VideoWriter& writer,
                  int& decoded_frame_count, int& last_white_corner_count,
                  int& detected_frame_count, int& last_marker_count) {
    // 持续读取、处理并写回每一帧，直到视频读取结束。
    cv::Mat frame;
    while (capture.read(frame)) {
        if (frame.empty()) {
            std::cerr << "Error: an empty frame was read.\n";
            return false;
        }

        // 对当前帧执行检测，并将带标注的结果写入输出视频。
        ProcessedFrame result = processFrame(frame);
        writer.write(result.annotated);
        last_white_corner_count = result.white_corner_count;
        last_marker_count = result.marker_count;
        // 统计检测到标记的帧数，以及最后一帧的检测数量。
        if (result.marker_count > 0) {
            ++detected_frame_count;
        }
        ++decoded_frame_count;
    }
    return true;
}

void printSummary(const std::string& input_path,
                  const fs::path& output_path,
                  const VideoMetadata& metadata,
                  int decoded_frame_count,
                  int last_white_corner_count, int detected_frame_count,
                  int last_marker_count) {
    // 输出处理概况，便于确认输入、输出和检测过程是否正常。
    std::cout << "Input: " << input_path << '\n'
              << "Output: " << output_path << '\n'
              << "Resolution: " << metadata.width << " x "
              << metadata.height << '\n'
              << "FPS: " << metadata.fps << '\n'
              << "Declared frames: " << metadata.declared_frame_count << '\n'
              << "Decoded frames: " << decoded_frame_count << '\n'
              << "Last-frame white-corner candidates: "
              << last_white_corner_count << '\n'
              << "Frames with a marker candidate: " << detected_frame_count
              << '\n'
              << "Last-frame marker candidates: " << last_marker_count
              << '\n';
}

int main(int argc, char* argv[]) {
    // 主流程按“解析参数 -> 打开输入 -> 创建输出 -> 处理视频 -> 汇总”的顺序执行。
    std::string input_path;
    fs::path output_path;
    if (!parseArguments(argc, argv, input_path, output_path)) {
        return EXIT_FAILURE;
    }

    // 打开输入视频，失败时立即终止，避免后续使用无效句柄。
    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cerr << "Error: cannot open video: " << input_path << '\n';
        return EXIT_FAILURE;
    }

    // 读取并校验视频元数据，输出文件需要使用有效的尺寸和帧率。
    VideoMetadata metadata;
    if (!readVideoMetadata(capture, metadata)) {
        std::cerr << "Error: invalid video metadata.\n";
        return EXIT_FAILURE;
    }

    // 创建输出视频写入器。
    cv::VideoWriter writer;
    if (!createVideoWriter(output_path, metadata, writer)) {
        std::cerr << "Error: cannot create output video: "
                  << output_path << '\n';
        return EXIT_FAILURE;
    }

    // 初始化处理统计量，并开始逐帧处理。
    int decoded_frame_count = 0;
    int last_white_corner_count = 0;
    int detected_frame_count = 0;
    int last_marker_count = 0;
    if (!processVideo(capture, writer, decoded_frame_count,
                      last_white_corner_count, detected_frame_count,
                      last_marker_count)) {
        return EXIT_FAILURE;
    }

    printSummary(input_path, output_path, metadata, decoded_frame_count,
                 last_white_corner_count, detected_frame_count,
                 last_marker_count);
    return EXIT_SUCCESS;
}
