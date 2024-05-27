#include <opencv2/opencv.hpp>
#include <mediapipe/framework/formats/landmark.pb.h>
#include <mediapipe/framework/port/opencv_core_inc.h>
#include <mediapipe/framework/port/opencv_highgui_inc.h>
#include <mediapipe/framework/port/parse_text_proto.h>
#include <mediapipe/framework/port/status.h>
#include <mediapipe/graphs/hand_tracking/hand_tracking_gpu.pb.h>
#include <mediapipe/framework/calculator.pb.h>
#include <mediapipe/framework/deps/status_builder.h>
#include <mediapipe/framework/port/logging.h>
#include <mediapipe/framework/port/status.h>

#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

bool power_on = false;
float initial_distance = -1;

float calculate_distance(float x1, float y1, float x2, float y2) {
    return hypot(x2 - x1, y2 - y1);
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera." << endl;
        return -1;
    }

    // Mediapipe 초기화
    mediapipe::CalculatorGraph graph;
    std::string calculator_graph_config_contents;
    mediapipe::Status run_status = mediapipe::ParseTextProtoOrDie(mediapipe::HandTrackingGpu())
        .SerializeToString(&calculator_graph_config_contents);
    run_status = graph.Initialize(calculator_graph_config_contents);
    if (!run_status.ok()) {
        cerr << "Error: " << run_status.message() << endl;
        return -1;
    }

    // Start running the graph.
    run_status = graph.StartRun({});
    if (!run_status.ok()) {
        cerr << "Error: " << run_status.message() << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Convert to RGB
        cvtColor(frame, frame, COLOR_BGR2RGB);

        // Wrapping Mat into an ImageFrame
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, frame.cols, frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        frame.copyTo(input_frame_mat);

        // Send image packet into the graph.
        size_t frame_timestamp_us = cv::getTickCount() / cv::getTickFrequency() * 1e6;
        run_status = graph.AddPacketToInputStream(
            "input_video", mediapipe::Adopt(input_frame.release())
                               .At(mediapipe::Timestamp(frame_timestamp_us)));
        if (!run_status.ok()) {
            cerr << "Error: " << run_status.message() << endl;
            break;
        }

        // Get hand landmarks.
        mediapipe::Packet packet;
        if (graph.GetOutputStream("hand_landmarks").Next(&packet)) {
            auto& output_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
            if (output_landmarks.landmark_size() > 0) {
                const auto& thumb_tip = output_landmarks.landmark(4);
                const auto& index_tip = output_landmarks.landmark(8);
                const auto& middle_tip = output_landmarks.landmark(12);
                const auto& ring_tip = output_landmarks.landmark(16);

                float thumb_index_dist = calculate_distance(thumb_tip.x(), thumb_tip.y(), index_tip.x(), index_tip.y());
                float thumb_middle_ring_dist = calculate_distance(thumb_tip.x(), thumb_tip.y(), (middle_tip.x() + ring_tip.x()) / 2, (middle_tip.y() + ring_tip.y()) / 2);

                // Power on/off gesture (thumb and index finger)
                if (thumb_index_dist < 0.05) {
                    power_on = !power_on;
                    initial_distance = -1;
                    cout << "Power " << (power_on ? "On" : "Off") << endl;
                }

                // Volume control gesture (thumb, middle finger, ring finger)
                if (power_on && thumb_middle_ring_dist < 0.1) {
                    if (initial_distance < 0) {
                        initial_distance = thumb_middle_ring_dist;
                    } else {
                        if (thumb_middle_ring_dist > initial_distance) {
                            cout << "Volume Up" << endl;
                        } else if (thumb_middle_ring_dist < initial_distance) {
                            cout << "Volume Down" << endl;
                        }
                    }
                }
            }
        }

        // Show the image in the window
        cv::imshow("Hand Tracking", frame);
        if (cv::waitKey(5) >= 0) break;
    }

    // Cleanup
    graph.CloseInputStream("input_video").IgnoreError();
    graph.WaitUntilDone().IgnoreError();
    cap.release();
    destroyAllWindows();
    return 0;
}
