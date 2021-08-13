cp ~/mediapipe/selfie_segmentation_cpu ./bin && GLOG_logtostderr=1 ./bin/selfie_segmentation_cpu \
--calculator_graph_config_file=graph_config/selfie_segmentation.pbtxt \
# --input_video_path=input/IMG_1280.mp4 \
# --output_video_path=output/output.mp4