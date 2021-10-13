# srcdir=mediapipe/mediapipe/util/tflite/operations
# destdir=tensorflow/tensorflow/lite/kernels

# cp ${srcdir}/max_pool_argmax.cc ${destdir} && \
# cp ${srcdir}/max_pool_argmax.h ${destdir} && \
# cp ${srcdir}/max_unpooling.cc ${destdir} && \
# cp ${srcdir}/max_unpooling.h ${destdir} && \
# cp ${srcdir}/transpose_conv_bias.cc ${destdir} && \
# cp ${srcdir}/transpose_conv_bias.h ${destdir}


git clone -b v2.4.0 https://github.com/tensorflow/tensorflow.git
git clone -b 0.8.2 https://github.com/google/mediapipe.git

cp changeFiles/* tensorflow/tensorflow/lite/kernels

cd tensorflow && \
bazel build -c opt --macos_minimum_os=10.13 --action_env MACOSX_DEPLOYMENT_TARGET=10.13 --define tflite_with_xnnpack=true //tensorflow/lite/c:libtensorflowlite_c.dylib && \
cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.dylib ../
