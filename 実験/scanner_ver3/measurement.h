#pragma once

#include <math.h>
#include "evaluation_data.h"

// このヘッダーはpredict()の定義後に読み込む。現在の非圧縮版用。
constexpr uint16_t WARMUP_COUNT = 100;
constexpr uint16_t INFERENCES_PER_SET = 1000;
constexpr uint8_t MEASUREMENT_SET_COUNT = 5;

static_assert(INPUT_SIZE == 3 && OUTPUT_SIZE == 3, "RGB input/output required");
static_assert(EVALUATION_PIXEL_COUNT == EVALUATION_WIDTH * EVALUATION_HEIGHT,
              "evaluation image size mismatch");
static_assert(sizeof(evaluationInputs) / sizeof(evaluationInputs[0]) ==
                  EVALUATION_PIXEL_COUNT,
              "evaluation input count mismatch");

// 各predict()の出力を計測区間内で集計し、計測後に表示する。
volatile float measurementSink = 0.0f;
uint8_t measurementImage[EVALUATION_PIXEL_COUNT][OUTPUT_SIZE] = {};

size_t parameterBytes() {
    return sizeof(weight_1) + sizeof(bias_1) +
           sizeof(weight_2) + sizeof(bias_2) +
           sizeof(weight_3) + sizeof(bias_3);
}

void printMeasurementInfo() {
    Serial.println("MEASUREMENT_INFO_BEGIN");
    Serial.print("hidden_size1="); Serial.println(HIDDEN_SIZE1);
    Serial.print("hidden_size2="); Serial.println(HIDDEN_SIZE2);
    Serial.println("storage=dense");
    Serial.println("pruning_threshold=none");
    Serial.print("float_bytes="); Serial.println(sizeof(float));
    Serial.print("parameter_bytes="); Serial.println(parameterBytes());
    Serial.println("compression_ratio_percent=100");
    Serial.print("evaluation_input_bytes="); Serial.println(sizeof(evaluationInputs));
    Serial.print("image_buffer_bytes="); Serial.println(sizeof(measurementImage));
    // predict()内の中間配列だけの計算値。総スタック量や総RAM量ではない。
    Serial.print("predict_intermediate_arrays_bytes=");
    Serial.println((HIDDEN_SIZE1 + HIDDEN_SIZE2) * sizeof(float));
    Serial.print("evaluation_width="); Serial.println(EVALUATION_WIDTH);
    Serial.print("evaluation_height="); Serial.println(EVALUATION_HEIGHT);
    Serial.print("evaluation_pixels="); Serial.println(EVALUATION_PIXEL_COUNT);
    Serial.print("evaluation_source_sha256="); Serial.println(EVALUATION_SOURCE_SHA256);
    Serial.println("input_order=row_major_RGB_normalized_by_255");
    Serial.println("output_conversion=clip_0_255_then_truncate");
    Serial.print("warmup_count="); Serial.println(WARMUP_COUNT);
    Serial.print("inferences_per_set="); Serial.println(INFERENCES_PER_SET);
    Serial.print("measurement_sets="); Serial.println(MEASUREMENT_SET_COUNT);
    Serial.println("MEASUREMENT_INFO_END");
}

void accumulateMeasurementOutput(const float output[OUTPUT_SIZE]) {
    for (uint8_t channel = 0; channel < OUTPUT_SIZE; ++channel) {
        measurementSink += output[channel];
    }
}

uint8_t measurementByte(float normalizedValue) {
    const float value = normalizedValue * 255.0f;
    if (value <= 0.0f) return 0;
    if (value >= 255.0f) return 255;
    return static_cast<uint8_t>(value);  // 既存スキャナーと同じ切り捨て。
}

bool createMeasurementImage() {
    float output[OUTPUT_SIZE];
    for (size_t pixel = 0; pixel < EVALUATION_PIXEL_COUNT; ++pixel) {
        predict(evaluationInputs[pixel], output);
        for (uint8_t channel = 0; channel < OUTPUT_SIZE; ++channel) {
            if (!isfinite(output[channel])) return false;
            measurementImage[pixel][channel] = measurementByte(output[channel]);
        }
    }
    return true;
}

void printMeasurementPpm() {
    Serial.println("PPM_BEGIN");
    Serial.println("P3");
    Serial.print(EVALUATION_WIDTH); Serial.print(' '); Serial.println(EVALUATION_HEIGHT);
    Serial.println("255");
    for (size_t pixel = 0; pixel < EVALUATION_PIXEL_COUNT; ++pixel) {
        for (uint8_t channel = 0; channel < OUTPUT_SIZE; ++channel) {
            Serial.print(measurementImage[pixel][channel]);
            Serial.print(' ');
        }
        if ((pixel + 1) % EVALUATION_WIDTH == 0) Serial.println();
    }
    Serial.println("PPM_END");
}

void runMeasurement() {
    uint32_t elapsedUs[MEASUREMENT_SET_COUNT];
    float checksums[MEASUREMENT_SET_COUNT];
    float output[OUTPUT_SIZE];
    // 先行する送信を処理する。以下、全セットの計時完了までSerial出力しない。
    Serial.flush();
    measurementSink = 0.0f;
    for (uint16_t i = 0; i < WARMUP_COUNT; ++i) {
        predict(evaluationInputs[i % EVALUATION_PIXEL_COUNT], output);
        accumulateMeasurementOutput(output);
    }
    const float warmupChecksum = measurementSink;

    for (uint8_t set = 0; set < MEASUREMENT_SET_COUNT; ++set) {
        measurementSink = 0.0f;
        const uint32_t start = static_cast<uint32_t>(micros());
        for (uint16_t i = 0; i < INFERENCES_PER_SET; ++i) {
            // 各セットとも入力0から同じ順序で繰り返す。
            predict(evaluationInputs[i % EVALUATION_PIXEL_COUNT], output);
            accumulateMeasurementOutput(output);
        }
        const uint32_t end = static_cast<uint32_t>(micros());
        // unsigned差分によりmicros()の1回の周回をまたいでも計算できる。
        elapsedUs[set] = end - start;
        checksums[set] = measurementSink;
    }

    if (!isfinite(warmupChecksum)) {
        Serial.println("MEASUREMENT_ERROR: non-finite warmup output");
        return;
    }
    for (uint8_t set = 0; set < MEASUREMENT_SET_COUNT; ++set) {
        if (!isfinite(checksums[set])) {
            Serial.println("MEASUREMENT_ERROR: non-finite timed output");
            return;
        }
    }
    // PPM生成・整数化・シリアル出力は計測区間外。
    if (!createMeasurementImage()) {
        Serial.println("MEASUREMENT_ERROR: non-finite image output");
        return;
    }

    printMeasurementInfo();
    Serial.print("warmup_checksum="); Serial.println(warmupChecksum, 6);
    Serial.println("set,elapsed_us,us_per_inference,checksum");
    float sumPerInferenceUs = 0.0f;
    for (uint8_t set = 0; set < MEASUREMENT_SET_COUNT; ++set) {
        const float perInferenceUs = static_cast<float>(elapsedUs[set]) /
                                     INFERENCES_PER_SET;
        sumPerInferenceUs += perInferenceUs;
        Serial.print(set + 1); Serial.print(',');
        Serial.print(elapsedUs[set]); Serial.print(',');
        Serial.print(perInferenceUs, 6); Serial.print(',');
        Serial.println(checksums[set], 6);
    }
    Serial.print("mean_inference_us=");
    Serial.println(sumPerInferenceUs / MEASUREMENT_SET_COUNT, 6);
    printMeasurementPpm();
    Serial.println("MEASUREMENT_COMPLETE");
}

void measurementSetup() {
    Serial.begin(9600);
    // 接続や起動だけでは推論・測定を開始しない。
}

void measurementLoop() {
    if (Serial.available() == 0) return;
    const char command = static_cast<char>(Serial.read());
    if (command == 'r' || command == 'R') {
        runMeasurement();
    } else if (command == 'i' || command == 'I') {
        printMeasurementInfo();
        Serial.println("Send r to run measurement.");
    }
}
