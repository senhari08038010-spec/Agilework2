#pragma once

// train_python/test_week3.ppmから固定。左上から行順、RGB順、各値/255。
// 元ファイルを編集しても自動では更新されない。
constexpr size_t EVALUATION_WIDTH = 3;
constexpr size_t EVALUATION_HEIGHT = 3;
constexpr size_t EVALUATION_PIXEL_COUNT = 9;
const char EVALUATION_SOURCE_SHA256[] = "bc140cf6e6702f818bc2ec8c41fd33fca109a606c667ccfb8a853691c1b2c4bd";
const float evaluationInputs[EVALUATION_PIXEL_COUNT][3] = {
    {114.0f / 255.0f, 30.0f / 255.0f, 81.0f / 255.0f},
    {102.0f / 255.0f, 194.0f / 255.0f, 81.0f / 255.0f},
    {25.0f / 255.0f, 37.0f / 255.0f, 23.0f / 255.0f},
    {255.0f / 255.0f, 253.0f / 255.0f, 255.0f / 255.0f},
    {102.0f / 255.0f, 213.0f / 255.0f, 115.0f / 255.0f},
    {51.0f / 255.0f, 37.0f / 255.0f, 34.0f / 255.0f},
    {0.0f / 255.0f, 1.0f / 255.0f, 0.0f / 255.0f},
    {38.0f / 255.0f, 77.0f / 255.0f, 34.0f / 255.0f},
    {255.0f / 255.0f, 255.0f / 255.0f, 255.0f / 255.0f},
};
